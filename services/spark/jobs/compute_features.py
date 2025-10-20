"""Spark job for feature engineering on streaming transaction data."""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    from_json,
    hour,
    lag,
    stddev_pop,
    struct,
    sum as spark_sum,
    to_json,
    when,
    window,
)
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType


def build_spark(app_name: str = "fraud-feature-engineering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.streaming.backpressure.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


transaction_schema = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("account_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("channel", StringType()),
        StructField("device_id", StringType()),
        StructField("geolocation", StructType([StructField("lat", DoubleType()), StructField("lon", DoubleType())])),
        StructField("behavior_score", DoubleType()),
        StructField("is_night", IntegerType()),
        StructField("velocity", DoubleType()),
    ]
)


def main():
    spark = build_spark()

    kafka_bootstrap = spark.sparkContext.getConf().get("spark.kafka.bootstrap.servers", "kafka:9092")
    input_topic = spark.sparkContext.getConf().get("spark.kafka.transactions.topic", "transactions")
    output_topic = spark.sparkContext.getConf().get("spark.kafka.features.topic", "transactions.features")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", input_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw.select(from_json(col("value").cast("string"), transaction_schema).alias("payload")).select("payload.*")

    parsed = parsed.withColumn("ts_long", col("timestamp").cast("long"))

    account_window = Window.partitionBy("account_id").orderBy("ts_long").rowsBetween(-10, 0)
    range_window = Window.partitionBy("account_id").orderBy("ts_long").rangeBetween(-300, 0)

    enriched = (
        parsed.withColumn("amount_avg", avg("amount").over(account_window))
        .withColumn("amount_std", stddev_pop("amount").over(account_window))
        .withColumn(
            "amount_z",
            when(col("amount_std") > 0, (col("amount") - col("amount_avg")) / col("amount_std")).otherwise(0.0),
        )
        .withColumn("previous_device", lag("device_id").over(account_window))
        .withColumn(
            "new_device",
            when(col("previous_device").isNull(), 0)
            .when(col("previous_device") != col("device_id"), 1)
            .otherwise(0),
        )
        .withColumn("hour_of_day", hour(col("timestamp")))
        .withColumn(
            "night_activity",
            when((col("hour_of_day") < 6) | (col("hour_of_day") > 22), 1).otherwise(0),
        )
        .withWatermark("timestamp", "2 minutes")
        .withColumn("rolling_amount_5m", spark_sum("amount").over(range_window))
        .withColumn("amount_ratio", col("amount") / (col("rolling_amount_5m") + 1e-3))
    )

    windowed = (
        enriched.groupBy(
            window("timestamp", "60 seconds", "10 seconds"),
            "account_id",
        )
        .agg(expr("avg(amount) as avg_amount"), expr("stddev_pop(amount) as std_amount"), expr("max(amount) as max_amount"))
        .select(
            col("window.end").alias("feature_generation_time"),
            "account_id",
            "avg_amount",
            "std_amount",
            "max_amount",
        )
    )

    joined = enriched.join(windowed, on="account_id", how="left")

    feature_vector = joined.select(
        "transaction_id",
        "account_id",
        to_json(
            struct(
                col("amount").alias("amount"),
                col("velocity"),
                col("night_activity").alias("is_night"),
                col("amount_ratio"),
                col("new_device"),
                col("behavior_score"),
                col("amount_z"),
            )
        ).alias("feature_vector"),
        col("timestamp").alias("event_time"),
    )

    query = (
        feature_vector.writeStream.outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("topic", output_topic)
        .option("checkpointLocation", "/opt/checkpoints/feature-job")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
