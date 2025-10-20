FROM apache/spark:3.5.0

COPY jobs /opt/jobs

CMD ["/opt/spark/bin/spark-submit", "--master", "local[2]", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0", "/opt/jobs/compute_features.py"]
