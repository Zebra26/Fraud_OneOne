"""Kafka consumer that routes transactions to inference API with low latency."""

import json
import logging
import os
from typing import Any, Dict

import requests
from kafka import KafkaConsumer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_consumer() -> KafkaConsumer:
    brokers = os.getenv("KAFKA_BROKERS", "kafka:9092")
    topic = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=brokers.split(","),
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
        enable_auto_commit=False,
        security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
        api_version=(2, 8, 0),
    )
    return consumer


def call_inference(transaction: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = os.getenv("INFERENCE_ENDPOINT", "http://ml-inference:8080/infer")
    payload = {"transaction_id": transaction["transaction_id"], "features": transaction.get("features", {})}
    resp = requests.post(endpoint, json=payload, timeout=0.2)
    resp.raise_for_status()
    body = resp.json()
    body["latency_target_ms"] = "50-100"
    return body


def main() -> None:
    consumer = build_consumer()
    for message in consumer:
        transaction = message.value
        try:
            inference = call_inference(transaction)
            logger.info(
                "Transaction scorée",
                extra={
                    "transaction_id": transaction.get("transaction_id"),
                    "fraud_probability": inference.get("fraud_probability"),
                },
            )
            consumer.commit()
        except Exception as exc:  # pragma: no cover - network path
            logger.exception("Erreur lors du scoring", exc_info=exc)


if __name__ == "__main__":
    main()

