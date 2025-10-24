import asyncio
import json
import logging
import uuid
import os
import time
from typing import Any, Dict, Iterable, Optional

try:
    from kafka import KafkaProducer  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    KafkaProducer = None  # type: ignore


from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

KAFKA_MESSAGES = Counter("kafka_producer_messages_total", "Kafka messages produced")
KAFKA_PAYLOAD_BYTES = Histogram(
    "kafka_producer_payload_bytes",
    "Payload size of produced messages",
    buckets=[256, 512, 1024, 2048, 4096, 8192, 16384, 65536],
)


class TransactionProducer:
    def __init__(self, brokers: Iterable[str]):
        self._brokers = list(brokers)
        self._producer: Optional[KafkaProducer] = None

    def _init_producer(self) -> None:
        if KafkaProducer is None:
            raise RuntimeError("kafka-python n'est pas disponible dans cet environnement")
        if self._producer is None:
            import os

            logger.info("Initialisation du producteur Kafka", extra={"brokers": self._brokers})
            kwargs = dict(
                bootstrap_servers=self._brokers,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                key_serializer=lambda key: key.encode("utf-8") if key else None,
                request_timeout_ms=2000,
                api_version_auto_timeout_ms=2000,
            )
            linger_ms = int(os.getenv("KAFKA_LINGER_MS", "10"))
            batch_bytes = int(os.getenv("KAFKA_BATCH_BYTES", "16384"))
            kwargs.update(linger_ms=linger_ms, batch_size=batch_bytes, acks="all", retries=5)
            if os.getenv("KAFKA_SSL", os.getenv("ENABLE_MTLS", "false")).strip().lower() in {"1", "true", "yes"}:
                kwargs.update(
                    security_protocol="SSL",
                    ssl_cafile=os.getenv("TLS_CA_PATH"),
                    ssl_certfile=os.getenv("TLS_CERT_PATH"),
                    ssl_keyfile=os.getenv("TLS_KEY_PATH"),
                )
            self._producer = KafkaProducer(**kwargs)

    def send_transaction(self, topic: str, payload: Dict[str, Any]) -> str:
        self._init_producer()
        assert self._producer is not None

        message_id = payload.get("message_id", str(uuid.uuid4()))
        envelope = {"message_id": message_id, **payload}

        logger.debug("Envoi d'une transaction dans Kafka", extra={"topic": topic, "message_id": message_id})
        payload_bytes = len(json.dumps(envelope).encode("utf-8"))
        KAFKA_PAYLOAD_BYTES.observe(payload_bytes)
        def _to_dlq(exc: BaseException) -> None:
            try:
                dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", f"{topic}.dlq")
                err_env = {**envelope, "dlq_reason": str(exc), "dlq_ts": time.time()}
                self._producer.send(dlq_topic, value=err_env, key=message_id)
                logger.error("Kafka send failed; sent to DLQ", extra={"topic": topic, "dlq": dlq_topic, "message_id": message_id})
            except Exception:
                logger.exception("Kafka DLQ send failed")

        future = self._producer.send(topic, value=envelope, key=message_id)
        future.add_errback(_to_dlq)
        self._producer.flush()
        KAFKA_MESSAGES.inc()
        return message_id

    async def send_transaction_async(self, topic: str, payload: Dict[str, Any]) -> str:
        return await asyncio.to_thread(self.send_transaction, topic, payload)
