from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from prometheus_client import Counter, Histogram

try:
    from kafka import KafkaConsumer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    KafkaConsumer = None  # type: ignore


logger = logging.getLogger(__name__)

KAFKA_CONSUMED_MESSAGES = Counter("kafka_consumer_messages_total", "Kafka messages consumed")
KAFKA_CONSUMER_BATCHES = Counter("kafka_consumer_batches_total", "Kafka batches processed")
KAFKA_CONSUMER_BATCH_SIZE = Histogram(
    "kafka_consumer_batch_size",
    "Kafka consumer batch sizes",
    buckets=[10, 50, 100, 200, 500, 1000, 2000],
)


class KafkaBatchConsumer:
    """Lightweight micro-batching Kafka consumer template.

    Accumulates messages up to `batch_size` or `max_wait_ms` then invokes `process_fn`.
    SSL/mTLS picks up settings from TLS_* envs when KAFKA_SSL/ENABLE_MTLS are true.
    """

    def __init__(
        self,
        topics: Iterable[str],
        brokers: Iterable[str],
        group_id: str,
        *,
        batch_size: int = 500,
        max_wait_ms: int = 200,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
    ) -> None:
        if KafkaConsumer is None:
            raise RuntimeError("kafka-python is not available")

        self._topics = list(topics)
        self._brokers = list(brokers)
        self._group_id = group_id
        self._batch_size = max(1, batch_size)
        self._max_wait = max(50, max_wait_ms) / 1000.0

        kwargs: Dict[str, Any] = dict(
            bootstrap_servers=self._brokers,
            group_id=self._group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            key_deserializer=lambda b: b.decode("utf-8") if b else None,
            consumer_timeout_ms=1000,
        )
        if os.getenv("KAFKA_SSL", os.getenv("ENABLE_MTLS", "false")).strip().lower() in {"1", "true", "yes"}:
            kwargs.update(
                security_protocol="SSL",
                ssl_cafile=os.getenv("TLS_CA_PATH"),
                ssl_certfile=os.getenv("TLS_CERT_PATH"),
                ssl_keyfile=os.getenv("TLS_KEY_PATH"),
            )
        self._consumer = KafkaConsumer(**kwargs)
        self._consumer.subscribe(self._topics)

    def _current_lag(self) -> int:
        try:
            parts = list(self._consumer.assignment())
            if not parts:
                return 0
            end_offsets = self._consumer.end_offsets(parts)
            lag = 0
            for tp in parts:
                pos = self._consumer.position(tp)
                end = end_offsets.get(tp, pos)
                lag += max(0, end - pos)
            return int(lag)
        except Exception:
            return 0

    def run(self, process_fn: Callable[[List[Dict[str, Any]]], None]) -> None:
        buf: List[Dict[str, Any]] = []
        last = time.perf_counter()
        try:
            while True:
                # backpressure: pause/resume based on lag
                max_lag = int(os.getenv("CONSUMER_MAX_LAG", "50000"))
                lag = self._current_lag()
                if lag > max_lag:
                    try:
                        self._consumer.pause(*list(self._consumer.assignment()))
                    except Exception:
                        pass
                else:
                    try:
                        self._consumer.resume(*list(self._consumer.assignment()))
                    except Exception:
                        pass
                for msg in self._consumer.poll(timeout_ms=100, max_records=self._batch_size).values():
                    for record in msg:
                        buf.append(record.value)
                        KAFKA_CONSUMED_MESSAGES.inc()
                if len(buf) >= self._batch_size or (buf and (time.perf_counter() - last) >= self._max_wait):
                    KAFKA_CONSUMER_BATCH_SIZE.observe(len(buf))
                    process_fn(buf)
                    KAFKA_CONSUMER_BATCHES.inc()
                    buf = []
                    last = time.perf_counter()
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            if buf:
                KAFKA_CONSUMER_BATCH_SIZE.observe(len(buf))
                process_fn(buf)
                KAFKA_CONSUMER_BATCHES.inc()
            try:
                self._consumer.close()
            except Exception:
                pass
