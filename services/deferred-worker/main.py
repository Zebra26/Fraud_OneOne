from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, List

import httpx

from backend.streaming.kafka_consumer_batch import KafkaBatchConsumer
from backend.streaming.kafka_producer import TransactionProducer
from backend.security.jwt_utils import generate_jwt
from backend.security.hmac_utils import sign_request_v2
from backend.database.redis_client import RedisCache
from backend.database.mongo import MongoDBClient


def _get_jwt() -> str:
    tok = os.getenv("JWT_TOKEN")
    if tok:
        return tok
    return generate_jwt({"sub": "deferred-worker", "roles": ["service"]}, expires_in=3600)


def _parse_redis_nodes(nodes_csv: str) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for part in nodes_csv.split(","):
        host, _, port = part.partition(":")
        if host:
            nodes.append({"host": host.strip(), "port": int(port or 7000)})
    return nodes


async def _process_batch(batch: List[Dict[str, Any]], predictions_topic: str, inference_url: str, producer: TransactionProducer) -> None:
    token = _get_jwt()
    timeout = httpx.Timeout(10.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Optional offline writes
        offline = os.getenv("OFFLINE_WRITE", "true").strip().lower() in {"1", "true", "yes"}
        redis_cache: RedisCache | None = None
        mongo_client: MongoDBClient | None = None
        if offline:
            try:
                rc_nodes = _parse_redis_nodes(os.getenv("REDIS_CLUSTER_NODES", "redis-cluster:7000,redis-cluster:7001,redis-cluster:7002"))
                redis_cache = RedisCache(
                    startup_nodes=rc_nodes,
                    password=os.getenv("REDIS_PASSWORD") or None,
                    use_ssl=os.getenv("REDIS_USE_SSL", "false").strip().lower() in {"1", "true", "yes"},
                    ca_certs=os.getenv("TLS_CA_PATH"),
                )
            except Exception:
                redis_cache = None
            try:
                mongo_client = MongoDBClient(
                    os.getenv("MONGO_URI", "mongodb://mongo:27017/fraud_db"),
                    os.getenv("MONGO_DB", "fraud_db"),
                    tls_enabled=os.getenv("MONGO_SSL", "false").strip().lower() in {"1", "true", "yes"},
                    tls_verify=True,
                    tls_ca_path=os.getenv("MONGO_TLS_CA_PATH"),
                )
            except Exception:
                mongo_client = None
        for msg in batch:
            try:
                payload = {
                    "transaction_id": msg.get("transaction_id"),
                    "account_id": msg.get("account_id"),
                    "features": msg.get("features"),
                    "graph": msg.get("graph"),
                    "sequence": msg.get("sequence"),
                    "channel": msg.get("channel"),
                }
                body = json.dumps(payload).encode("utf-8")
                ts = str(int(time.time()))
                sig = sign_request_v2("POST", "/infer", ts, body)
                headers = {"Authorization": f"Bearer {token}", "X-Request-Timestamp": ts, "X-Request-Signature": sig}
                resp = await client.post(f"{inference_url}/infer", content=body, headers=headers)
                if resp.status_code == 200:
                    result = resp.json()
                    producer.send_transaction(predictions_topic, {
                        "transaction_id": payload["transaction_id"],
                        "account_id": payload.get("account_id"),
                        "result": result,
                        "channel": payload.get("channel"),
                        "deferred": True,
                    })
                    # Offline writes (best-effort): Redis cache + Mongo explanation
                    if offline and redis_cache is not None:
                        try:
                            await redis_cache.set_feature_vector(
                                f"features:{payload['transaction_id']}", json.dumps(payload), ttl_seconds=3600
                            )
                            await redis_cache.set_risk_score(payload["transaction_id"], float(result.get("score", 0.0)), ttl_seconds=600)
                        except Exception:
                            pass
                    if offline and mongo_client is not None:
                        try:
                            from datetime import datetime as _dt

                            doc = {
                                "transaction_id": payload["transaction_id"],
                                "decision": result.get("decision"),
                                "fraud_probability": float(result.get("score", 0.0)),
                                "components": result.get("breakdown", {}),
                                "threshold": float(result.get("threshold", 0.7)),
                                "channel": payload.get("channel"),
                                "recorded_at": _dt.utcnow(),
                                "deferred": True,
                            }
                            await mongo_client.insert_explanation(doc)
                        except Exception:
                            pass
            except Exception:
                # best-effort; skip on failure
                pass


def main() -> None:
    brokers = os.getenv("KAFKA_BROKERS", "kafka:9092").split(",")
    deferred_topic = os.getenv("DEFERRED_TOPIC", "predictions.deferred")
    predictions_topic = os.getenv("KAFKA_TOPIC_PREDICTIONS", "predictions")
    inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")

    consumer = KafkaBatchConsumer([deferred_topic], brokers, group_id="deferred-worker", batch_size=200, max_wait_ms=250)
    producer = TransactionProducer(brokers)

    def _proc(batch: List[Dict[str, Any]]):
        import asyncio

        asyncio.run(_process_batch(batch, predictions_topic, inference_url, producer))

    consumer.run(_proc)


if __name__ == "__main__":
    main()
