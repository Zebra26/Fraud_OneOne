import json
import logging
import os
import time
from datetime import datetime

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections import deque
from fastapi import APIRouter, Depends, HTTPException, Request
from prometheus_client import Counter, Histogram, Gauge

from ..config import get_settings
from ..dependencies import (
    get_model_service,
    get_mongo_client,
    get_redis_cache,
    get_explanations_batch_writer,
    get_redis_batch_writer,
)
from ...security.crypto_utils import encrypt_text, pii_token
from ..schemas import PredictionExplanation, PredictionResponse, RiskBreakdownSchema, TransactionIn
from ...security.hmac_utils import sign_request_v2

router = APIRouter(prefix="/predictions", tags=["predictions"])

logger = logging.getLogger(__name__)

REDIS_OPS = Counter("redis_ops_total", "Redis operations", ["operation"])
MONGO_WRITE_LATENCY = Histogram(
    "mongo_write_ms",
    "MongoDB write latency (ms)",
    buckets=[1, 5, 10, 20, 50, 100, 200],
)
MODEL_DECISIONS = Counter(
    "model_decisions_total",
    "Model decisions emitted",
    ["decision", "channel", "model_version"],
)
DEGRADED_DECISIONS = Counter("degraded_predictions_total", "Predictions served in degraded mode")
DEFERRED_PREDICTIONS = Counter("deferred_predictions_total", "Predictions deferred to Kafka")


async def _call_inference(payload: dict, inference_url: str, request: Request) -> dict:
    # Circuit breaker (simple rolling window)
    if CIRCUIT_BREAKER.is_open():
        raise HTTPException(status_code=503, detail="Inference upstream unavailable (circuit open)")

    # Forward JWT and HMAC for S2S auth
    body_bytes = json.dumps(payload).encode("utf-8")
    ts = str(int(time.time()))
    signature = sign_request_v2("POST", "/infer", ts, body_bytes)
    headers = {
        "X-Request-Timestamp": ts,
        "X-Request-Signature": signature,
    }
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    corr = getattr(request.state, "correlation_id", None)
    if corr:
        headers["X-Correlation-ID"] = corr
    client = getattr(request.app.state, "http_client", None)
    owns_client = False
    if client is None:
        # Fallback if startup client not available
        client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0))
        owns_client = True
    try:
        attempts = int(os.getenv("RETRY_ATTEMPTS", "2"))
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, attempts)),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=2.0),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                response = await client.post(
                    f"{inference_url}/infer", content=body_bytes, headers=headers
                )
    except httpx.HTTPError as exc:  # pragma: no cover - network
        CIRCUIT_BREAKER.record_failure()
        raise HTTPException(status_code=502, detail=f"Inference service unreachable: {exc}")
    finally:
        if owns_client:
            await client.aclose()
    if response.status_code != 200:
        CIRCUIT_BREAKER.record_failure()
        raise HTTPException(status_code=response.status_code, detail=response.text)
    CIRCUIT_BREAKER.record_success()
    return response.json()


@router.post("/score", response_model=PredictionResponse)
async def score_transaction(
    payload: TransactionIn,
    request: Request,
    settings=Depends(get_settings),
    mongo=Depends(get_mongo_client),
    writer=Depends(get_explanations_batch_writer),
    redis_cache=Depends(get_redis_cache),
    redis_writer=Depends(get_redis_batch_writer),
):
    perf_mode = os.getenv("PERF_MODE", "false").strip().lower() in {"1", "true", "yes"}
    use_local = os.getenv("USE_LOCAL_INFERENCE", "false").strip().lower() in {"1", "true", "yes"}

    inference_payload = {
        "transaction_id": payload.transaction_id,
        "account_id": payload.account_id,
        "features": payload.features.model_dump(),
        "graph": payload.graph_context.dict() if payload.graph_context else None,
        "sequence": payload.sequence,
        "channel": payload.channel,
    }

    if use_local:
        service = get_model_service()
        prob = float(service.predict_proba(inference_payload["features"]))
        decision = "FRAUD" if prob >= service.thresholds.get("block", 0.85) else "NORMAL"
        result = {
            "score": prob,
            "breakdown": {"tabular": prob, "graph": 0.0, "autoencoder": 0.0, "lstm": 0.0},
            "decision": decision,
            "threshold": service.thresholds.get("block", 0.85),
        }
    else:
        inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")
        # Call inference with circuit breaker and timeout; fall back on failure
        try:
            result = await _call_inference(inference_payload, inference_url, request)
        except HTTPException:
            # Graceful degradation: compute a light heuristic and enqueue for offline processing
            result = _light_score(payload)
            DEGRADED_DECISIONS.inc()
            if not perf_mode:
                try:
                    from ..dependencies import get_kafka_producer

                    producer = get_kafka_producer()
                    deferred_topic = os.getenv("DEFERRED_TOPIC", os.getenv("KAFKA_TOPIC_PREDICTIONS", "predictions") + ".deferred")
                    producer.send_transaction(
                        deferred_topic,
                        {
                            "type": "deferred_prediction",
                            "transaction_id": payload.transaction_id,
                            "account_id": payload.account_id,
                            "features": payload.features.model_dump(),
                            "graph": inference_payload["graph"],
                            "sequence": payload.sequence,
                            "channel": payload.channel,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )
                    DEFERRED_PREDICTIONS.inc()
                except Exception:
                    pass

    fraud_probability = float(result.get("score", 0.0))
    decision = result.get("decision", "NORMAL")
    threshold = float(result.get("threshold", 0.7))
    breakdown = result.get("breakdown", {})

    try:
        metadata = get_model_service().model_metadata()
    except Exception:
        metadata = {}

    explanation = PredictionExplanation(
        base_value=0.0,
        shap_values=[],
        model_version=metadata.get("model_version", "unknown"),
    )

    risk_breakdown = RiskBreakdownSchema(
        supervised_probability=breakdown.get("tabular", fraud_probability),
        anomaly_risk=breakdown.get("autoencoder", 0.0),
        deep_reconstruction_risk=breakdown.get("lstm", 0.0),
        aggregated_risk=fraud_probability,
    )

    response = PredictionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=fraud_probability,
        decision=decision,
        risk_breakdown=risk_breakdown,
        explanation=explanation,
    )

    MODEL_DECISIONS.labels(
        decision=decision,
        channel=payload.channel or "unknown",
        model_version=metadata.get("model_version", "unknown"),
    ).inc()

    feature_key = f"features:{payload.transaction_id}"
    feature_val = json.dumps(
        {
            "features": payload.features.model_dump(),
            "graph": inference_payload["graph"],
            "sequence": payload.sequence,
        }
    )
    if not perf_mode:
        try:
            # enqueue in Redis pipeline
            await redis_writer.enqueue_set(feature_key, feature_val, ttl_seconds=3600)
            await redis_writer.enqueue_set(f"risk:{payload.transaction_id}", fraud_probability, ttl_seconds=600)
            REDIS_OPS.labels(operation="enqueue_feature_vector").inc()
            REDIS_OPS.labels(operation="enqueue_risk_score").inc()
        except Exception:
            # fallback to direct writes
            await redis_cache.set_feature_vector(feature_key, feature_val)
            await redis_cache.set_risk_score(payload.transaction_id, fraud_probability)
            REDIS_OPS.labels(operation="fallback_set").inc()

    # Micro-batch enqueue (non-blocking if configured)
    is_fraud = decision == "FRAUD"
    # Tokenize PII for indexing
    customer_token = pii_token(payload.account_id) if payload.account_id else None
    device_token = pii_token(payload.device_id) if payload.device_id else None
    # Encrypt PII for storage
    account_enc = encrypt_text(payload.account_id) if payload.account_id else None
    device_enc = encrypt_text(payload.device_id) if payload.device_id else None
    geo_enc = encrypt_text(json.dumps(payload.geolocation)) if payload.geolocation else None

    doc = {
        "transaction_id": payload.transaction_id,
        "decision": decision,
        "fraud_probability": fraud_probability,
        "components": breakdown,
        "threshold": threshold,
        "channel": payload.channel,
        "recorded_at": datetime.utcnow(),
        "is_fraud": is_fraud,
        "customer_token": customer_token,
        "device_token": device_token,
        "account_enc": account_enc,
        "device_enc": device_enc,
        "geolocation_enc": geo_enc,
    }
    if not perf_mode:
        try:
            await writer.enqueue(doc)
        except Exception:
            # fallback to direct write on failure
            await mongo.insert_explanation(doc)

    logger.info(
        "Prediction computed",
        extra={
            "transaction_id": payload.transaction_id,
            "probability": fraud_probability,
            "decision": decision,
        },
    )

    return response


def _light_score(payload: TransactionIn) -> dict:
    """Heuristic-based degraded scoring.

    Combines a few high-signal features for a quick approximation.
    Tunable via env: LIGHT_THRESHOLD, LIGHT_RATIO_STRONG/MEDIUM/LIGHT,
    LIGHT_WEIGHT_IP/LOC/VPN/IP_CHANGE/VELOCITY
    """
    import os

    f = payload.features
    score = 0.0
    # amount anomaly vs avg
    r_strong = float(os.getenv("LIGHT_RATIO_STRONG", "5"))
    r_med = float(os.getenv("LIGHT_RATIO_MEDIUM", "3"))
    r_light = float(os.getenv("LIGHT_RATIO_LIGHT", "2"))
    if f.avg_transaction_amount_24h > 0:
        ratio = f.transaction_amount / max(1.0, f.avg_transaction_amount_24h)
        if ratio > r_strong:
            score += 0.5
        elif ratio > r_med:
            score += 0.3
        elif ratio > r_light:
            score += 0.15
    # IP and location risks
    w_ip = float(os.getenv("LIGHT_WEIGHT_IP", "0.2"))
    w_loc = float(os.getenv("LIGHT_WEIGHT_LOC", "0.2"))
    score += w_ip * float(f.ip_risk_score)
    score += w_loc * float(f.location_risk_score)
    # VPN and login IP change signals
    w_vpn = float(os.getenv("LIGHT_WEIGHT_VPN", "0.1"))
    w_ipchg = float(os.getenv("LIGHT_WEIGHT_IP_CHANGE", "0.1"))
    if f.vpn_detected:
        score += w_vpn
    if f.login_ip_changed_last_hour:
        score += w_ipchg
    # High recent velocity
    w_vel = float(os.getenv("LIGHT_WEIGHT_VELOCITY", "0.1"))
    if f.transaction_frequency_30min > 20 or f.transactions_last_24h > 200:
        score += w_vel
    score = max(0.0, min(1.0, score))
    threshold = float(os.getenv("LIGHT_THRESHOLD", "0.7"))
    decision = "FRAUD" if score >= threshold else "NORMAL"
    return {
        "score": score,
        "breakdown": {
            "tabular": score,
            "graph": 0.0,
            "autoencoder": 0.0,
            "lstm": 0.0,
        },
        "decision": decision,
        "threshold": threshold,
    }


# --- Circuit breaker (lightweight) ---

CIRCUIT_OPEN = Gauge("inference_circuit_open", "Circuit breaker state for inference upstream (1=open,0=closed)")
CIRCUIT_TRIPS = Counter("inference_circuit_trips_total", "Total circuit breaker open events")


class _CircuitBreaker:
    def __init__(self, max_failures: int = 5, window_sec: int = 60, open_sec: int = 30):
        self.max_failures = max_failures
        self.window_sec = window_sec
        self.open_sec = open_sec
        self.failures: deque[float] = deque()
        self.open_until: float = 0.0

    def is_open(self) -> bool:
        now = time.time()
        if self.open_until > now:
            CIRCUIT_OPEN.set(1)
            return True
        CIRCUIT_OPEN.set(0)
        return False

    def _prune(self) -> None:
        now = time.time()
        cutoff = now - self.window_sec
        while self.failures and self.failures[0] < cutoff:
            self.failures.popleft()

    def record_failure(self) -> None:
        now = time.time()
        self.failures.append(now)
        self._prune()
        if len(self.failures) >= self.max_failures:
            self.open_until = now + self.open_sec
            CIRCUIT_TRIPS.inc()
            CIRCUIT_OPEN.set(1)

    def record_success(self) -> None:
        self.failures.clear()
        self.open_until = 0.0
        CIRCUIT_OPEN.set(0)


import os as _os

CIRCUIT_BREAKER = _CircuitBreaker(
    max_failures=int(_os.getenv("CB_MAX_FAILURES", "5")),
    window_sec=int(_os.getenv("CB_WINDOW_SEC", "60")),
    open_sec=int(_os.getenv("CB_OPEN_SEC", "30")),
)


@router.get("/models/info")
async def models_info(settings=Depends(get_settings)):
    inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{inference_url}/models/info")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
