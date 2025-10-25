from __future__ import annotations

import atexit
from functools import lru_cache
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
except Exception:
    torch = None  # type: ignore
    class _NN:  # minimal stubs to allow module import
        class Module:  # type: ignore
            pass
        class LSTM:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass
        class Linear:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass
        class Sigmoid:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass
        class Sequential:  # type: ignore
            def __init__(self, *layers, **kwargs):
                pass
    nn = _NN()  # type: ignore
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, generate_latest
import asyncio
from pydantic import BaseModel, Field, ConfigDict

from backend.security.crypto_utils import compute_file_hash, decrypt_file_aes256, secure_delete
try:
    from backend.security.gpg_utils import gpg_verify_file
except Exception:
    def gpg_verify_file(*args, **kwargs):  # type: ignore
        return False
from backend.security.hmac_utils import verify_request_v2 as verify_hmac_signature
from backend.security.jwt_utils import verify_jwt
from backend.security.model_integrity import load_hash_manifest
from safe_mode import SafeModeState
from metrics import PrometheusMiddleware
from logging_config import configure_json_logging, set_log_context
from backend.monitoring.drift import DriftMonitor, DRIFT_ALERTS

configure_json_logging(service="ml-inference")
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_dict(model_obj):
    """Serialize Pydantic model compatibly across v1/v2."""
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    if hasattr(model_obj, "dict"):
        return model_obj.dict()
    return model_obj


if 'JWT_VALIDATIONS' not in globals():
    JWT_VALIDATIONS = Counter("infer_jwt_validations_total", "Total JWT validations")
if 'JWT_FAILURES' not in globals():
    JWT_FAILURES = Counter("infer_jwt_failures_total", "Total JWT verification failures")
if 'HMAC_VALIDATIONS' not in globals():
    HMAC_VALIDATIONS = Counter("infer_hmac_verifications_total", "Total HMAC verifications")
if 'AUTH_FAILURES' not in globals():
    AUTH_FAILURES = Counter("infer_auth_failures_total", "Authentication failures")
if 'REQUEST_LATENCY' not in globals():
    REQUEST_LATENCY = Histogram(
        "infer_request_latency_ms",
        "Inference request latency (ms)",
        buckets=[1, 5, 10, 20, 50, 100, 200, 400, 1000],
    )
if 'MODEL_DECISIONS' not in globals():
    MODEL_DECISIONS = Counter(
        "infer_model_decisions_total",
        "Model decisions emitted",
        ["decision", "channel", "model_version"],
    )
if 'INFER_BATCH_SIZE' not in globals():
    INFER_BATCH_SIZE = Histogram("inference_batch_size", "Batch size used per inference call", buckets=[1,2,4,8,16,32,64])


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


class _LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class HybridInferenceService:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.decrypted_dir = Path(os.getenv("MODEL_DECRYPT_DIR", "/tmp/model_decrypted"))
        self.decrypted_dir.mkdir(parents=True, exist_ok=True)
        self._temp_paths: List[Path] = []
        atexit.register(self._cleanup)

        self.verify_hash = _env_flag("VERIFY_HASH_ON_LOAD", True)
        self.models_encrypted = _env_flag("MODELS_ENCRYPTED", True)
        self.verify_signature = _env_flag("VERIFY_SIGNATURE_ON_LOAD", True)

        self.public_key_path = os.getenv("GPG_PUBLIC_KEY_PATH")
        self.gpg_key_id = os.getenv("GPG_KEY_ID")
        if self.verify_signature and not (self.public_key_path and Path(self.public_key_path).exists()):
            logger.warning("Signature verification disabled: missing public key")
            self.verify_signature = False

        self.performance: Dict[str, Any] = {}
        self.hash_manifest: Dict[str, str] = load_hash_manifest(self.model_dir / "model_hashes.json")

        metadata_path = self.model_dir / "model_performance.json"
        if metadata_path.exists():
            self.performance = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not self.hash_manifest:
                self.hash_manifest = self.performance.get("model_hashes", {})
        else:
            logger.warning("model_performance.json not found in %s", self.model_dir)

        self.threshold: float = self.performance.get("optimal_threshold", 0.7)
        self.weights: Dict[str, float] = {
            "tabular": 0.7,
            "graph": 0.1,
            "autoencoder": 0.15,
            "lstm": 0.05,
        }
        self.weights.update(self.performance.get("weights", {}))
        self.model_version: str = self.performance.get("model_version", "unknown")
        self.feature_names: List[str] = self.performance.get("feature_names", [])
        self.artifacts_meta: Dict[str, Dict[str, str]] = {
            item["name"]: item for item in self.performance.get("artifacts", []) if isinstance(item, dict)
        }

        self.enable_jwt = _env_flag("ENABLE_JWT_AUTH", True)
        self.enable_hmac = _env_flag("ENABLE_HMAC_SIGNING", True)
        self.allowed_skew = int(os.getenv("ALLOWED_SKEW_SECONDS", "120"))
        self.safe_mode_env = _env_flag("SAFE_MODE", False)

        self.tabular_model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.graph_classifier = None
        self.autoencoder: Optional[_Autoencoder] = None
        self.autoencoder_scaler = None
        self.lstm: Optional[_LSTM] = None
        self.sequential_scaler = None
        try:
            self._load_models()
        except Exception as exc:  # pragma: no cover - fatal path
            logger.exception("Failed to load models: %s", exc)
            SafeModeState.enable(str(exc))

        if self.safe_mode_env:
            logger.warning("SAFE_MODE enforced by environment")
            SafeModeState.enable("SAFE_MODE_ENV")

    def _cleanup(self) -> None:
        for path in self._temp_paths:
            try:
                secure_delete(path)
            except Exception:  # pragma: no cover
                pass

    def _resolve_artifact(self, filename: str, *, critical: bool = False) -> Tuple[Optional[Path], bool]:
        expected_hash = self.hash_manifest.get(filename)
        record = self.artifacts_meta.get(filename, {})
        enc_name = record.get("enc", f"{filename}.enc")
        sig_name = record.get("sig", f"{filename}.asc")
        enc_path = self.model_dir / enc_name
        sig_path = self.model_dir / sig_name
        decrypted_path = self.decrypted_dir / filename

        if self.models_encrypted and enc_path.exists():
            try:
                decrypt_file_aes256(enc_path, decrypted_path)
                self._temp_paths.append(decrypted_path)
                if self.verify_signature and sig_path.exists() and self.public_key_path:
                    if not gpg_verify_file(
                        decrypted_path,
                        sig_path,
                        public_key_path=self.public_key_path,
                        key_id=self.gpg_key_id,
                    ):
                        raise RuntimeError("Signature verification failed")
                elif self.verify_signature:
                    logger.warning("Missing signature for %s", filename)

                if self.verify_hash and expected_hash:
                    actual_hash = compute_file_hash(decrypted_path)
                    if actual_hash != expected_hash:
                        raise RuntimeError("Hash mismatch after decryption")

                logger.info("Decrypted and verified %s", filename)
                return decrypted_path, True
            except Exception as exc:
                logger.error("Failed to use encrypted artifact %s: %s", filename, exc)
                if decrypted_path.exists():
                    secure_delete(decrypted_path)

        plain_path = self.model_dir / filename
        if plain_path.exists():
            if self.verify_hash and expected_hash:
                actual_hash = compute_file_hash(plain_path)
                if actual_hash != expected_hash:
                    message = f"Hash mismatch for plaintext artifact {filename}"
                    if critical:
                        raise RuntimeError(message)
                    logger.warning(message)
            logger.warning("Using plaintext artifact for %s", filename)
            return plain_path, False

        if critical:
            raise RuntimeError(f"Required artifact missing: {filename}")
        logger.warning("Optional artifact %s unavailable", filename)
        return None, False

    def _load_models(self) -> None:
        if torch is None:
            # Torch not available; skip loading torch-based models
            return
        path, temp = self._resolve_artifact("fraud_detection_advanced_model.pkl", critical=True)
        with path.open("rb") as handle:
            self.tabular_model = pickle.load(handle)
        if temp and path:
            secure_delete(path)

        path, temp = self._resolve_artifact("node2vec_model.pkl")
        if path:
            with path.open("rb") as handle:
                self.embeddings = pickle.load(handle)
            if temp and path:
                secure_delete(path)

        path, temp = self._resolve_artifact("optimized_fraud_detection_system.pkl")
        if path:
            with path.open("rb") as handle:
                self.graph_classifier = pickle.load(handle)
            if temp and path:
                secure_delete(path)

        scaler_path, scaler_temp = self._resolve_artifact("autoencoder_scaler.pkl")
        if scaler_path:
            with scaler_path.open("rb") as handle:
                self.autoencoder_scaler = pickle.load(handle)
            if scaler_temp and scaler_path:
                secure_delete(scaler_path)

        weights_path, weights_temp = self._resolve_artifact("fraud_autoencoder.pth")
        if weights_path and self.autoencoder_scaler is not None:
            input_dim = len(self.autoencoder_scaler.mean_)
            self.autoencoder = _Autoencoder(input_dim)
            state = torch.load(weights_path, map_location="cpu")
            self.autoencoder.load_state_dict(state)
            self.autoencoder.eval()
        if weights_temp and weights_path:
            secure_delete(weights_path)

        scaler_path, scaler_temp = self._resolve_artifact("sequential_scaler.pkl")
        if scaler_path:
            with scaler_path.open("rb") as handle:
                self.sequential_scaler = pickle.load(handle)
            if scaler_temp and scaler_path:
                secure_delete(scaler_path)

        lstm_path, lstm_temp = self._resolve_artifact("lstm_fraud_detector.pth")
        if lstm_path and self.sequential_scaler is not None:
            input_dim = len(self.sequential_scaler.mean_)
            self.lstm = _LSTM(input_dim)
            state = torch.load(lstm_path, map_location="cpu")
            self.lstm.load_state_dict(state)
            self.lstm.eval()
        if lstm_temp and lstm_path:
            secure_delete(lstm_path)
        SafeModeState.disable()

    def _prepare_tabular(self, features: Mapping[str, Any]) -> pd.DataFrame:
        if self.feature_names:
            ordered = {name: features.get(name) for name in self.feature_names}
        else:
            ordered = dict(features)
        return pd.DataFrame([ordered])

    def _score_tabular(self, features: Mapping[str, Any]) -> float:
        if self.tabular_model is None:
            return 0.0
        frame = self._prepare_tabular(features)
        prob = self.tabular_model.predict_proba(frame)[0][1]
        return float(prob)

    def _score_graph(self, graph: Optional[Mapping[str, str]]) -> float:
        if not graph or not self.embeddings:
            return 0.0
        source = graph.get("source_id")
        target = graph.get("target_id")
        if source is None or target is None:
            return 0.0
        emb_src = self.embeddings.get(source)
        emb_dst = self.embeddings.get(target)
        if emb_src is None or emb_dst is None:
            return 0.0
        if self.graph_classifier is not None:
            vec = np.concatenate([emb_src, emb_dst, np.abs(emb_src - emb_dst)])
            return float(self.graph_classifier.predict_proba([vec])[0][1])
        denom = np.linalg.norm(emb_src) * np.linalg.norm(emb_dst)
        if denom == 0:
            return 0.0
        similarity = float(np.dot(emb_src, emb_dst) / denom)
        return (1 - similarity) * 0.5 + 0.5

    def _score_autoencoder(self, features: Iterable[float]) -> float:
        if self.autoencoder is None or self.autoencoder_scaler is None:
            return 0.0
        array = np.asarray(list(features), dtype=np.float32)
        scaled = self.autoencoder_scaler.transform([array])[0]
        tensor = torch.from_numpy(scaled)
        with torch.no_grad():
            recon = self.autoencoder(tensor)
            loss = torch.mean((tensor - recon) ** 2).item()
        return float(min(1.0, loss * 10))

    def _score_lstm(self, sequence: Optional[Sequence[Iterable[float]]]) -> float:
        if self.lstm is None or self.sequential_scaler is None or not sequence:
            return 0.0
        seq_np = np.asarray([list(vec) for vec in sequence], dtype=np.float32)
        scaled = self.sequential_scaler.transform(seq_np)
        tensor = torch.from_numpy(scaled).unsqueeze(0)
        with torch.no_grad():
            prob = self.lstm(tensor).item()
        return float(prob)

    def infer(
        self,
        features: Mapping[str, Any],
        graph: Optional[Mapping[str, str]] = None,
        sequence: Optional[Sequence[Iterable[float]]] = None,
    ) -> Dict[str, Any]:
        tabular_score = self._score_tabular(features)
        graph_score = self._score_graph(graph)
        autoencoder_features = (
            [features.get(name, 0.0) for name in self.feature_names]
            if self.feature_names
            else list(features.values())
        )
        autoencoder_score = self._score_autoencoder(autoencoder_features)
        lstm_score = self._score_lstm(sequence)

        components = {
            "tabular": tabular_score,
            "graph": graph_score,
            "autoencoder": autoencoder_score,
            "lstm": lstm_score,
        }

        total_weight = sum(self.weights.values()) or 1.0
        combined = sum(self.weights[key] * components[key] for key in components) / total_weight

        decision = "FRAUD" if combined >= self.threshold else "NORMAL"

        return {
            "score": combined,
            "breakdown": components,
            "decision": decision,
            "threshold": self.threshold,
        }


@lru_cache(maxsize=1)
def get_service(model_dir: str = "/app/models") -> HybridInferenceService:
    return HybridInferenceService(model_dir=model_dir)


app = FastAPI(title="Hybrid Fraud Inference", version="0.5.0")
app.add_middleware(PrometheusMiddleware)

SERVICE = get_service()

# Drift monitor (optional, requires baseline in model_performance.json)
DRIFT_MONITOR: DriftMonitor | None = None
try:
    from backend.monitoring.drift import load_baseline_from_model_meta

    baseline = load_baseline_from_model_meta(Path(SERVICE.model_dir))
    if baseline:
        import os

        warn = float(os.getenv("DRIFT_PSI_WARN", "0.2"))
        crit = float(os.getenv("DRIFT_PSI_CRIT", "0.3"))
        window = int(os.getenv("DRIFT_WINDOW", "5000"))
        DRIFT_MONITOR = DriftMonitor(baseline, window=window, warn=warn, crit=crit)
except Exception as _exc:  # pragma: no cover
    DRIFT_MONITOR = None


@app.get("/health")
def health():
    info = SERVICE.performance.copy() if SERVICE.performance else {}
    info.update(
        {
            "status": "ok" if not SERVICE.safe_mode else "safe_mode",
            "model_version": SERVICE.model_version,
            "threshold": SERVICE.threshold,
            "weights": SERVICE.weights,
        }
    )
    return info


@app.get("/models/info")
def models_info():
    info = SERVICE.performance.copy() if SERVICE.performance else {}
    info.setdefault("weights", SERVICE.weights)
    info.setdefault("optimal_threshold", SERVICE.threshold)
    info.setdefault("feature_names", SERVICE.feature_names)
    info.setdefault("model_version", SERVICE.model_version)
    return info


async def _authorize(request: Request) -> None:
    if SafeModeState.is_enabled():
        status = SafeModeState.status()
        raise HTTPException(
            status_code=503,
            detail={"status": "safe_mode", "reason": status["reason"], "since": status["since"]},
        )

    claims = {}
    # In mTLS-only mode, skip JWT/HMAC checks (assumed network-level auth)
    if os.getenv("ENABLE_MTLS", "false").strip().lower() in {"1", "true", "yes"}:
        request.state.jwt_claims = {}
        return

    if SERVICE.enable_jwt:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            JWT_FAILURES.inc()
            AUTH_FAILURES.inc()
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = auth_header.split(" ", 1)[1]
        try:
            claims = verify_jwt(token)
            JWT_VALIDATIONS.inc()
        except Exception as exc:
            JWT_FAILURES.inc()
            AUTH_FAILURES.inc()
            logger.warning("JWT verification failed: %s", exc)
            raise HTTPException(status_code=401, detail="Invalid token") from exc

    if SERVICE.enable_hmac:
        signature = request.headers.get("X-Request-Signature")
        timestamp = request.headers.get("X-Request-Timestamp")
        if not signature or not timestamp:
            AUTH_FAILURES.inc()
            raise HTTPException(status_code=401, detail="Missing signature headers")
        try:
            ts = int(timestamp)
        except ValueError as exc:
            AUTH_FAILURES.inc()
            raise HTTPException(status_code=401, detail="Invalid timestamp") from exc
        if abs(time.time() - ts) > SERVICE.allowed_skew:
            AUTH_FAILURES.inc()
            raise HTTPException(status_code=401, detail="Signature expired")
        body = await request.body()
        if not verify_hmac_signature(request.method.upper(), request.url.path, timestamp, body, signature):
            AUTH_FAILURES.inc()
            raise HTTPException(status_code=401, detail="Invalid signature")
        HMAC_VALIDATIONS.inc()
    request.state.jwt_claims = claims
    # Bind logging context
    set_log_context(
        correlation_id=request.headers.get("X-Correlation-ID"),
        route=request.url.path,
        user_agent=request.headers.get("user-agent", "")[:120],
    )


class FraudFeatures(BaseModel):
    transaction_amount: float
    transaction_time_seconds: int
    is_weekend: int
    hour_of_day: int
    location_risk_score: float
    transaction_frequency_30min: int
    login_ip_changed_last_hour: int
    avg_transaction_amount_24h: float
    time_since_last_tx: int
    ip_risk_score: float
    vpn_detected: int
    transactions_last_24h: int
    is_round_amount: int
    unique_receivers_24h: int

    model_config = ConfigDict(extra="forbid")


class GraphContext(BaseModel):
    source_id: str
    target_id: str

    model_config = ConfigDict(extra="forbid")


class InferRequest(BaseModel):
    transaction_id: str
    account_id: str
    features: FraudFeatures
    graph: Optional[GraphContext] = None
    sequence: Optional[List[List[float]]] = Field(default=None, description="Sequence for LSTM scoring")
    channel: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class InferResponse(BaseModel):
    score: float
    breakdown: Dict[str, float]
    decision: str
    threshold: float


@app.post("/infer", response_model=InferResponse)
async def infer(request: Request, payload: InferRequest):
    await _authorize(request)
    start = time.perf_counter()

    # ONNX fast-path (supervised component only) if enabled
    result: Dict[str, Any]
    if 'ONNX_BATCHER' in globals() and ONNX_BATCHER is not None and SERVICE.feature_names:
        try:
            fv = payload.features.model_dump()
            vec = [fv.get(name, 0.0) for name in SERVICE.feature_names]
            prob = float(await ONNX_BATCHER.run(vec))
            decision = "FRAUD" if prob >= SERVICE.threshold else "NORMAL"
            result = {
                "score": prob,
                "breakdown": {"tabular": prob, "graph": 0.0, "autoencoder": 0.0, "lstm": 0.0},
                "decision": decision,
                "threshold": SERVICE.threshold,
            }
        except Exception as _exc:  # pragma: no cover
            result = SERVICE.infer(
                features=payload.features.model_dump(),
                graph=payload.graph.model_dump() if payload.graph else None,
                sequence=payload.sequence,
            )
    elif ONNX_SUPERVISED and ONNX_INPUT_NAME and SERVICE.feature_names:
        try:
            import numpy as _np

            fv = payload.features.model_dump()
            vec = _np.array([[fv.get(name, 0.0) for name in SERVICE.feature_names]], dtype=_np.float32)
            out = ONNX_SUPERVISED.run(None, {ONNX_INPUT_NAME: vec})  # type: ignore
            prob = float(out[0].ravel()[0]) if out and len(out) > 0 else 0.0
            # Combine with weights (only supervised available)
            decision = "FRAUD" if prob >= SERVICE.threshold else "NORMAL"
            result = {
                "score": prob,
                "breakdown": {"tabular": prob, "graph": 0.0, "autoencoder": 0.0, "lstm": 0.0},
                "decision": decision,
                "threshold": SERVICE.threshold,
            }
        except Exception as _exc:  # pragma: no cover
            result = SERVICE.infer(
                features=payload.features.model_dump(),
                graph=payload.graph.model_dump() if payload.graph else None,
                sequence=payload.sequence,
            )
    else:
        result = SERVICE.infer(
            features=payload.features.model_dump(),
            graph=payload.graph.model_dump() if payload.graph else None,
            sequence=payload.sequence,
        )

    # Drift update (best effort)
    try:
        if DRIFT_MONITOR is not None:
            DRIFT_MONITOR.observe(payload.features.model_dump())
    except Exception:  # pragma: no cover
        pass

    REQUEST_LATENCY.observe((time.perf_counter() - start) * 1000)
    # Decision metric
    MODEL_DECISIONS.labels(
        decision=result.get("decision", "UNKNOWN"),
        channel=payload.channel or "unknown",
        model_version=SERVICE.model_version,
    ).inc()
    return InferResponse(**result)


def _require_admin(request: Request) -> None:
    claims = getattr(request.state, "jwt_claims", {}) or {}
    roles = set(claims.get("roles", []))
    if not roles and claims.get("role"):
        roles = {claims.get("role")}
    if "admin" not in roles:
        raise HTTPException(status_code=403, detail="Admin role required")


@app.post("/admin/safe-mode/enable")
async def enable_safe_mode(request: Request, reason: Dict[str, str]):
    await _authorize(request)
    _require_admin(request)
    r = reason.get("reason", "manual")
    SafeModeState.enable(r)
    logger.warning("Safe mode enabled", extra={"reason": r})
    return {"status": "safe_mode_enabled", "state": SafeModeState.status()}


@app.post("/admin/safe-mode/disable")
async def disable_safe_mode(request: Request):
    await _authorize(request)
    _require_admin(request)
    SafeModeState.disable()
    logger.warning("Safe mode disabled")
    return {"status": "safe_mode_disabled", "state": SafeModeState.status()}


@app.get("/admin/safe-mode/status")
async def safe_mode_status(request: Request):
    await _authorize(request)
    _require_admin(request)
    return SafeModeState.status()


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    await _authorize(request)
    _require_admin(request)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def _ensure_correlation_id(request: Request, call_next):
    corr = request.headers.get("X-Correlation-ID") or str(time.time_ns())
    request.state.correlation_id = corr
    set_log_context(
        correlation_id=corr,
        route=request.url.path,
        user_agent=request.headers.get("user-agent", "")[:120],
    )
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr
    return response
# Enforce TLS/mTLS presence in strict mode
try:
    if os.getenv("ENABLE_MTLS", "false").strip().lower() in {"1", "true", "yes"}:
        cert = os.getenv("TLS_CERT_PATH")
        key = os.getenv("TLS_KEY_PATH")
        ca = os.getenv("TLS_CA_PATH")
        if not (cert and key and ca and Path(cert).exists() and Path(key).exists() and Path(ca).exists()):
            logger.error("mTLS enabled but cert/key/ca missing; enabling SAFE MODE")
            from .safe_mode import SafeModeState

            SafeModeState.enable("MTLS_MISSING")
except Exception:
    pass
# Optional ONNX Runtime acceleration and micro-batching
USE_ONNX = os.getenv("USE_ONNXRUNTIME", "false").strip().lower() in {"1", "true", "yes"}
ONNX_SUPERVISED: Optional[object] = None
ONNX_INPUT_NAME: Optional[str] = None
ONNX_BATCHER: Optional["_OnnxBatcher"] = None
if USE_ONNX:
    try:
        import onnxruntime as ort

        sup_path = Path(SERVICE.model_dir) / "supervised.onnx"
        if not sup_path.exists():
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                model = getattr(SERVICE, 'tabular_model', None)
                feat_names = getattr(SERVICE, 'feature_names', [])
                n_features = len(feat_names) if feat_names else None
                if model is not None and n_features and n_features > 0:
                    initial_type = [('input', FloatTensorType([None, n_features]))]
                    onx = convert_sklearn(model, initial_types=initial_type)
                    sup_path.parent.mkdir(parents=True, exist_ok=True)
                    sup_path.write_bytes(onx.SerializeToString())
                    logger.info("Exported supervised model to ONNX", extra={"path": str(sup_path), "n_features": n_features})
            except Exception as _exc:
                logger.warning("Failed to export ONNX: %s", _exc)
        if sup_path.exists():
            ONNX_SUPERVISED = ort.InferenceSession(str(sup_path))
            ONNX_INPUT_NAME = ONNX_SUPERVISED.get_inputs()[0].name  # type: ignore[attr-defined]
            logger.info("ONNX supervised model loaded", extra={"path": str(sup_path)})
            # enable micro-batching if requested
            if os.getenv("INFER_ONNX_BATCH", "false").strip().lower() in {"1", "true", "yes"}:
                try:
                    batch_size = int(os.getenv("INFER_BATCH_SIZE", "16"))
                    timeout_ms = int(os.getenv("INFER_BATCH_TIMEOUT_MS", "5"))
                    if batch_size > 1:
                        ONNX_BATCHER = None  # will be created after class definition
                except Exception as _exc:
                    logger.warning("Failed to parse batching env: %s", _exc)
    except Exception as exc:  # pragma: no cover
        logger.warning("ONNX Runtime unavailable: %s", exc)
        ONNX_SUPERVISED = None
        ONNX_INPUT_NAME = None


class _OnnxBatcher:
    def __init__(self, session: object, input_name: str, batch_size: int = 16, timeout_ms: int = 5):
        self._sess = session
        self._input = input_name
        self._batch = max(2, batch_size)
        self._timeout = max(1, timeout_ms) / 1000.0
        self._q: asyncio.Queue[Tuple[List[float], asyncio.Future]] = asyncio.Queue()
        self._task = asyncio.create_task(self._worker())

    async def run(self, vec: List[float]) -> float:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._q.put((vec, fut))
        return float(await fut)

    async def _worker(self) -> None:
        import numpy as _np
        while True:
            vecs: List[List[float]] = []
            futs: List[asyncio.Future] = []
            try:
                # wait for first item
                item = await asyncio.wait_for(self._q.get(), timeout=self._timeout)
                v, f = item
                vecs.append(v); futs.append(f)
            except asyncio.TimeoutError:
                continue
            # accumulate up to batch size within timeout
            start = time.perf_counter()
            while len(vecs) < self._batch:
                timeout = max(self._timeout - (time.perf_counter() - start), 0)
                if timeout <= 0:
                    break
                try:
                    v, f = await asyncio.wait_for(self._q.get(), timeout=timeout)
                    vecs.append(v); futs.append(f)
                except asyncio.TimeoutError:
                    break
            try:
                INFER_BATCH_SIZE.observe(len(vecs))
            except Exception:
                pass
            try:
                arr = _np.array(vecs, dtype=_np.float32)
                out = ONNX_SUPERVISED.run(None, {ONNX_INPUT_NAME: arr})  # type: ignore
                probs: List[float] = []
                if out and len(out) > 0:
                    y = out[0].ravel()
                    probs = [float(y[i]) for i in range(min(len(y), len(vecs)))]
                # fulfill futures
                for i, f in enumerate(futs):
                    if not f.done():
                        f.set_result(probs[i] if i < len(probs) else 0.0)
            except Exception as _exc:
                for f in futs:
                    if not f.done():
                        f.set_result(0.0)

# finalize ONNX_BATCHER creation if env enabled and session is loaded
if USE_ONNX and ONNX_SUPERVISED and ONNX_INPUT_NAME and os.getenv("INFER_ONNX_BATCH", "false").strip().lower() in {"1", "true", "yes"}:
    try:
        _bs = int(os.getenv("INFER_BATCH_SIZE", "16"))
        _tm = int(os.getenv("INFER_BATCH_TIMEOUT_MS", "5"))
        if _bs > 1:
            ONNX_BATCHER = _OnnxBatcher(ONNX_SUPERVISED, ONNX_INPUT_NAME, _bs, _tm)
    except Exception:
        ONNX_BATCHER = None
