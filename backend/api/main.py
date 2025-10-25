import logging
import os
import sys

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from ..middleware.security import SecurityMiddleware
from ..security.rbac import require_roles

from .config import get_settings
from .dependencies import get_model_service
from .routers import admin, predictions, transactions, auth
from ..monitoring.metrics import PrometheusMiddleware
from ..middleware.load_shed import LoadShedMiddleware
from ..logging_config import configure_json_logging, set_log_context


try:  # pragma: no cover - depends on optional dependency
    import uvloop
except ImportError:  # pragma: no cover - uvloop not installed
    uvloop = None

if uvloop is not None and sys.platform.startswith("linux"):
    uvloop.install()

configure_json_logging(service="backend")

settings = get_settings()

app = FastAPI(
    title="Fraud Detection Realtime API",
    version="0.2.0",
    description="API FastAPI pour scoring de fraude avec MLflow tracking et suivi des metriques.",
)

origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoadShedMiddleware)

app.include_router(transactions.router)
app.include_router(predictions.router)
app.include_router(admin.router)
app.include_router(auth.router)


@app.get("/", tags=["meta"])
def root():
    service = get_model_service()
    return {
        "message": "Fraud detection realtime backend",
        "environment": settings.app_env,
        "model_version": service.model_version,
        "review_threshold": service.thresholds.get("review"),
        "block_threshold": service.thresholds.get("block"),
    }


@app.get("/metrics", include_in_schema=False)
@require_roles("admin")
def metrics(request: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def _bind_log_context(request: Request, call_next):
    # Bind per-request context for JSON logs
    corr = getattr(request.state, "correlation_id", None) or request.headers.get("X-Correlation-ID")
    set_log_context(
        correlation_id=corr,
        route=request.url.path,
        user_agent=request.headers.get("user-agent", "")[:120],
    )
    return await call_next(request)


@app.get("/models/info", tags=["models"])
async def get_model_info():
    inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{inference_url}/models/info")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.on_event("startup")
async def _init_http_client():
    # Shared HTTP client for upstream calls (keep-alive)
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    timeout = httpx.Timeout(5.0, connect=2.0)
    app.state.http_client = httpx.AsyncClient(limits=limits, timeout=timeout)


@app.on_event("shutdown")
async def _close_http_client():
    client = getattr(app.state, "http_client", None)
    if client is not None:
        await client.aclose()
