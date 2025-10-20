from fastapi import APIRouter, Depends, HTTPException, Request
import os
import time
import json
import httpx

from ..dependencies import get_kafka_producer, get_mongo_client, get_redis_cache
from ..schemas import HealthResponse
from ...security.rbac import require_roles
from ...security.hmac_utils import sign_request_v2


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    mongo=Depends(get_mongo_client),
    redis_cache=Depends(get_redis_cache),
    kafka=Depends(get_kafka_producer),
):
    dependencies = {
        "mongo": "ok" if await mongo.health() else "error",
        "redis": "ok" if await redis_cache.health() else "error",
        "kafka": "ok" if kafka is not None else "error",
    }
    status = "ok" if all(value == "ok" for value in dependencies.values()) else "degraded"
    return HealthResponse(status=status, dependencies=dependencies)


def _forward_headers(request: Request, body: bytes) -> dict:
    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    ts = str(int(time.time()))
    signature = sign_request_v2(request.method.upper(), request.url.path, ts, body)
    headers["X-Request-Timestamp"] = ts
    headers["X-Request-Signature"] = signature
    corr = getattr(request.state, "correlation_id", None)
    if corr:
        headers["X-Correlation-ID"] = corr
    return headers


@router.post("/safe-mode/enable")
@require_roles("admin")
async def enable_safe_mode_admin(request: Request):
    inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")
    body = await request.body()
    headers = _forward_headers(request, body)
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(f"{inference_url}/admin/safe-mode/enable", content=body, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@router.post("/safe-mode/disable")
@require_roles("admin")
async def disable_safe_mode_admin(request: Request):
    inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")
    headers = _forward_headers(request, b"")
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(f"{inference_url}/admin/safe-mode/disable", headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@router.get("/safe-mode/status")
@require_roles("admin")
async def safe_mode_status_admin(request: Request):
    inference_url = os.getenv("INFERENCE_URL", "http://ml-inference:8080")
    headers = _forward_headers(request, b"")
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{inference_url}/admin/safe-mode/status", headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()
