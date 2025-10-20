from __future__ import annotations

import json
import logging
import time
from typing import Callable

import os
import redis
from fastapi import Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram

from ..security.hmac_utils import verify_request_v2
from ..security.jwt_utils import verify_jwt

logger = logging.getLogger(__name__)

JWT_VALIDATIONS = Counter("jwt_validations_total", "Total JWT validations")
JWT_FAILURES = Counter("jwt_failures_total", "Total JWT failures")
HMAC_VALIDATIONS = Counter("hmac_verifications_total", "Total HMAC verifications")
RATE_LIMIT_BLOCK = Counter("rate_limit_block_total", "Requests blocked by rate limiter")
AUTH_FAILURES = Counter("auth_failures_total", "Authentication failures", ["reason"])
REQUEST_LATENCY = Histogram(
    "request_latency_ms",
    "Request latency (ms)",
    buckets=[1, 5, 10, 20, 50, 100, 200, 400, 1000],
)


class SecurityMiddleware:
    def __init__(self, app, redis_url: str = "redis://redis-cluster:6379/0"):
        self.app = app
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.capacity = int(os.getenv("RATE_LIMIT_CAPACITY", "100"))
        self.refill_per_min = int(os.getenv("RATE_LIMIT_REFILL_PER_MIN", "50"))
        self.allowed_skew = int(os.getenv("ALLOWED_SKEW_SECONDS", "60"))
        self.require_jti = os.getenv("JWT_REQUIRE_JTI", "false").strip().lower() in {"1", "true", "yes"}

    async def __call__(self, request: Request, call_next: Callable):
        start = time.perf_counter()

        if request.url.path in {"/health", "/models/info"}:
            return await call_next(request)

        correlation_id = request.headers.get("X-Correlation-ID", str(time.time_ns()))
        request.state.correlation_id = correlation_id

        # JWT Verification
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            JWT_FAILURES.inc()
            AUTH_FAILURES.labels(reason="missing_bearer").inc()
            return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
        token = auth_header.split(" ", 1)[1]
        try:
            claims = verify_jwt(token)
            request.state.jwt_claims = claims
            JWT_VALIDATIONS.inc()
            # Optional JTI replay prevention
            if self.require_jti:
                jti = claims.get("jti")
                exp = int(claims.get("exp", 0))
                if not jti:
                    AUTH_FAILURES.labels(reason="missing_jti").inc()
                    return JSONResponse({"detail": "Missing JTI"}, status_code=401)
                ttl = max(0, exp - int(time.time()))
                if ttl <= 0:
                    AUTH_FAILURES.labels(reason="jti_expired").inc()
                    return JSONResponse({"detail": "Token expired"}, status_code=401)
                # Use SET NX to prevent replay
                if not self.redis.set(f"jti:{jti}", "1", ex=ttl, nx=True):
                    AUTH_FAILURES.labels(reason="jti_replay").inc()
                    return JSONResponse({"detail": "Replay detected"}, status_code=401)
        except Exception as exc:
            JWT_FAILURES.inc()
            AUTH_FAILURES.labels(reason="jwt_invalid").inc()
            logger.warning("JWT verification failed: %s", exc)
            return JSONResponse({"detail": "Invalid token"}, status_code=401)

        # HMAC verification
        signature = request.headers.get("X-Request-Signature")
        timestamp = request.headers.get("X-Request-Timestamp")
        nonce = request.headers.get("X-Request-Nonce")
        if not signature or not timestamp:
            AUTH_FAILURES.labels(reason="missing_signature").inc()
            return JSONResponse({"detail": "Missing signature headers"}, status_code=401)
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > self.allowed_skew:
                AUTH_FAILURES.labels(reason="signature_expired").inc()
                return JSONResponse({"detail": "Signature expired"}, status_code=401)
        except ValueError:
            AUTH_FAILURES.labels(reason="invalid_timestamp").inc()
            return JSONResponse({"detail": "Invalid timestamp"}, status_code=401)
        body = await request.body()
        method = request.method.upper()
        path = request.url.path
        if not verify_request_v2(method, path, timestamp, body, signature, nonce=nonce):
            AUTH_FAILURES.labels(reason="invalid_signature").inc()
            return JSONResponse({"detail": "Invalid signature"}, status_code=401)
        HMAC_VALIDATIONS.inc()

        # Idempotency-Key validation and device binding
        if method == "POST" and path == "/predictions/score":
            idemp = request.headers.get("Idempotency-Key")
            if not idemp:
                return JSONResponse({"detail": "Missing Idempotency-Key"}, status_code=400)
            if not self.redis.set(f"idemp:{idemp}", "1", ex=600, nx=True):
                return JSONResponse({"detail": "Duplicate request"}, status_code=409)
            device_id = request.headers.get("X-Device-ID")
            if not device_id or not device_id.isalnum():
                return JSONResponse({"detail": "Invalid or missing X-Device-ID"}, status_code=400)
            # Device binding rate limit by (user_id, device_id, ip_cidr)
            user_id = str(claims.get("sub", "anon"))
            ip = request.client.host if request.client else "0.0.0.0"
            ip_cidr = ".".join(ip.split(".")[:3]) + ".0/24" if "." in ip else ip
            bucket_key = f"ratelimit_dev:{user_id}:{device_id}:{ip_cidr}"
            now = time.time()
            bucket = self.redis.get(bucket_key)
            if bucket is None:
                tokens = self.capacity - 1
            else:
                tokens, timestamp_str = bucket.split(":")
                tokens = int(tokens)
                last = float(timestamp_str)
                refill = int((now - last) / 60 * self.refill_per_min)
                tokens = min(self.capacity, tokens + refill) - 1
            if tokens < 0:
                RATE_LIMIT_BLOCK.inc()
                return JSONResponse({"detail": "Device rate limit exceeded"}, status_code=429)
            self.redis.set(bucket_key, f"{tokens}:{now}")

        # Rate limiting
        key = f"rate_limit:{token}"
        now = time.time()
        bucket = self.redis.get(key)
        if bucket is None:
            tokens = self.capacity - 1
        else:
            tokens, timestamp_str = bucket.split(":")
            tokens = int(tokens)
            last = float(timestamp_str)
            refill = int((now - last) / 60 * self.refill_per_min)
            tokens = min(self.capacity, tokens + refill) - 1
        if tokens < 0:
            RATE_LIMIT_BLOCK.inc()
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        self.redis.set(key, f"{tokens}:{now}")

        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY.observe(duration_ms)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
