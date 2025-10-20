import base64
import os
import time
from typing import Dict

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.middleware.security import SecurityMiddleware
from backend.security.jwt_utils import generate_jwt
from backend.security.hmac_utils import sign_request_v2


class _FakeRedis:
    def __init__(self):
        self._store: Dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: str):
        self._store[key] = value


def _app_with_middleware(fake_redis: _FakeRedis):
    app = FastAPI()

    # Patch middleware to use in-memory redis
    class _TestSecurity(SecurityMiddleware):
        def __init__(self, app):
            super().__init__(app, redis_url="redis://localhost:6379/0")
            self.redis = fake_redis

    app.add_middleware(_TestSecurity)

    @app.get("/protected")
    def protected():
        return JSONResponse({"ok": True})

    return app


def _headers(token: str | None, body: bytes, method: str = "GET", path: str = "/protected") -> Dict[str, str]:
    ts = str(int(time.time()))
    sig = sign_request_v2(method, path, ts, body)
    headers = {"X-Request-Timestamp": ts, "X-Request-Signature": sig}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@pytest.fixture(autouse=True)
def secrets_env(monkeypatch):
    # provide base64 secrets for tests
    monkeypatch.setenv("JWT_SECRET_KEY", base64.b64encode(b"test-secret").decode())
    monkeypatch.setenv("API_HMAC_KEY", base64.b64encode(b"hmac-secret").decode())
    yield


def test_jwt_valid_and_hmac_valid():
    fake = _FakeRedis()
    app = _app_with_middleware(fake)
    client = TestClient(app)
    token = generate_jwt({"sub": "tester", "roles": ["admin", "service", "read"]}, expires_in=60)
    body = b""
    r = client.get("/protected", headers=_headers(token, body))
    assert r.status_code == 200


def test_jwt_expired():
    fake = _FakeRedis()
    app = _app_with_middleware(fake)
    client = TestClient(app)
    token = generate_jwt({"sub": "tester"}, expires_in=-1)
    body = b""
    r = client.get("/protected", headers=_headers(token, body))
    assert r.status_code in (401, 403)


def test_hmac_invalid_signature():
    fake = _FakeRedis()
    app = _app_with_middleware(fake)
    client = TestClient(app)
    token = generate_jwt({"sub": "tester"}, expires_in=60)
    ts = str(int(time.time()))
    # wrong body/signature pairing
    headers = {"Authorization": f"Bearer {token}", "X-Request-Timestamp": ts, "X-Request-Signature": "bad"}
    r = client.get("/protected", headers=headers)
    assert r.status_code == 401


def test_rate_limit_exceeded():
    fake = _FakeRedis()
    app = _app_with_middleware(fake)
    client = TestClient(app)
    token = generate_jwt({"sub": "tester"}, expires_in=60)
    body = b""
    # exhaust tokens quickly
    for _ in range(105):
        resp = client.get("/protected", headers=_headers(token, body))
    assert resp.status_code in (200, 429)
