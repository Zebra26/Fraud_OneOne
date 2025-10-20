import base64
import os
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.middleware.security import SecurityMiddleware
from backend.security.jwt_utils import generate_jwt


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None, nx=False):
        if nx:
            if key in self.store:
                return False
            self.store[key] = (value, time.time() + (ex or 0))
            return True
        self.store[key] = (value, time.time() + (ex or 0))
        return True

    def get(self, key):
        v = self.store.get(key)
        if not v:
            return None
        value, expiry = v
        if expiry and expiry < time.time():
            del self.store[key]
            return None
        return value


def _app_with_security(fake_redis: _FakeRedis):
    app = FastAPI()

    class _Sec(SecurityMiddleware):
        def __init__(self, app):
            super().__init__(app)
            self.redis = fake_redis
            self.require_jti = True

    app.add_middleware(_Sec)

    @app.get("/protected")
    def protected():
        return JSONResponse({"ok": True})

    return app


def test_jti_replay_protection(monkeypatch):
    # Provide HS256 secret for JWT
    monkeypatch.setenv("JWT_SECRET_KEY", base64.b64encode(b"test-secret-32-bytes-key-123456").decode())
    fake = _FakeRedis()
    app = _app_with_security(fake)
    client = TestClient(app)

    token = generate_jwt({"sub": "u1", "jti": "nonce-1"}, expires_in=60)
    headers = {"Authorization": f"Bearer {token}", "X-Request-Timestamp": str(int(time.time())), "X-Request-Signature": "stub"}
    # bypass HMAC by stubbing verification
    monkeypatch.setattr("backend.middleware.security.verify_request_v2", lambda *args, **kwargs: True)

    r1 = client.get("/protected", headers=headers)
    assert r1.status_code == 200

    r2 = client.get("/protected", headers=headers)
    assert r2.status_code == 401
    assert r2.json().get("detail") in {"Replay detected"}

