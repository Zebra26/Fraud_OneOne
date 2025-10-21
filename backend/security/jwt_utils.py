from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict

import jwt

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER", "fraud_k")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "fraud_api")


def _get_secret() -> bytes:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY environment variable is required")
    try:
        return base64.b64decode(secret)
    except Exception:
        return secret.encode("utf-8")


def generate_jwt(payload: Dict[str, Any], expires_in: int = 900) -> str:
    now = int(time.time())
    claims = {
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": now + expires_in,
        **payload,
    }
    token = jwt.encode(claims, _get_secret(), algorithm=JWT_ALGORITHM)
    return token


def verify_jwt(token: str) -> Dict[str, Any]:
    options = {"require": ["exp", "iat", "iss", "aud"]}
    decoded = jwt.decode(
        token,
        _get_secret(),
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
        options=options,
    )
    return decoded
