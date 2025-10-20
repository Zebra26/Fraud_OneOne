from __future__ import annotations

import base64
import hashlib
import hmac
import os


def _get_hmac_key() -> bytes:
    raw = os.getenv("API_HMAC_KEY")
    if not raw:
        raise RuntimeError("API_HMAC_KEY environment variable is required")
    try:
        return base64.b64decode(raw)
    except Exception:
        return raw.encode("utf-8")


def _payload_v2(method: str, path: str, timestamp: str, body: bytes, nonce: str | None = None) -> bytes:
    nonce_part = nonce or ""
    return f"{method}|{path}|{timestamp}|{nonce_part}|".encode("utf-8") + (body or b"")


def sign_request_v2(method: str, path: str, timestamp: str, body: bytes, nonce: str | None = None) -> str:
    """Return base64 signature of HMAC-SHA256(method|path|timestamp|body)."""
    key = _get_hmac_key()
    mac = hmac.new(key, _payload_v2(method, path, timestamp, body, nonce), hashlib.sha256).digest()
    return base64.b64encode(mac).decode("ascii")


def verify_request_v2(method: str, path: str, timestamp: str, body: bytes, signature: str, nonce: str | None = None) -> bool:
    key = _get_hmac_key()
    data = _payload_v2(method, path, timestamp, body, nonce)
    # Primary: base64 signature
    try:
        expected = hmac.new(key, data, hashlib.sha256).digest()
        provided = base64.b64decode(signature)
        if hmac.compare_digest(expected, provided):
            return True
    except Exception:
        pass
    # Backward compatibility: hex signature over (timestamp + b'.' + body)
    try:
        legacy = f"{timestamp}.".encode("utf-8") + (body or b"")
        legacy_expected = hmac.new(key, legacy, hashlib.sha256).hexdigest()
        if hmac.compare_digest(legacy_expected, signature):
            return True
    except Exception:
        pass
    return False
