"""AES-256 GCM utilities for encrypting artifacts and PII fields."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import hmac

logger = logging.getLogger(__name__)

_KEY_ENV = "MODELS_AES_KEY"
_NONCE_SIZE = 12  # 96-bit nonce for AES-GCM
_PII_TOKEN_ENV = "PII_TOKEN_KEY"


def _decode_env_key(raw_key: str) -> bytes:
    """Decode a base64 or hex encoded key."""
    try:
        key = base64.b64decode(raw_key, validate=True)
    except Exception:
        try:
            key = bytes.fromhex(raw_key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("MODELS_AES_KEY must be base64 or hex encoded") from exc
    if len(key) != 32:
        raise ValueError("MODELS_AES_KEY must decode to 32 bytes (AES-256 key length)")
    return key


def get_aes_key() -> bytes:
    """Return the AES-256 key from environment."""
    raw_key = os.getenv(_KEY_ENV)
    if not raw_key:
        raise RuntimeError(f"Environment variable {_KEY_ENV} is required for model encryption")
    return _decode_env_key(raw_key.strip())


def encrypt_text(plain: str) -> str:
    """Encrypt a short plaintext string and return base64(nonce|ciphertext)."""
    key = get_aes_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)
    ct = aesgcm.encrypt(nonce, plain.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("ascii")


def pii_token(value: str) -> str:
    """Return a deterministic pseudonymization token for PII using HMAC-SHA256.

    Requires env PII_TOKEN_KEY (base64 or hex) to be set.
    """
    raw = os.getenv(_PII_TOKEN_ENV)
    if not raw:
        raise RuntimeError("PII_TOKEN_KEY must be set for PII tokenization")
    key = _decode_env_key(raw.strip())
    mac = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return mac


def encrypt_file_aes256(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Encrypt *src* into *dst* using AES-256-GCM."""
    key = get_aes_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)

    src_path = Path(src)
    data = src_path.read_bytes()
    ciphertext = aesgcm.encrypt(nonce, data, None)

    dst_path = Path(dst)
    dst_path.write_bytes(nonce + ciphertext)
    try:
        os.chmod(dst_path, 0o600)
    except OSError:  # pragma: no cover - platform specific
        pass


def decrypt_file_aes256(src: str | os.PathLike[str], dst: Optional[str | os.PathLike[str]] = None) -> bytes:
    """Decrypt *src* and optionally write plaintext to *dst*."""
    key = get_aes_key()
    aesgcm = AESGCM(key)

    src_path = Path(src)
    payload = src_path.read_bytes()
    if len(payload) <= _NONCE_SIZE:
        raise ValueError("Encrypted payload too short")

    nonce = payload[:_NONCE_SIZE]
    ciphertext = payload[_NONCE_SIZE:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    if dst is not None:
        dst_path = Path(dst)
        dst_path.write_bytes(plaintext)
        try:
            os.chmod(dst_path, 0o600)
        except OSError:  # pragma: no cover
            pass
    return plaintext


def secure_delete(path: str | os.PathLike[str]) -> None:
    """Best-effort secure delete (overwrite + unlink)."""
    file_path = Path(path)
    if not file_path.exists():
        return
    try:
        with file_path.open("r+b") as fh:
            length = fh.seek(0, os.SEEK_END)
            fh.seek(0)
            fh.write(os.urandom(length))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # pragma: no cover
                pass
        file_path.unlink()
    except OSError as exc:  # pragma: no cover
        logger.warning("Unable to securely delete %s: %s", file_path, exc)


def compute_file_hash(path: str | os.PathLike[str]) -> str:
    """Return SHA-256 hex digest of *path*."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
