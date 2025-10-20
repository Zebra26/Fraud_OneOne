"""Helpers for signing model artifacts with GPG."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import gnupg

logger = logging.getLogger(__name__)

_DEFAULT_GPG_HOME = Path(os.getenv("GPG_HOME", "/tmp/fraud_k_gpg"))


@lru_cache(maxsize=1)
def _get_gpg() -> gnupg.GPG:
    home = _DEFAULT_GPG_HOME
    home.mkdir(parents=True, exist_ok=True)
    return gnupg.GPG(gnupghome=str(home))


@lru_cache(maxsize=1)
def _private_key_loaded(private_key_path: str) -> bool:
    gpg = _get_gpg()
    key_data = Path(private_key_path).read_text(encoding="utf-8")
    import_result = gpg.import_keys(key_data)
    if not import_result.count:
        raise RuntimeError(f"Failed to import GPG private key from {private_key_path}")
    return True


def gpg_sign_file(
    file_path: str | os.PathLike[str],
    signature_path: Optional[str | os.PathLike[str]] = None,
    *,
    private_key_path: str,
    key_id: str,
    passphrase: str,
) -> Path:
    """Create a detached signature for *file_path* and return the signature Path."""
    if not private_key_path:
        raise RuntimeError("GPG_PRIVATE_KEY_PATH is required to sign artifacts")
    if not key_id:
        raise RuntimeError("GPG_KEY_ID is required to sign artifacts")
    if not passphrase:
        raise RuntimeError("GPG_KEY_PASS is required to sign artifacts")

    gpg = _get_gpg()
    _private_key_loaded(private_key_path)

    src = Path(file_path)
    if signature_path is None:
        signature_path = src.with_suffix(src.suffix + ".asc")
    dst = Path(signature_path)

    with src.open("rb") as fh:
        result = gpg.sign_file(
            fh,
            keyid=key_id,
            passphrase=passphrase,
            detach=True,
            output=str(dst),
        )
    if not result or not dst.exists():
        raise RuntimeError(f"Failed to sign artifact {src}")
    try:
        os.chmod(dst, 0o600)
    except OSError:  # pragma: no cover - platform specific
        pass
    logger.info("Signed artifact %s -> %s", src.name, dst.name)
    return dst


@lru_cache(maxsize=1)
def _public_key_loaded(public_key_path: str) -> bool:
    gpg = _get_gpg()
    key_data = Path(public_key_path).read_text(encoding="utf-8")
    result = gpg.import_keys(key_data)
    if not result.count:
        raise RuntimeError(f"Failed to import GPG public key from {public_key_path}")
    return True


def gpg_verify_file(
    file_path: str | os.PathLike[str],
    signature_path: str | os.PathLike[str],
    *,
    public_key_path: str,
    key_id: str | None = None,
) -> bool:
    gpg = _get_gpg()
    _public_key_loaded(public_key_path)

    sig_path = Path(signature_path)
    with sig_path.open("rb") as sig_handle:
        verified = gpg.verify_file(sig_handle, str(file_path))

    if not verified or not verified.valid:
        logger.error("Signature verification failed for %s", file_path)
        return False

    if key_id and verified.username and key_id not in verified.username:
        logger.warning("Signature verified but key id mismatch for %s", file_path)

    logger.info("Signature verified for %s", Path(file_path).name)
    return True
