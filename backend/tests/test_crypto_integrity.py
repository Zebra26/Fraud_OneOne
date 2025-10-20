import base64
import os
from pathlib import Path

import pytest

from backend.security.crypto_utils import (
    compute_file_hash,
    decrypt_file_aes256,
    encrypt_file_aes256,
    secure_delete,
)
from backend.security.model_integrity import compute_hashes, verify_hashes, write_hash_manifest


@pytest.fixture(autouse=True)
def aes_key_env(monkeypatch):
    key = base64.b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("MODELS_AES_KEY", key)
    yield


def test_aes_round_trip(tmp_path: Path):
    plaintext = tmp_path / "data.bin"
    plaintext.write_text("sensitive-data")
    encrypted = tmp_path / "data.bin.enc"
    decrypted = tmp_path / "data.bin.dec"

    encrypt_file_aes256(plaintext, encrypted)
    secure_delete(plaintext)

    decrypt_file_aes256(encrypted, decrypted)
    assert decrypted.read_text() == "sensitive-data"


def test_compute_hash(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")
    expected = compute_file_hash(sample)

    import hashlib

    manual = hashlib.sha256(b"hello world").hexdigest()
    assert expected == manual


def test_manifest_detection(tmp_path: Path):
    models_dir = tmp_path
    model_file = models_dir / "fraud_detection_advanced_model.pkl"
    model_file.write_bytes(b"model-binary")

    hashes = compute_hashes(models_dir)
    manifest = models_dir / "model_hashes.json"
    write_hash_manifest(hashes, manifest)

    assert verify_hashes(manifest) == []

    model_file.write_bytes(b"tampered")
    mismatches = verify_hashes(manifest)
    assert "fraud_detection_advanced_model.pkl" in mismatches

    model_file.unlink()
    mismatches = verify_hashes(manifest)
    assert "fraud_detection_advanced_model.pkl" in mismatches
