import importlib
import os


def test_safe_mode_tls_failure(monkeypatch):
    monkeypatch.setenv("ENABLE_MTLS", "true")
    # Ensure cert paths are not set or invalid
    monkeypatch.delenv("TLS_CERT_PATH", raising=False)
    monkeypatch.delenv("TLS_KEY_PATH", raising=False)
    monkeypatch.delenv("TLS_CA_PATH", raising=False)

    # Reload module to trigger SAFE MODE check
    mod = importlib.import_module("services.ml-inference.app")
    importlib.reload(mod)

    SafeModeState = importlib.import_module("services.ml-inference.safe_mode").SafeModeState
    assert SafeModeState.is_enabled() is True

