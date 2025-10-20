from __future__ import annotations

import atexit
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from backend.security.crypto_utils import compute_file_hash, decrypt_file_aes256, secure_delete
from backend.security.gpg_utils import gpg_verify_file
from backend.security.model_integrity import load_hash_manifest

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


class _LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class HybridInferenceService:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.decrypted_dir = Path(os.getenv("MODEL_DECRYPT_DIR", "/tmp/model_decrypted"))
        self.decrypted_dir.mkdir(parents=True, exist_ok=True)
        self._temp_paths: List[Path] = []
        atexit.register(self._cleanup)

        self.verify_hash = _env_flag("VERIFY_HASH_ON_LOAD", True)
        self.models_encrypted = _env_flag("MODELS_ENCRYPTED", True)
        self.verify_signature = _env_flag("VERIFY_SIGNATURE_ON_LOAD", True)

        self.public_key_path = os.getenv("GPG_PUBLIC_KEY_PATH")
        self.gpg_key_id = os.getenv("GPG_KEY_ID")
        if self.verify_signature and not (self.public_key_path and Path(self.public_key_path).exists()):
            logger.warning("Signature verification disabled: missing public key")
            self.verify_signature = False

        self.performance: Dict[str, Any] = {}
        self.hash_manifest: Dict[str, str] = load_hash_manifest(self.model_dir / "model_hashes.json")

        metadata_path = self.model_dir / "model_performance.json"
        if metadata_path.exists():
            self.performance = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not self.hash_manifest:
                self.hash_manifest = self.performance.get("model_hashes", {})
        else:
            logger.warning("model_performance.json not found in %s", self.model_dir)

        self.threshold: float = self.performance.get("optimal_threshold", 0.7)
        self.weights: Dict[str, float] = {
            "tabular": 0.7,
            "graph": 0.1,
            "autoencoder": 0.15,
            "lstm": 0.05,
        }
        self.weights.update(self.performance.get("weights", {}))
        self.model_version: str = self.performance.get("model_version", "unknown")
        self.feature_names: List[str] = self.performance.get("feature_names", [])
        self.artifacts_meta: Dict[str, Dict[str, str]] = {
            item["name"]: item for item in self.performance.get("artifacts", []) if isinstance(item, dict)
        }

        self.tabular_model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.graph_classifier = None
        self.autoencoder: Optional[_Autoencoder] = None
        self.autoencoder_scaler = None
        self.lstm: Optional[_LSTM] = None
        self.sequential_scaler = None

        self._load_models()

    def _cleanup(self) -> None:
        for path in self._temp_paths:
            try:
                secure_delete(path)
            except Exception:  # pragma: no cover
                pass

    def _resolve_artifact(self, filename: str, *, critical: bool = False) -> Tuple[Optional[Path], bool]:
        expected_hash = self.hash_manifest.get(filename)
        record = self.artifacts_meta.get(filename, {})
        enc_name = record.get("enc", f"{filename}.enc")
        sig_name = record.get("sig", f"{filename}.asc")
        enc_path = self.model_dir / enc_name
        sig_path = self.model_dir / sig_name
        decrypted_path = self.decrypted_dir / filename

        if self.models_encrypted and enc_path.exists():
            try:
                decrypt_file_aes256(enc_path, decrypted_path)
                self._temp_paths.append(decrypted_path)
                if self.verify_signature and sig_path.exists() and self.public_key_path:
                    if not gpg_verify_file(
                        decrypted_path,
                        sig_path,
                        public_key_path=self.public_key_path,
                        key_id=self.gpg_key_id,
                    ):
                        raise RuntimeError("Signature verification failed")
                elif self.verify_signature:
                    logger.warning("Missing signature for %s", filename)

                if self.verify_hash and expected_hash:
                    actual_hash = compute_file_hash(decrypted_path)
                    if actual_hash != expected_hash:
                        raise RuntimeError("Hash mismatch after decryption")

                logger.info("Decrypted and verified %s", filename)
                return decrypted_path, True
            except Exception as exc:
                logger.error("Failed to use encrypted artifact %s: %s", filename, exc)
                if decrypted_path.exists():
                    secure_delete(decrypted_path)

        plain_path = self.model_dir / filename
        if plain_path.exists():
            if self.verify_hash and expected_hash:
                actual_hash = compute_file_hash(plain_path)
                if actual_hash != expected_hash:
                    message = f"Hash mismatch for plaintext artifact {filename}"
                    if critical:
                        raise RuntimeError(message)
                    logger.warning(message)
            logger.warning("Using plaintext artifact for %s", filename)
            return plain_path, False

        if critical:
            raise RuntimeError(f"Required artifact missing: {filename}")
        logger.warning("Optional artifact %s unavailable", filename)
        return None, False

    def _load_models(self) -> None:
        path, temp = self._resolve_artifact("fraud_detection_advanced_model.pkl", critical=True)
        with path.open("rb") as handle:
            self.tabular_model = pickle.load(handle)
        if temp:
            secure_delete(path)

        path, temp = self._resolve_artifact("node2vec_model.pkl")
        if path:
            with path.open("rb") as handle:
                self.embeddings = pickle.load(handle)
            if temp:
                secure_delete(path)

        path, temp = self._resolve_artifact("optimized_fraud_detection_system.pkl")
        if path:
            with path.open("rb") as handle:
                self.graph_classifier = pickle.load(handle)
            if temp:
                secure_delete(path)

        scaler_path, scaler_temp = self._resolve_artifact("autoencoder_scaler.pkl")
        if scaler_path:
            with scaler_path.open("rb") as handle:
                self.autoencoder_scaler = pickle.load(handle)
            if scaler_temp:
                secure_delete(scaler_path)

        weights_path, weights_temp = self._resolve_artifact("fraud_autoencoder.pth")
        if weights_path and self.autoencoder_scaler is not None:
            input_dim = len(self.autoencoder_scaler.mean_)
            self.autoencoder = _Autoencoder(input_dim)
            state = torch.load(weights_path, map_location="cpu")
            self.autoencoder.load_state_dict(state)
            self.autoencoder.eval()
        if weights_temp and weights_path:
            secure_delete(weights_path)

        scaler_path, scaler_temp = self._resolve_artifact("sequential_scaler.pkl")
        if scaler_path:
            with scaler_path.open("rb") as handle:
                self.sequential_scaler = pickle.load(handle)
            if scaler_temp:
                secure_delete(scaler_path)

        lstm_path, lstm_temp = self._resolve_artifact("lstm_fraud_detector.pth")
        if lstm_path and self.sequential_scaler is not None:
            input_dim = len(self.sequential_scaler.mean_)
            self.lstm = _LSTM(input_dim)
            state = torch.load(lstm_path, map_location="cpu")
            self.lstm.load_state_dict(state)
            self.lstm.eval()
        if lstm_temp and lstm_path:
            secure_delete(lstm_path)

    def _prepare_tabular(self, features: Mapping[str, Any]) -> pd.DataFrame:
        if self.feature_names:
            ordered = {name: features.get(name) for name in self.feature_names}
        else:
            ordered = dict(features)
        return pd.DataFrame([ordered])

    def _score_tabular(self, features: Mapping[str, Any]) -> float:
        if self.tabular_model is None:
            return 0.0
        frame = self._prepare_tabular(features)
        prob = self.tabular_model.predict_proba(frame)[0][1]
        return float(prob)

    def _score_graph(self, graph: Optional[Mapping[str, str]]) -> float:
        if not graph or not self.embeddings:
            return 0.0
        source = graph.get("source_id")
        target = graph.get("target_id")
        if source is None or target is None:
            return 0.0
        emb_src = self.embeddings.get(source)
        emb_dst = self.embeddings.get(target)
        if emb_src is None or emb_dst is None:
            return 0.0
        if self.graph_classifier is not None:
            vec = np.concatenate([emb_src, emb_dst, np.abs(emb_src - emb_dst)])
            return float(self.graph_classifier.predict_proba([vec])[0][1])
        denom = np.linalg.norm(emb_src) * np.linalg.norm(emb_dst)
        if denom == 0:
            return 0.0
        similarity = float(np.dot(emb_src, emb_dst) / denom)
        return (1 - similarity) * 0.5 + 0.5

    def _score_autoencoder(self, features: Iterable[float]) -> float:
        if self.autoencoder is None or self.autoencoder_scaler is None:
            return 0.0
        array = np.asarray(list(features), dtype=np.float32)
        scaled = self.autoencoder_scaler.transform([array])[0]
        tensor = torch.from_numpy(scaled)
        with torch.no_grad():
            recon = self.autoencoder(tensor)
            loss = torch.mean((tensor - recon) ** 2).item()
        return float(min(1.0, loss * 10))

    def _score_lstm(self, sequence: Optional[Sequence[Iterable[float]]]) -> float:
        if self.lstm is None or self.sequential_scaler is None or not sequence:
            return 0.0
        seq_np = np.asarray([list(vec) for vec in sequence], dtype=np.float32)
        scaled = self.sequential_scaler.transform(seq_np)
        tensor = torch.from_numpy(scaled).unsqueeze(0)
        with torch.no_grad():
            prob = self.lstm(tensor).item()
        return float(prob)

    def infer(
        self,
        features: Mapping[str, Any],
        graph: Optional[Mapping[str, str]] = None,
        sequence: Optional[Sequence[Iterable[float]]] = None,
    ) -> Dict[str, Any]:
        tabular_score = self._score_tabular(features)
        graph_score = self._score_graph(graph)
        autoencoder_features = (
            [features.get(name, 0.0) for name in self.feature_names]
            if self.feature_names
            else list(features.values())
        )
        autoencoder_score = self._score_autoencoder(autoencoder_features)
        lstm_score = self._score_lstm(sequence)

        components = {
            "tabular": tabular_score,
            "graph": graph_score,
            "autoencoder": autoencoder_score,
            "lstm": lstm_score,
        }

        total_weight = sum(self.weights.values()) or 1.0
        combined = sum(self.weights[key] * components[key] for key in components) / total_weight

        decision = "FRAUD" if combined >= self.threshold else "NORMAL"

        return {
            "score": combined,
            "breakdown": components,
            "decision": decision,
            "threshold": self.threshold,
        }


@lru_cache(maxsize=1)
def get_service(model_dir: str = "/app/models") -> HybridInferenceService:
    return HybridInferenceService(model_dir=model_dir)
