import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedFraudModelService:
    """Load the advanced fraud detection pipeline and expose scoring helpers."""

    def __init__(self, model_path: str, performance_path: str):
        self.model_path = Path(model_path)
        self.performance_path = Path(performance_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)

        self.performance: Dict[str, Any] = {}
        if self.performance_path.exists():
            self.performance = json.loads(self.performance_path.read_text(encoding="utf-8"))

        self.expected_hash: str | None = self.performance.get("artifact_hash")
        self.feature_names: List[str] = self.performance.get("feature_names", [])
        self.thresholds: Dict[str, float] = self.performance.get(
            "thresholds",
            {"review": 0.6, "block": 0.85},
        )
        self.model_version: str = self.performance.get("model_version", "unknown")

        self.artifact_hash = self._compute_sha256(self.model_path)
        if self.expected_hash:
            if self.artifact_hash != self.expected_hash:
                raise ValueError(
                    f"Model artifact hash mismatch: expected {self.expected_hash}, got {self.artifact_hash}"
                )
        else:
            logger.warning(
                "No artifact hash provided for %s; computed hash=%s",
                self.model_path,
                self.artifact_hash,
            )

    def _build_frame(self, features: Dict[str, Any]) -> pd.DataFrame:
        if self.feature_names:
            ordered = {name: features.get(name) for name in self.feature_names}
        else:
            ordered = features
        df = pd.DataFrame([ordered])
        return df

    def predict_proba(self, features: Dict[str, Any]) -> float:
        frame = self._build_frame(features)
        proba = self.model.predict_proba(frame)[0][1]
        return float(proba)

    def decision(self, probability: float) -> str:
        if probability >= self.thresholds.get("block", 0.85):
            return "block"
        if probability >= self.thresholds.get("review", 0.6):
            return "review"
        return "approve"

    def model_metadata(self) -> Dict[str, Any]:
        metadata = {
            "model_version": self.model_version,
            "thresholds": self.thresholds,
            "feature_names": self.feature_names,
            "artifact_hash": self.artifact_hash,
            "raw": self.performance,
        }
        return metadata

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
