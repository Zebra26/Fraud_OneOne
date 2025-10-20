"""Trainer-side ensemble definitions (mirrors backend implementation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class RiskBreakdown:
    supervised_probability: float
    anomaly_risk: float
    deep_reconstruction_risk: float
    aggregated_risk: float


class EnsembleFraudDetector:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()
        self.feature_names: List[str] = self.metadata.get(
            "feature_names",
            ["amount", "velocity", "is_night", "country_risk", "new_device", "behavior_deviation"],
        )

        self.scaler: StandardScaler = self._load_or_fit("scaler.joblib", self._fallback_scaler)
        self.supervised_model: RandomForestClassifier = self._load_or_fit(
            "supervised.joblib", self._fallback_supervised_model
        )
        self.anomaly_model: IsolationForest = self._load_or_fit("anomaly.joblib", self._fallback_anomaly_model)
        self.deep_model: MLPRegressor = self._load_or_fit("deep_autoencoder.joblib", self._fallback_autoencoder)
        self._background = self._load_background()

    def _load_metadata(self) -> Dict[str, str]:
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"model_version": "0.1.0"}

    def _load_or_fit(self, filename: str, fallback):
        path = self.model_dir / filename
        if path.exists():
            return joblib.load(path)
        model = fallback()
        joblib.dump(model, path)
        return model

    def _generate_synthetic_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2025)
        n_features = len(self.feature_names)
        X = rng.normal(size=(n_samples, n_features))
        weights = rng.uniform(-0.5, 0.5, size=(n_features,))
        weights[0] = 1.1
        weights[1] = 0.6
        weights[2] = 0.5
        weights[4] = 1.0
        weights[5] = 0.9
        linear = X @ weights
        y = (linear > np.percentile(linear, 70)).astype(int)
        return X, y

    def _fallback_scaler(self) -> StandardScaler:
        X, _ = self._generate_synthetic_data()
        return StandardScaler().fit(X)

    def _fallback_supervised_model(self) -> RandomForestClassifier:
        X, y = self._generate_synthetic_data()
        X_scaled = self.scaler.transform(X)
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_scaled, y)
        model.feature_names_in_ = np.array(self.feature_names)  # type: ignore[attr-defined]
        return model

    def _fallback_anomaly_model(self) -> IsolationForest:
        X, _ = self._generate_synthetic_data()
        X_scaled = self.scaler.transform(X)
        model = IsolationForest(contamination=0.02, random_state=42)
        model.fit(X_scaled)
        return model

    def _fallback_autoencoder(self) -> MLPRegressor:
        X, _ = self._generate_synthetic_data()
        X_scaled = self.scaler.transform(X)
        model = MLPRegressor(hidden_layer_sizes=(32, 16, 32), activation="relu", random_state=42, max_iter=200)
        model.fit(X_scaled, X_scaled)
        return model

    def _load_background(self) -> np.ndarray:
        path = self.model_dir / "background.npy"
        if path.exists():
            return np.load(path)
        samples, _ = self._generate_synthetic_data(n_samples=256)
        scaled = self.scaler.transform(samples)
        np.save(path, scaled)
        return scaled

    def vectorize(self, features: Dict[str, float]) -> np.ndarray:
        if not self.feature_names:
            self.feature_names = list(features.keys())
        ordered = [features.get(name, 0.0) for name in self.feature_names]
        vector = np.array(ordered, dtype=float).reshape(1, -1)
        return self.scaler.transform(vector)

    def predict(self, features: Dict[str, float]) -> Tuple[float, RiskBreakdown]:
        vector = self.vectorize(features)
        supervised_prob = float(self.supervised_model.predict_proba(vector)[0][1])
        anomaly_score_raw = float(self.anomaly_model.decision_function(vector))
        anomaly_risk = float(np.clip(1.0 - (anomaly_score_raw + 0.5) / 2.0, 0.0, 1.0))
        reconstruction = self.deep_model.predict(vector)[0]
        reconstruction_error = float(np.linalg.norm(vector - reconstruction))
        deep_risk = float(np.clip(reconstruction_error / 5.0, 0.0, 1.0))
        aggregated = float(np.clip(0.6 * supervised_prob + 0.3 * anomaly_risk + 0.1 * deep_risk, 0.0, 1.0))

        breakdown = RiskBreakdown(
            supervised_probability=supervised_prob,
            anomaly_risk=anomaly_risk,
            deep_reconstruction_risk=deep_risk,
            aggregated_risk=aggregated,
        )
        return aggregated, breakdown

    def background(self, size: int = 100):
        if size >= len(self._background):
            return self._background
        idx = np.random.default_rng().choice(len(self._background), size=size, replace=False)
        return self._background[idx]

    @property
    def model_version(self) -> str:
        return self.metadata.get("model_version", "0.1.0")


__all__ = ["EnsembleFraudDetector", "RiskBreakdown"]

