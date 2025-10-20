"""Shared ensemble definitions with graceful fallback when backend package is unavailable."""

from __future__ import annotations

try:  # pragma: no cover - dynamic import
    from backend.ml_models.ensemble import EnsembleFraudDetector, RiskBreakdown  # type: ignore
except Exception:  # pragma: no cover - fallback definition
    import json
    import logging
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple

    import joblib
    import numpy as np
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - optional dependency
        ort = None  # type: ignore[assignment]

    logger = logging.getLogger(__name__)

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
                "supervised.joblib",
                self._fallback_supervised_model,
            )
            self.anomaly_model: IsolationForest = self._load_or_fit(
                "anomaly.joblib",
                self._fallback_anomaly_model,
            )
            self.deep_model: MLPRegressor = self._load_or_fit(
                "deep_autoencoder.joblib",
                self._fallback_autoencoder,
            )

            self.supervised_session: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]
            self.deep_session: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]
            self._ensure_onnx_sessions()

            self._background = self._load_background()

        def _load_metadata(self) -> Dict[str, str]:
            path = self.model_dir / "metadata.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return {"model_version": "0.1.0"}

        def _load_or_fit(self, filename: str, fallback):
            path = self.model_dir / filename
            if path.exists():
                return joblib.load(path)
            model = fallback()
            joblib.dump(model, path)
            return model

        def _ensure_onnx_sessions(self) -> None:
            self.supervised_session = self._load_onnx_session("supervised.onnx")
            self.deep_session = self._load_onnx_session("deep_autoencoder.onnx")

        def _load_onnx_session(self, filename: str) -> Optional["ort.InferenceSession"]:  # type: ignore[name-defined]
            if ort is None:
                return None
            path = self.model_dir / filename
            if not path.exists():
                return None
            try:
                return ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.warning("Impossible de charger le modèle ONNX %s: %s", path, exc)
                return None

        def _generate_synthetic_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
            rng = np.random.default_rng(1234)
            n_features = len(self.feature_names)
            X = rng.normal(loc=0, scale=1, size=(n_samples, n_features))
            weights = rng.uniform(-0.5, 0.5, size=(n_features,))
            weights[0] = 1.2
            weights[1] = 0.6
            weights[2] = 0.8
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

        def _predict_supervised_probability(self, vector: np.ndarray) -> float:
            if self.supervised_session is not None:
                inputs = {
                    self.supervised_session.get_inputs()[0].name: vector.astype(np.float32, copy=False)
                }
                outputs = self.supervised_session.run(None, inputs)
                probs = None
                for idx, meta in enumerate(self.supervised_session.get_outputs()):
                    if "prob" in meta.name.lower():
                        probs = outputs[idx]
                        break
                if probs is None and outputs:
                    probs = outputs[-1]
                if probs is not None:
                    probs = np.asarray(probs, dtype=float)
                    if probs.ndim == 1:
                        return float(probs[1] if probs.size > 1 else probs[0])
                    return float(probs[0][1] if probs.shape[1] > 1 else probs[0][0])
            return float(self.supervised_model.predict_proba(vector)[0][1])

        def _autoencoder_reconstruction(self, vector: np.ndarray) -> np.ndarray:
            if self.deep_session is not None:
                inputs = {self.deep_session.get_inputs()[0].name: vector.astype(np.float32, copy=False)}
                outputs = self.deep_session.run(None, inputs)
                reconstruction = np.asarray(outputs[0], dtype=float)
                if reconstruction.ndim == 1:
                    reconstruction = reconstruction.reshape(1, -1)
                return reconstruction
            return self.deep_model.predict(vector)

        def predict(self, features: Dict[str, float]):
            vector = self.vectorize(features)
            supervised_prob = self._predict_supervised_probability(vector)
            anomaly_score_raw = float(self.anomaly_model.decision_function(vector))
            anomaly_risk = float(np.clip(1.0 - (anomaly_score_raw + 0.5) / 2.0, 0.0, 1.0))
            reconstruction = self._autoencoder_reconstruction(vector)
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

        def __getstate__(self) -> Dict[str, object]:
            state = self.__dict__.copy()
            state["supervised_session"] = None
            state["deep_session"] = None
            return state

        def __setstate__(self, state: Dict[str, object]) -> None:
            self.__dict__.update(state)
            self._ensure_onnx_sessions()


__all__ = ["EnsembleFraudDetector", "RiskBreakdown"]
