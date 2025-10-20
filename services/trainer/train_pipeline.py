"""Offline training pipeline orchestrating the multi-layer models."""

import json
import os
import types
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import mlflow
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import classification_report, roc_auc_score

from ensemble import EnsembleFraudDetector, RiskBreakdown
from backend.security.crypto_utils import encrypt_file_aes256, secure_delete
from backend.security.gpg_utils import gpg_sign_file
from backend.security.model_integrity import compute_hashes, write_hash_manifest

# Align module paths so persisted objects load in backend (`backend.ml_models.ensemble`).
backend_module = types.ModuleType("backend")
ml_models_module = types.ModuleType("backend.ml_models")
ensemble_module = types.ModuleType("backend.ml_models.ensemble")
ensemble_module.EnsembleFraudDetector = EnsembleFraudDetector
ensemble_module.RiskBreakdown = RiskBreakdown
ml_models_module.ensemble = ensemble_module  # type: ignore[attr-defined]
backend_module.ml_models = ml_models_module  # type: ignore[attr-defined]

sys_modules = {
    "backend": backend_module,
    "backend.ml_models": ml_models_module,
    "backend.ml_models.ensemble": ensemble_module,
}
import sys
for name, module in sys_modules.items():
    sys.modules.setdefault(name, module)

EnsembleFraudDetector.__module__ = "backend.ml_models.ensemble"  # type: ignore[attr-defined]
RiskBreakdown.__module__ = "backend.ml_models.ensemble"  # type: ignore[attr-defined]


DATA_PATH = Path(os.getenv("TRAIN_DATA_PATH", "./data/training.parquet"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "../../models"))




def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    rng = np.random.default_rng(2024)
    rows = 10000
    data = {
        "amount": rng.lognormal(mean=3, sigma=0.5, size=rows),
        "velocity": rng.random(rows),
        "is_night": rng.integers(0, 2, size=rows),
        "country_risk": rng.random(rows),
        "new_device": rng.integers(0, 2, size=rows),
        "behavior_deviation": rng.random(rows),
    }
    df = pd.DataFrame(data)
    target = (df["amount"] > df["amount"].quantile(0.95)) & (df["new_device"] == 1)
    df["label"] = target.astype(int)
    return df


def compute_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        f1 = report["1"]["f1-score"]
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        score = 0.5 * f1 + 0.25 * precision + 0.25 * recall
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def train() -> Dict[str, float]:
    df = load_dataset()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = [col for col in df.columns if col != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    model = EnsembleFraudDetector(MODEL_DIR)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/workspace/mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
    run_name = os.getenv("MLFLOW_RUN_NAME", "trainer-run")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_version": model.metadata.get("model_version", "0.1.0"),
                "n_estimators": getattr(model.supervised_model, "n_estimators", None),
                "max_depth": getattr(model.supervised_model, "max_depth", None),
                "feature_count": len(feature_cols),
                "dataset_size": len(df),
            }
        )

        model.feature_names = feature_cols
        model.scaler.fit(X)
        joblib.dump(model.scaler, MODEL_DIR / "scaler.joblib")

        model.supervised_model.fit(model.scaler.transform(X), y)
        joblib.dump(model.supervised_model, MODEL_DIR / "supervised.joblib")

        model.anomaly_model.fit(model.scaler.transform(X[y == 0]))
        joblib.dump(model.anomaly_model, MODEL_DIR / "anomaly.joblib")

        model.deep_model.fit(model.scaler.transform(X), model.scaler.transform(X))
        joblib.dump(model.deep_model, MODEL_DIR / "deep_autoencoder.joblib")

        # Export ONNX artefacts for accelerated inference
        initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]
        try:
            supervised_onnx = convert_sklearn(model.supervised_model, initial_types=initial_type)
            (MODEL_DIR / "supervised.onnx").write_bytes(supervised_onnx.SerializeToString())
        except Exception as exc:  # pragma: no cover - conversion optional
            print("ONNX export failed for supervised model:", exc)

        try:
            deep_onnx = convert_sklearn(model.deep_model, initial_types=initial_type)
            (MODEL_DIR / "deep_autoencoder.onnx").write_bytes(deep_onnx.SerializeToString())
        except Exception as exc:  # pragma: no cover - conversion optional
            print("ONNX export failed for autoencoder:", exc)

        joblib.dump(model, MODEL_DIR / "ensemble_detector.joblib")
        print("MODEL_MODULE", model.__class__.__module__)

        preds = model.supervised_model.predict_proba(model.scaler.transform(X))[:, 1]
        auc = roc_auc_score(y, preds)
        report = classification_report(y, preds > 0.5, output_dict=True)
        optimal_threshold = compute_optimal_threshold(y, preds)

        metrics = {
            "auc": auc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "optimal_threshold": optimal_threshold,
        }
        mlflow.log_metrics(metrics)

        metadata_payload = {
            "model_version": model.metadata.get("model_version", "0.1.0"),
            "feature_names": feature_cols,
            "auc": auc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "optimal_threshold": optimal_threshold,
        }
        (MODEL_DIR / "metadata.json").write_text(
            json.dumps(metadata_payload, indent=2),
            encoding="utf-8",
        )

        (MODEL_DIR / "optimal_threshold.txt").write_text(f"{optimal_threshold}\n", encoding="utf-8")

        hashes = compute_hashes(MODEL_DIR)
        write_hash_manifest(hashes, MODEL_DIR / "model_hashes.json")
        for name, digest in hashes.items():
            mlflow.log_param(f"hash_{name}", digest)

        performance_payload = {
            "model_version": metadata_payload["model_version"],
            "feature_names": feature_cols,
            "optimal_threshold": optimal_threshold,
            "weights": {"tabular": 0.7, "graph": 0.1, "autoencoder": 0.15, "lstm": 0.05},
            "model_hashes": hashes,
        }
        performance_payload["encrypted"] = True
        performance_payload["signed"] = True
        performance_payload["artifacts"] = artifact_records
        primary_hash = hashes.get("fraud_detection_advanced_model.pkl") or hashes.get("ensemble_detector.joblib")
        if primary_hash:
            performance_payload["artifact_hash"] = primary_hash

        performance_path = MODEL_DIR / "model_performance.json"
        performance_path.write_text(json.dumps(performance_payload, indent=2), encoding="utf-8")

        required_env = {
            "MODELS_AES_KEY": os.getenv("MODELS_AES_KEY"),
            "GPG_PRIVATE_KEY_PATH": os.getenv("GPG_PRIVATE_KEY_PATH"),
            "GPG_KEY_ID": os.getenv("GPG_KEY_ID"),
            "GPG_KEY_PASS": os.getenv("GPG_KEY_PASS"),
        }
        missing_env = [name for name, value in required_env.items() if not value]
        if missing_env:
            raise RuntimeError(f"Missing security environment variables: {', '.join(missing_env)}")

        private_key_path = required_env["GPG_PRIVATE_KEY_PATH"]
        key_id = required_env["GPG_KEY_ID"]
        key_pass = required_env["GPG_KEY_PASS"]

        artifacts_to_secure = [
            "fraud_detection_advanced_model.pkl",
            "node2vec_model.pkl",
            "fraud_autoencoder.pth",
            "autoencoder_scaler.pkl",
            "lstm_fraud_detector.pth",
            "sequential_scaler.pkl",
            "scaler.joblib",
            "supervised.joblib",
            "anomaly.joblib",
            "deep_autoencoder.joblib",
            "supervised.onnx",
            "deep_autoencoder.onnx",
            "ensemble_detector.joblib",
        ]

        encrypted_files: list[str] = []
        signature_files: list[str] = []
        artifact_records: list[dict[str, str]] = []
        for artifact_name in artifacts_to_secure:
            artifact_path = MODEL_DIR / artifact_name
            if not artifact_path.exists():
                continue
            signature_path = gpg_sign_file(
                artifact_path,
                private_key_path=private_key_path,
                key_id=key_id,
                passphrase=key_pass,
            )
            enc_path = artifact_path.with_suffix(artifact_path.suffix + ".enc")
            encrypt_file_aes256(artifact_path, enc_path)
            secure_delete(artifact_path)
            encrypted_files.append(enc_path.name)
            signature_files.append(signature_path.name)
            artifact_records.append({"name": artifact_name, "enc": enc_path.name, "sig": signature_path.name})

        artifacts_to_log = [
            "model_hashes.json",
            "model_performance.json",
            "metadata.json",
            "optimal_threshold.txt",
        ] + encrypted_files + signature_files
        for artifact_name in artifacts_to_log:
            artifact_path = MODEL_DIR / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

    return metrics


if __name__ == "__main__":
    metrics = train()
    print("Training metrics", metrics)
