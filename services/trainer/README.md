# Trainer

- Batch retraining pipeline (`train_pipeline.py`) exporting the layered detector artefacts.
- Produces scaler, supervised classifier, anomaly detector, autoencoder, and metadata (AUC/precision/recall).
- Outputs stored in `models/` directory and consumed by backend + inference services.
- Integrates adversarial samples / class imbalance strategies (extend with SMOTE, synthetic fraud injection).

