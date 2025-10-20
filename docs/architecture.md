# Architecture globale

## Vue d'ensemble

### Phase I – Acquisition & Streaming
- Apache Kafka (topics `transactions`, `transactions.features`, `predictions`).
- Sécurisation: TLS/mTLS (`infrastructure/tls`), ACL Kafka, audit logs.
- Capteurs: données transactionnelles, signaux device, biométrie comportementale.

### Phase II – Feature Engineering
- Spark Structured Streaming (`services/spark/jobs/compute_features.py`) produit des vecteurs en <100 ms.
- Redis sert de Feature Store chaud (caching latence < 5 ms) tandis que Feast/Hive assurent l'historicité.
- Gestion du déséquilibre: SMOTE/anomalies injectées (pipeline trainer).

### Phase III – Cadre ML multicouche
- Couche 1 (supervisée): modèles arborescents (RandomForest fallback) entraînés via `services/trainer`.
- Couche 2 (anomalie): IsolationForest + autoencoder MLP (détection signaux rares, MITRE ATLAS coverage).
- Couche 3 (deep): réseaux profonds (placeholder) pour profils utilisateurs.
- Ensemble final: `backend/ml_models/ensemble.py` (pondération adaptative) exposant SHAP TreeExplainer.

### Phase IV – Décision, MLOps & Conformité
- API FastAPI (`backend/api`) + service d'inférence dédié (`services/ml-inference`).
- Observabilité: Prometheus/Grafana (`monitoring/`), boucle de feedback vers `services/trainer`.
- CI/CD: Docker/K8s manifests (`infrastructure/k8s`), intégration recommandée avec Jenkins + MLflow + DVC.
- Conformité: journalisation MongoDB (`explanations`), note AI Act, traçabilité SHAP.

## Chaîne XAI (AI Act)

1. Kafka reçoit la transaction et les features dérivés.
2. Backend déclenche l'ensemble ML et calcule les valeurs SHAP.
3. Résultat + explication stockés dans MongoDB pour audit.
4. Dashboards (monitoring/) présentent Précision, Rappel, latence, drift.
