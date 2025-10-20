# MLOps & Gouvernance

## Pipelines
- **Entraînement** : `services/trainer/train_pipeline.py` (placeholder) orchestre l'ingénierie des données, le réentraînement LightGBM/XGBoost (à brancher) et la validation (Précision, Rappel, AUC, latence d'inférence).
- **Validation** : tests unitaires (FastAPI), tests de dérive (Kolmogorov-Smirnov), tests adversariaux (MITRE ATLAS).
- **Déploiement** : artefacts versionnés (DVC/MLflow via `models/mlruns/`), conteneurisation Docker, déploiement GitOps (ArgoCD ou Jenkins).

## Observabilité
- Métriques temps réel: latence p50/p95, throughput (transactions/min), taux de faux positifs/faux négatifs.
- Logs: score, décision, note SHAP, identifiant opérateur (si escalade).
- Alertes: dérive > seuil, latence > 120 ms, indisponibilité Kafka/Redis.

## Conformité & Auditabilité
- Traçabilité AI Act via stockage des contributions SHAP.
- Conservation des décisions 10 ans (recommandation bancaire) avec anonymisation/rétention adaptatives.
- Replays possibles depuis Kafka (mode compacté) pour investigations.
