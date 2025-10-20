# Spark Jobs

- Real-time feature engineering (`jobs/compute_features.py`) for behavioural/device signals.
- Triggered via Docker Compose or Kubernetes CronJob.
- Writes enriched vectors to Kafka and caches aggregates for sub-100 ms scoring.

