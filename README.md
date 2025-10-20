[![Validate Observability](https://github.com/Zebra26/Fraud_One/actions/workflows/observability.yml/badge.svg)](https://github.com/Zebra26/Fraud_One/actions/workflows/observability.yml)

# fraud-detection-realtime

Real-time fraud detection platform for mobile banking (Account Takeover mitigation, regulatory compliant).

- Backend: FastAPI (Python)
- Streaming: Apache Kafka (transactions, predictions) running in KRaft mode (broker + controller)
- Storage: MongoDB (events), Redis Cluster (feature cache, high availability)
- Inference: Dedicated ML service with SHAP-based explainability (AI Act alignment)
- Orchestration: Docker Compose + Kubernetes manifests
- Artefacts: `models/ensemble_detector.joblib`, `models/optimal_threshold.txt`, MLflow runs under `models/mlruns/`

## Non-functional targets

- Latency: 50-100 ms end-to-end scoring
- Scalability: millions of transactions per minute (horizontal scale + Kafka partitions)
- Quality: optimise precision and recall (tracked via Prometheus/Grafana)
- Security: mTLS, access control, MITRE ATLAS aligned monitoring

## Getting started (DEV)

1. Copy `.env.example` to `.env` and adapt values.
2. Launch stack: `docker compose -f docker-compose.dev.yml up --build`
3. Without Docker: create a virtualenv, install `backend/requirements.txt`, run `uvicorn backend.api.main:app --reload`

Useful services once the stack is up:
- Kafka UI (Provectus) on http://localhost:8080 connected to `kafka:9092` (verify topics `transactions`, `transactions.features`, `predictions` exist)
- RedisInsight on http://localhost:8001 - for dev compose use cluster nodes `redis-cluster:7000-7005` (or add `redis-cluster:7000, redis-cluster:7001, redis-cluster:7002`), inspect cached feature vectors and risk scores
- Kafka broker accessible externally on `localhost:9094` (internal clients use `kafka:9092`)
- Prometheus on http://localhost:9090 (scrapes backend & inference metrics on `/metrics`); Grafana on http://localhost:3000 (default `admin`/`admin`)
- MLflow UI on http://localhost:5000 reading from shared `./mlruns` (trainer logs params/metrics/artifacts automatically)
- Spark feature-engineering job runs from a custom image (`services/spark/Dockerfile`) and needs outbound access to download Kafka connector jars the first time it starts.
- FastAPI services automatically enable `uvloop` when running on Linux/Docker for lower-latency async IO.
- Backend data layer relies on Motor (async MongoDB) and `redis.asyncio` Cluster clients, so endpoints are fully `async def`.
- RandomForest classifier and autoencoder are exported to ONNX during training and served with onnxruntime for accelerated inference.

## Observability (Prometheus + Alertmanager + Grafana + Loki)

- Compose overlay: `docker-compose.observability.yml`
- Services: Prometheus (9090), Alertmanager (9093), Grafana (3000), Loki (3100), Promtail, node-exporter (9100), cAdvisor (8081)

Run with dev stack:
- docker compose -f docker-compose.dev.yml -f docker-compose.observability.yml up -d prometheus alertmanager grafana loki promtail
- (Linux hosts) add: `node-exporter cadvisor`

Dashboards and datasources:
- Grafana auto-provisions Prometheus and Loki datasources.
- Add dashboards via UI or provisioning if needed.

Alerts:
- Rules in `monitoring/alerts.yml`, delivered to Alertmanager (`monitoring/alertmanager.yml`). Receiver is a no-op by default; configure Slack/webhooks as needed.

Logs:
- Promtail scrapes Docker JSON logs and pushes to Loki. Correlation ID is extracted when present.

Notes:
- node-exporter and cAdvisor mounts may not work on Docker Desktop for Windows/macOS; they are intended for Linux hosts.

## RBAC and Auth
- JWT must include a `roles` claim, e.g. `{ "roles": ["admin","service","read"] }`.
- Admin-only routes use `require_roles("admin")`.
- Backend forwards JWT + HMAC to the inference service.

## Secrets and Rotation (CI/CD)
- Store `JWT_SECRET_KEY` and `API_HMAC_KEY` in Docker/K8s secrets or Vault.
- Enforce MFA and approvals for rotation. Rotate keys by updating secrets and restarting deployments.
- Do not commit real certificates; use `security/certs/` only for dev self-signed testing.

Verification
- Compile check: `python -m compileall backend services/ml-inference`
- Tests: `pytest backend/tests/test_auth_security.py`

## Core endpoints

- POST `/transactions/ingest` – ingest a transaction (Kafka + Mongo persistence)
- POST `/predictions/score` – scoring with SHAP explanations and risk breakdown
- GET `/admin/health` – service health status

### Shared model artefacts

- A named Docker volume `fraud_k_models_data` stores model artefacts (`ensemble_detector.joblib`, `optimal_threshold.txt`, `metadata.json`).
- `backend`, `ml-inference`, and `trainer` mount this volume to exchange models and thresholds seamlessly.

## Explainability (SHAP)

- TreeExplainer/KernalExplainer depending on the model
- Local explanations (feature contributions) returned with predictions
- Optional persistence in MongoDB collection `explanations` for audits

## Architecture overview

- **Phase I – Streaming backbone**: Kafka with TLS/mTLS, behavioural biometrics, device intelligence
- **Phase II – Feature engineering**: Spark Structured Streaming (rolling stats, cyclic encoding) feeding Redis/Feast
- **Phase III – Multi-layer detection**:
  - Layer 1 (supervised): RandomForest/LightGBM (fallback RandomForest) under 50 ms
  - Layer 2 (anomaly): Isolation Forest + autoencoder for rare signals
  - Layer 3 (deep): enriched models on user history with weighted aggregation
- **Phase IV – Decision & MLOps**: FastAPI + inference microservice, SHAP traceability, CI/CD (Jenkins, MLflow, DVC)

Detailed diagrams and notes live in `docs/architecture.md`.

## Testing with Postman

- Import the collection `docs/postman_collection.json` into Postman (covers `/transactions/ingest`, `/predictions/score`, `/admin/health`, and root endpoint).
- Set the collection-level variable `baseUrl` to `http://localhost:8000` (or the exposed FastAPI host).
- Start the stack with `docker compose -f docker-compose.dev.yml up --build`; wait until Kafka UI and RedisInsight show healthy status.
- Run the `Health` request first to ensure dependencies (Mongo, Redis Cluster, Kafka) report `ok`, then trigger `Transactions > Ingest transaction` followed by `Predictions > Score transaction` to validate the full flow.

