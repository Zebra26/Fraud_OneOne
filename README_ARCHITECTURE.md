# Fraud Detection Realtime – Architecture Guide

This document explains the system architecture, core components, operational modes, observability, and security posture for the realtime fraud detection platform.

The project targets low-latency transaction scoring with layered ML, strong security controls (JWT + HMAC + TLS/mTLS), and first‑class observability (Prometheus/Grafana/Loki/Alertmanager). It is Docker/Kubernetes ready and supports both development and production topologies.


## High‑Level Overview

- API Backend (FastAPI)
  - Transaction ingestion, scoring orchestration, admin endpoints, metrics, RBAC, JWT/HMAC verification, Redis‑backed rate limiting, correlation‑ID propagation, structured JSON logs.
- Inference Service (FastAPI)
  - Hosts inference pipeline (tabular + anomaly + deep components), emits decision metrics, handles SAFE MODE fallback, supports JWT/HMAC or mTLS‑only mode, structured JSON logs.
- Streaming
  - Kafka (KRaft) for event flow (transactions, predictions). Optional JMX/exporter for metrics.
- Data Stores
  - Redis Cluster for fast feature store and rate limiting tokens.
  - MongoDB for explanations/audit documents and optional feature persistence.
  - MLflow for experiment tracking and artifacts (local or remote).
- Observability
  - Prometheus (metrics), Alertmanager (routing), Grafana (dashboards), Loki (logs), Promtail (log shipper), node‑exporter & cAdvisor (system/container metrics).
- Security
  - JWT (HS256/RS256 ready) + HMAC request signing, TLS/mTLS, RBAC, timestamp skew checks, rate limiting, correlation‑ID, and tamper‑evidence primitives.


## Data Flow (Happy Path)

1. Client sends `/predictions/score` to Backend with headers:
   - `Authorization: Bearer <JWT>` (roles include at least `service`)
   - `X-Request-Timestamp` (epoch seconds, ±120s)
   - `X-Request-Signature` (base64 HMAC‑SHA256 over `method|path|timestamp|body`)
   - Optional `X-Correlation-ID` (otherwise generated)
2. Backend middleware validates JWT, HMAC, timestamp, applies Redis token‑bucket rate limiting, binds `correlation_id` to logs.
3. Backend forwards request to Inference, passing JWT and a new HMAC signature (same v2 scheme). The `correlation_id` is forwarded.
4. Inference executes hybrid model (tabular + anomaly + AE/LSTM) and returns a decision + breakdown + threshold. It emits `model_decisions_total{decision,channel,model_version}`.
5. Backend writes explanation to MongoDB, caches derived artifacts in Redis (feature vectors & risk scores), and returns the response. Both services emit metrics and structured JSON logs.


## Environments

- Development (Docker Compose)
  - Redis Cluster: `grokzen/redis-cluster` on ports `7000-7005`.
  - TLS relaxed by default, JWT/HMAC enabled for Inference; Backend can run without TLS (set in `docker-compose.dev.yml`).
  - Observability overlay publishes Prometheus, Grafana, Loki, etc. ports.
- Production (Docker Compose or Kubernetes)
  - Managed Redis/Valkey Cluster with TLS, managed Kafka, and production‑grade Mongo. Strict TLS/mTLS end‑to‑end. Secrets via Docker/K8s/Vault.
  - RBAC enforced for admin/safe‑mode/metrics; no public exposure of observability UIs (VPN/SSO required).


## Services & Key Files

- Backend (FastAPI)
  - `backend/api/main.py` – App setup, Prometheus middleware, Security middleware, `/metrics` (admin), correlation‑ID binding.
  - `backend/middleware/security.py` – JWT, HMAC v2 validation, timestamp drift check, Redis rate limiter, counters for auth failures and rate limiting.
  - `backend/api/routers/predictions.py` – Score endpoint, inference forwarding with HMAC v2, decision counter, Mongo write metrics, Redis ops counter.
  - `backend/security/hmac_utils.py` – HMAC v2 (`method|path|timestamp|body`, base64) + legacy compatibility.
  - `backend/logging_config.py` – Structured JSON logging, PII masking, request context (correlation_id, route, user_agent).
  - `backend/tests/test_auth_security.py` – Unit tests for JWT/HMAC/rate‑limit basics (dev context).

- Inference (FastAPI)
  - `services/ml-inference/app.py` – Auth guard (JWT+HMAC or mTLS‑only), `/infer`, SAFE MODE, decision metrics, `/metrics` (admin), JSON logging + correlation‑ID.
  - `services/ml-inference/logging_config.py` – Structured logs (reuses masking), request context binding.
  - `services/ml-inference/safe_mode.py` – SAFE MODE gauge + state.

- Trainer
  - `services/trainer/train_pipeline.py` – MLflow tracking, artifact packaging (encrypted/signed), metadata & performance files for linkage (`model_version`, thresholds, hashes).

- Observability
  - `observability/prometheus/prometheus.yml` – Scrapes backend, inference, exporters.
  - `observability/prometheus/alert_rules.yml` – SLO/security rules (latency, auth failures, SAFE MODE, drift, Redis/Kafka health).
  - `observability/alertmanager/alertmanager.yml` – Routing, grouping, inhibition; receivers via secrets.
  - `observability/grafana/provisioning/...` – Prometheus & Loki data sources, dashboards JSON (API performance, Security & Integrity, Drift, System Health, Inference detail).
  - `observability/promtail/promtail.yml` – Docker SD; ships container logs to Loki; labels: `job=fraud_k`, `service`, `correlation_id`, `route`.

- Compose
  - `docker-compose.dev.yml` – Dev stack, local images, Redis Cluster (grokzen), relaxed TLS.
  - `docker-compose.observability.yml` – Internal-only Prometheus/Alertmanager/Grafana/Loki/Promtail (+ node‑exporter, cAdvisor) network.
  - `docker-compose.observability.dev.yml` – Dev overlay publishing observability ports.
  - `docker-compose.prod.yml` – TLS/mTLS wiring in production; mounts certs; strict envs.

- Tooling
  - `scripts/validate_observability.py` – CI‑friendly validator for metrics reachability, SAFE MODE alert, auth-failure alert, latency alert (best-effort), and Loki logs by correlation_id.


## Security Model

- Authentication & Authorization
  - JWT (HS256 by default) with `iss`, `aud`, `exp` checks. Roles claim (`roles`: `admin`, `service`, `read`).
  - Admin routes use `require_roles("admin")` (metrics, SAFE MODE admin proxy).
- HMAC Request Signing (v2)
  - Header `X-Request-Signature` = base64(HMAC‑SHA256(API_HMAC_KEY, `method|path|timestamp|body`)).
  - `X-Request-Timestamp` validated with ±120s drift.
  - Backward‑compatible legacy hex signature accepted when enabled.
- Rate Limiting
  - Redis token bucket (capacity 100, refill 50/min default, configurable via env), keyed by JWT claims.
- TLS/mTLS
  - Dev: can run without TLS for iteration; Inference can require JWT+HMAC even without TLS.
  - Prod: TLS/mTLS enforced across services; Backend/Inference/Trainer mount certs (`./security/certs`).
- SAFE MODE
  - If integrity, signature, or critical env validation fails, Inference enters SAFE MODE (503 on protected ops). Gauge `safe_mode_state` exposed.
- Logging
  - Structured JSON with PII masking (PAN/IBAN/IP). Fields: `ts, level, service, correlation_id, route, decision, score, model_version, hash_verified, sig_verified, user_agent(masked)`.


## Metrics & Alerts

- Backend
  - `fraud_backend_requests_total{method,path,status}`
  - `fraud_backend_request_latency_seconds_{bucket,sum,count}`
  - `request_latency_ms_bucket` (middleware), `auth_failures_total{reason}`, `jwt_validations_total`, `jwt_failures_total`, `hmac_verifications_total`, `rate_limit_block_total`
  - `mongo_write_ms`, `redis_ops_total{operation}`
- Inference
  - `fraud_inference_requests_total{method,path,status}`
  - `fraud_inference_request_latency_seconds_{bucket,sum,count}`, `infer_request_latency_ms_bucket`
  - `model_decisions_total{decision,channel,model_version}`
  - `safe_mode_state{service}`, optional `drift_alerts_total`, `psi_*` (if drift implemented)
- Alerts (Prometheus)
  - SAFE MODE active (critical), auth failures (critical), p95 latency > 60ms (warning), model integrity/signature/decrypt failures (critical), drift alerts (warning), Kafka lag & Redis saturation (warning/critical).


## Dashboards

- API Performance – p50/p90/p95, RPS, error rate, per‑route latency
- Security & Integrity – auth/JWT/HMAC failures, model integrity signals, SAFE MODE
- Fraud Scores & Drift – average score/std, drift alerts, PSI if present
- System Health – Redis/Mongo/Kafka, node/cAdvisor CPU/mem/I/O
- Inference Detail – by sub‑component if exported (tabular/graph/AE/LSTM)


## Configuration & Secrets

- `.env` (dev), `.env.prod.example` (prod template). Secrets: `JWT_SECRET_KEY`, `API_HMAC_KEY`, TLS certs (mounted), Kafka/Mongo/Redis URIs with SSL settings.
- Store production secrets in Vault/K8s/Docker secrets. Enforce MFA & approval for rotation. Avoid committing private keys.


## Development

- Bring up core + observability
  - `docker compose -f docker-compose.dev.yml up -d`
  - `docker compose -f docker-compose.dev.yml -f docker-compose.observability.yml -f docker-compose.observability.dev.yml up -d prometheus alertmanager grafana loki promtail`
- UIs
  - Backend: `http://localhost:8000` (admin metrics require JWT/HMAC)
  - Inference: `http://ml-inference:8080` (internal; not published to host by default)
  - Kafka UI: `http://localhost:8080`
  - RedisInsight: `http://localhost:8001`
  - Prometheus: `http://localhost:9090`, Grafana: `http://localhost:3000`, Loki: `http://localhost:3100`
- Observability validator
  - Set `JWT_SECRET_KEY`, `API_HMAC_KEY`, and run `python scripts/validate_observability.py`


## Quickstart & Automation (Dev)

- Bootstrap local venv and deps (Windows/PowerShell):
  - `./scripts/bootstrap.ps1`
- Start core services and wait for health:
  - `./scripts/dev_up.ps1`
- Stop services (optionally remove volumes):
  - `./scripts/dev_down.ps1` (add `-CleanVolumes` to wipe volumes)


## Synthetic Traffic & Stress Testing

- Send a few synthetic requests (JWT + HMAC are generated for you):
  - `python -m scripts.send_synthetic_scores --count 10 --sleep-ms 50`
- Aggregate summary with PowerShell wrapper:
  - `./scripts/send_scores.ps1 -Count 100 -DelayMs 100`
- High-concurrency benchmark (with worker scaling and perf flags):
  - `./scripts/stress_up.ps1 -Total 5000 -Concurrency 400`
  - Parameters: `-BackendWorkers`, `-InferenceWorkers`, `-TimeoutSec`, `-NoBuild`
  - Prints summary: Total/OK/ERRORS/elapsed/RPS, plus latency p50/p95/avg.


## Performance Modes & Flags

- Backend fast path (no network hop):
  - `USE_LOCAL_INFERENCE=true` computes the score in backend using `AdvancedFraudModelService` (tabular model).
- Benchmark mode (avoid storage overhead):
  - `PERF_MODE=true` disables Mongo/Redis writes on the score path and relaxes security checks that require Redis (JTI, per-device limiter), keeping JWT+HMAC verification intact.
- Workers:
  - Backend: `USE_GUNICORN=true`, `GUNICORN_WORKERS`, `GUNICORN_TIMEOUT`.
  - Inference: Uvicorn workers via `UVICORN_WORKERS` (see docker-compose.dev.yml overrides).


## ONNX Acceleration (Inference)

- Enable runtime acceleration:
  - Set `USE_ONNXRUNTIME=true` in the `ml-inference` environment.
- Supervised model export:
  - If `/app/models/supervised.onnx` is missing, the service attempts an on-the-fly export from the loaded scikit-learn model using `skl2onnx`.
  - Alternatively, provide `supervised.onnx` in the mounted models volume (`models_data:/app/models`).
- Optional micro-batching:
  - `INFER_ONNX_BATCH=true`, `INFER_BATCH_SIZE` (default 16), `INFER_BATCH_TIMEOUT_MS` (default 5) for batched ONNX calls.


## Production Considerations

- Keep observability endpoints internal (no public ports). Protect Grafana/Prometheus via SSO or VPN.
- Use managed Redis/Valkey/Kafka/Mongo where possible; configure TLS/mTLS and CA bundles.
- Autoscaling: scale Backend/Inference horizontally; Kafka partitions for throughput; ensure model artifact distribution strategy (shared volume or object store).
- Backpressure & resilience: bound queues, timeouts, circuit‑breakers if needed; SAFE MODE provides controlled degradation.

## Horizontal Scalability & Auto‑scaling

- Target throughput: plan for several thousand requests per second.
- Backend and Inference are stateless; scale horizontally behind a load balancer or Kubernetes Service.
- Use Kubernetes HPA/KEDA based on Prometheus metrics (p95 latency, request rate), CPU, or custom SLOs.
- Increase Kafka partitions to match consumer parallelism; pin partition keys as needed for ordering.
- Tune Redis cluster slots and shards for throughput; avoid hot keys.

## Circuit Breakers & Timeouts

- The backend’s call to Inference includes:
  - Fixed client timeout (connect and overall) to avoid head‑of‑line blocking.
  - A lightweight circuit breaker that opens after a burst of failures, returning 503 immediately to avoid cascading failures.
- Adjust via env: `CB_MAX_FAILURES`, `CB_WINDOW_SEC`, `CB_OPEN_SEC`.


## Failure Modes & SAFE MODE

- Integrity/signature/env failures put Inference into SAFE MODE (503 with `{"status":"safe_mode"}` and reason). Gauge `safe_mode_state=1` triggers alert.
- Admin can toggle SAFE MODE via Backend proxy `/admin/safe-mode/*` (JWT role `admin` + HMAC required).


## Extensibility & Next Steps

- Drift detection exports (`psi_*`, `drift_alerts_total`) and automatic thresholding
- mTLS across Kafka client code (producer/consumer) and Redis/Mongo drivers in prod manifests
- Grafana data links from metrics panels to Loki logs (by `correlation_id`)
- CI integration to run validator and publish JSON summary artifacts

## Latency Optimization

- Measure in milliseconds
  - Inference exports `infer_request_latency_ms_bucket`; backend middleware exports `request_latency_ms_bucket`. Prefer ms buckets for fine granularity; keep seconds histograms for coarse trends.
- Profiling critical paths
  - Profile scoring and feature fetching (e.g., `SERVICE.infer`, Redis/Mongo interactions) to identify bottlenecks. Consider async concurrency tuning and connection pool sizes.
- Caching heavy features
  - Cache frequently used feature vectors in Redis to avoid repeated Mongo round‑trips. This project writes features into Redis per request; consider read‑through caching for hot keys.
- Warm model
  - Models load at service start (no cold start per request). Keep models hot by avoiding per‑request re‑init; reload only on new artifact deployment.
- Micro‑batching & back‑pressure
  - Mongo explanations use in‑memory micro‑batching (configurable size/interval/queue bound) to absorb bursts while keeping median latency low.
  - Redis writes support pipeline batching; critical paths enqueue and return quickly, with back‑pressure to protect latency.
  - Kafka producer uses linger/batch settings to coalesce small messages at peak throughput.

---

This architecture balances low latency, strong security, and deep observability. Use the dev compose for rapid iteration and the prod templates as a secure baseline for deployment.
