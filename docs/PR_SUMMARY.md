# Recap: Automation, Performance, and Documentation Updates

This pull request summarizes recent changes improving local dev UX, benchmarking, and performance.

## What’s Included

- Automation scripts (Windows/PowerShell)
  - `scripts/bootstrap.ps1`: venv creation and dependency install
  - `scripts/dev_up.ps1` / `scripts/dev_down.ps1`: start/stop services + health wait
  - `scripts/send_scores.ps1`: send synthetic traffic + decision summary
  - `scripts/stress_up.ps1`: scale workers, set perf flags, run standard stress
  - `scripts/stress_send_scores.py`: async stress sender with latency p50/p95/avg

- Backend performance
  - Shared `httpx.AsyncClient` for inference calls (keep-alive)
  - `PERF_MODE` flag: skip Mongo/Redis writes & Redis-heavy checks during benchmarks
  - `USE_LOCAL_INFERENCE` flag: optional in-backend scoring (no network hop)

- Inference service
  - ONNX acceleration path: try to load `/app/models/supervised.onnx`
  - If missing, attempt on-the-fly export from scikit-learn using `skl2onnx`
  - Uvicorn worker scaling via `UVICORN_WORKERS`

- Docker Compose adjustments
  - Backend Gunicorn multi-workers (`USE_GUNICORN`, `GUNICORN_WORKERS`)
  - Perf flags in environment, relaxed rate-limit for dev/stress
  - Service URLs clarified (inference not exposed to host by default)

- Windows compatibility & deps
  - Conditional `uvloop`, `uvicorn[standard]`, `onnxruntime` in `backend/requirements.txt`
  - Package init for `backend/security` imports

- Documentation
  - `README.md`: quickstart, automation scripts, stress usage, perf flags, ONNX notes
  - `README_ARCHITECTURE.md`: UIs, quickstart, perf flags, ONNX section
  - `DEVELOPMENT.md`: concise dev workflow

## How to Validate

- Bootstrap & start
  - `./scripts/bootstrap.ps1`
  - `./scripts/dev_up.ps1`
- Functional smoke
  - `python -m scripts.send_synthetic_scores --count 10 --sleep-ms 50`
- Stress benchmark
  - `./scripts/stress_up.ps1 -Total 5000 -Concurrency 400 -TimeoutSec 30`

## Notes

- PERF_MODE is intended for benchmarking; disable for functional tests.
- ONNX fast-path requires either `supervised.onnx` or a scikit-learn model for export.
- Inference remains a separate service to preserve the micro-services topology.
