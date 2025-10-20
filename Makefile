SHELL := /bin/bash

.PHONY: validate-observability obs-up obs-down

obs-up:
	docker compose -f docker-compose.dev.yml -f docker-compose.observability.yml -f docker-compose.observability.dev.yml up -d backend ml-inference prometheus grafana loki promtail

obs-down:
	docker compose -f docker-compose.dev.yml -f docker-compose.observability.yml -f docker-compose.observability.dev.yml down -v

validate-observability: obs-up
	BACKEND_URL=http://localhost:8000 \
	INFERENCE_URL=http://localhost:8080 \
	PROMETHEUS_URL=http://localhost:9090 \
	LOKI_URL=http://localhost:3100 \
	python scripts/validate_observability.py --output json --output-file observability_report.json; \
	RC=$$?; echo "Validation exit code: $$RC"; exit $$RC

.PHONY: perf-test
perf-test:
	@if ! command -v k6 >/dev/null 2>&1; then echo "k6 is not installed. See https://k6.io/docs/get-started/installation/"; exit 2; fi
	@if [ -z "$$JWT_SECRET_KEY" ] || [ -z "$$API_HMAC_KEY" ]; then echo "Set JWT_SECRET_KEY and API_HMAC_KEY in your environment"; exit 2; fi
	BACKEND_URL=$${BACKEND_URL:-http://127.0.0.1:8000} \
	JWT_SECRET_KEY=$${JWT_SECRET_KEY} API_HMAC_KEY=$${API_HMAC_KEY} \
	k6 run scripts/perf_score.js || true
	@if [ -f k6_summary.json ]; then \
	  if command -v jq >/dev/null 2>&1; then \
	    echo -n "k6 p95 latency (ms): "; jq -r '.p95_ms // .metrics.http_req_duration.percentiles["p(95)"] // empty' k6_summary.json; \
	  else \
	    python - << 'PY' \
import json,sys;\
data=json.load(open('k6_summary.json'));\
print('k6 p95 latency (ms):', data.get('p95_ms') or data.get('metrics',{}).get('http_req_duration',{}).get('percentiles',{}).get('p(95)'))\
PY\
	  ; fi; \
	else echo "k6_summary.json not found"; fi

.PHONY: perf-test-onnx
perf-test-onnx:
	@echo "Starting ml-inference with ONNX batching via docker-compose.onnx.yml overrides..."
	docker compose -f docker-compose.dev.yml -f docker-compose.onnx.yml up -d ml-inference
	@sleep 2
	$(MAKE) perf-test
	@echo "To revert to baseline, restart ml-inference without the override:"
	@echo "  docker compose -f docker-compose.dev.yml up -d ml-inference"

.PHONY: finalize-report
finalize-report:
	python scripts/finalize_report.py
