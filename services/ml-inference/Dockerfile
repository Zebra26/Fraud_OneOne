FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential openssl \
    && rm -rf /var/lib/apt/lists/*

# Requirements are in the service directory when building from repo root
COPY services/ml-inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy service sources (all needed modules)
COPY services/ml-inference/app.py .
COPY services/ml-inference/ensemble.py .
COPY services/ml-inference/inference.py .
COPY services/ml-inference/metrics.py .
COPY services/ml-inference/logging_config.py .
COPY services/ml-inference/safe_mode.py .

# Include shared backend modules required by app imports
COPY backend/security ./backend/security
COPY backend/monitoring ./backend/monitoring
COPY backend/logging_config.py ./backend/logging_config.py

RUN mkdir -p /app/models

ENV TLS_CERT_PATH=/app/security/certs/server.crt TLS_KEY_PATH=/app/security/certs/server.key TLS_CA_PATH=/app/security/certs/ca.crt

EXPOSE 8080 8444

CMD ["bash", "-lc", "if [ \"$ENABLE_TLS\" = \"true\" ]; then uvicorn app:app --host 0.0.0.0 --port 8444 --ssl-certfile $TLS_CERT_PATH --ssl-keyfile $TLS_KEY_PATH; else uvicorn app:app --host 0.0.0.0 --port 8080; fi"]
