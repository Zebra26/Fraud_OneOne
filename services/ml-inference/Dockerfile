FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential openssl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ensemble.py metrics.py .

RUN mkdir -p /app/models

ENV TLS_CERT_PATH=/app/security/certs/server.crt TLS_KEY_PATH=/app/security/certs/server.key TLS_CA_PATH=/app/security/certs/ca.crt

EXPOSE 8080 8444

CMD ["bash", "-lc", "if [ \"$ENABLE_TLS\" = \"true\" ]; then uvicorn app:app --host 0.0.0.0 --port 8444 --ssl-certfile $TLS_CERT_PATH --ssl-keyfile $TLS_KEY_PATH; else uvicorn app:app --host 0.0.0.0 --port 8080; fi"]
