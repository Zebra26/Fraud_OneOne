FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend package into /app/backend so imports like 'backend.api' work
COPY backend/ ./backend

ENV TLS_CERT_PATH=/app/tls/server.crt TLS_KEY_PATH=/app/tls/server.key

EXPOSE 8000 8443

ENV USE_GUNICORN=false GUNICORN_WORKERS=8 GUNICORN_TIMEOUT=15

CMD ["bash", "-lc", "\
if [ \"$USE_GUNICORN\" = \"true\" ]; then \
  gunicorn backend.api.main:app -k uvicorn.workers.UvicornWorker --workers ${GUNICORN_WORKERS} --timeout ${GUNICORN_TIMEOUT} --bind 0.0.0.0:8000; \
else \
  if [ \"$ENABLE_TLS\" = \"true\" ]; then \
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8443 --ssl-certfile $TLS_CERT_PATH --ssl-keyfile $TLS_KEY_PATH; \
  else \
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000; \
  fi; \
fi"]
