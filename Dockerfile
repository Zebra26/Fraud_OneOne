# Root dev Dockerfile: optional base for multi-service dev
FROM python:3.11-slim as base

WORKDIR /workspace
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

CMD ["bash"]

