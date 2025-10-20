FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace/..:/workspace

CMD ["python", "train_pipeline.py"]

