# Monitoring

- Prometheus scrapes backend, inference, Kafka brokers (`monitoring/prometheus.yml`).
- Grafana dashboards track latency (p50/p95), throughput, precision/recall, drift alerts.
- Alerts trigger when latency > 120 ms, precision < target, or Kafka lag spikes.

