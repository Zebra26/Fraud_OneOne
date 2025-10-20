# Security

- End-to-end mTLS between services (certs in `infrastructure/tls`).
- Secrets managed via Vault/KMS and injected as Kubernetes Secrets/ConfigMaps.
- SHAP audit trail persisted for regulatory evidence and operator review.
- MITRE ATLAS alignment documented in `security/threat_model.md`.

