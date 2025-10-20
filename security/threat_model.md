# Threat Model (MITRE ATLAS alignment)

## Assets
- Données transactionnelles (Kafka topics)
- Modèles ML (RandomForest, IsolationForest, Autoencoder)
- Secrets (.env, certificats TLS)
- Journaux SHAP (MongoDB `explanations`)

## Adversaires & Techniques
- **Poisoning** : injection de transactions malveillantes -> mitigation via validation, détection d'anomalies.
- **Model Extraction** : appels massifs à l'API -> limitation de débit, surveillance, réponses différées.
- **Evasion** : modification progressive des features -> couche anomalie + autoencoder.
- **Credential Stuffing / ATO** : détection comportementale + MFA.

## Contrôles
- mTLS, rotation des certificats, secrets Vault.
- Monitoring: drift, latence, tentatives d'attaque (dashboard MITRE mapping).
- Tests adversariaux continus (adversarial training) intégrés dans `services/trainer`.
- Journaux immuables (SIEM) pour conformité bancaire.

