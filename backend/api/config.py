from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field, model_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field("development", validation_alias=AliasChoices("APP_ENV"))
    app_host: str = Field("0.0.0.0", validation_alias=AliasChoices("APP_HOST"))
    app_port: int = Field(8000, validation_alias=AliasChoices("APP_PORT"))

    enable_tls: bool = Field(False, validation_alias=AliasChoices("ENABLE_TLS"))
    tls_cert_path: str = Field("/app/tls/server.crt", validation_alias=AliasChoices("TLS_CERT_PATH"))
    tls_key_path: str = Field("/app/tls/server.key", validation_alias=AliasChoices("TLS_KEY_PATH"))
    tls_ca_path: str | None = Field(None, validation_alias=AliasChoices("TLS_CA_PATH"))
    enable_mtls: bool = Field(False, validation_alias=AliasChoices("ENABLE_MTLS"))

    mongo_uri: str = Field(..., validation_alias=AliasChoices("MONGO_URI"))
    mongo_db: str = Field("fraud_db", validation_alias=AliasChoices("MONGO_DB"))
    mongo_ssl: bool = Field(True, validation_alias=AliasChoices("MONGO_SSL"))
    mongo_tls_ca_path: str | None = Field(None, validation_alias=AliasChoices("MONGO_TLS_CA_PATH"))

    redis_cluster_nodes: str = Field("redis-cluster:6379", validation_alias=AliasChoices("REDIS_CLUSTER_NODES"))
    redis_password: str | None = Field(None, validation_alias=AliasChoices("REDIS_PASSWORD"))
    redis_use_ssl: bool = Field(True, validation_alias=AliasChoices("REDIS_USE_SSL"))
    feature_store_uri: str = Field("redis+cluster://redis-cluster:6379", validation_alias=AliasChoices("FEATURE_STORE_URI"))

    kafka_brokers: str = Field("kafka:9092", validation_alias=AliasChoices("KAFKA_BROKERS"))
    kafka_topic_transactions: str = Field("transactions", validation_alias=AliasChoices("KAFKA_TOPIC_TRANSACTIONS"))
    kafka_topic_predictions: str = Field("predictions", validation_alias=AliasChoices("KAFKA_TOPIC_PREDICTIONS"))
    kafka_topic_features: str = Field("transactions.features", validation_alias=AliasChoices("KAFKA_TOPIC_FEATURES"))

    model_directory: str = Field("/app/models", validation_alias=AliasChoices("MODEL_DIR"))
    model_path: str = Field("/app/models/fraud_detection_advanced_model.pkl", validation_alias=AliasChoices("MODEL_PATH"))
    model_performance_path: str = Field("/app/models/model_performance.json", validation_alias=AliasChoices("MODEL_PERFORMANCE_PATH"))

    threshold_review: float = Field(0.7, validation_alias=AliasChoices("THRESHOLD_REVIEW"))
    threshold_block: float = Field(0.9, validation_alias=AliasChoices("THRESHOLD_BLOCK"))
    optimal_threshold_path: str = Field("/app/models/optimal_threshold.txt", validation_alias=AliasChoices("OPTIMAL_THRESHOLD_PATH"))

    # Micro-batching / back-pressure (Mongo explanations)
    batch_max_size: int = Field(1000, validation_alias=AliasChoices("BATCH_MAX_SIZE"))
    batch_flush_interval_ms: int = Field(200, validation_alias=AliasChoices("BATCH_FLUSH_INTERVAL_MS"))
    batch_queue_max: int = Field(10000, validation_alias=AliasChoices("BATCH_QUEUE_MAX"))
    batch_block_on_full: bool = Field(True, validation_alias=AliasChoices("BATCH_BLOCK_ON_FULL"))

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def _security_checks(self) -> "Settings":
        app_env = (self.app_env or "").lower()

        # Enforce strict security in production; relax for local/dev usage.
        if app_env == "production":
            if self.enable_tls:
                for path_key in ("tls_cert_path", "tls_key_path"):
                    candidate = getattr(self, path_key)
                    if not candidate or not Path(candidate).exists():
                        raise ValueError(f"{path_key.upper()} must point to an existing file when ENABLE_TLS=true.")
            if not (self.redis_password or ""):
                raise ValueError("REDIS_PASSWORD must be set to secure Redis connections.")

            if not self.redis_use_ssl:
                raise ValueError("REDIS_USE_SSL must be true to enforce TLS access to Redis.")

            if not self.mongo_ssl:
                raise ValueError("MONGO_SSL must be true to enforce TLS access to MongoDB.")
            if "@" not in (self.mongo_uri or ""):
                raise ValueError("When MONGO_SSL=true, MONGO_URI must include credentials to avoid anonymous access.")

            ca_path = self.mongo_tls_ca_path
            if ca_path and not Path(ca_path).exists():
                raise ValueError("MONGO_TLS_CA_PATH must point to an existing CA certificate file.")

        return self

    @property
    def kafka_broker_list(self) -> List[str]:
        return [broker.strip() for broker in self.kafka_brokers.split(",") if broker.strip()]

    @property
    def redis_cluster_startup_nodes(self) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        for node in self.redis_cluster_nodes.split(","):
            host_port = node.strip()
            if not host_port:
                continue
            host, _, port_str = host_port.partition(":")
            port = int(port_str or 6379)
            nodes.append({"host": host, "port": port})
        if not nodes:
            raise ValueError("Aucune configuration de noeud Redis Cluster na ete fournie")
        return nodes


@lru_cache
def get_settings() -> Settings:
    return Settings()
