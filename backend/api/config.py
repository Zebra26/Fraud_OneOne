from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseSettings, Field, root_validator


class Settings(BaseSettings):
    app_env: str = Field("development", env="APP_ENV")
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")

    enable_tls: bool = Field(False, env="ENABLE_TLS")
    tls_cert_path: str = Field("/app/tls/server.crt", env="TLS_CERT_PATH")
    tls_key_path: str = Field("/app/tls/server.key", env="TLS_KEY_PATH")
    tls_ca_path: str | None = Field(None, env="TLS_CA_PATH")
    enable_mtls: bool = Field(False, env="ENABLE_MTLS")

    mongo_uri: str = Field(..., env="MONGO_URI")
    mongo_db: str = Field("fraud_db", env="MONGO_DB")
    mongo_ssl: bool = Field(True, env="MONGO_SSL")
    mongo_tls_ca_path: str | None = Field(None, env="MONGO_TLS_CA_PATH")

    redis_cluster_nodes: str = Field("redis-cluster:6379", env="REDIS_CLUSTER_NODES")
    redis_password: str | None = Field(None, env="REDIS_PASSWORD")
    redis_use_ssl: bool = Field(True, env="REDIS_USE_SSL")
    feature_store_uri: str = Field("redis+cluster://redis-cluster:6379", env="FEATURE_STORE_URI")

    kafka_brokers: str = Field("kafka:9092", env="KAFKA_BROKERS")
    kafka_topic_transactions: str = Field("transactions", env="KAFKA_TOPIC_TRANSACTIONS")
    kafka_topic_predictions: str = Field("predictions", env="KAFKA_TOPIC_PREDICTIONS")
    kafka_topic_features: str = Field("transactions.features", env="KAFKA_TOPIC_FEATURES")

    model_directory: str = Field("/app/models", env="MODEL_DIR")
    model_path: str = Field("/app/models/fraud_detection_advanced_model.pkl", env="MODEL_PATH")
    model_performance_path: str = Field("/app/models/model_performance.json", env="MODEL_PERFORMANCE_PATH")

    threshold_review: float = Field(0.7, env="THRESHOLD_REVIEW")
    threshold_block: float = Field(0.9, env="THRESHOLD_BLOCK")
    optimal_threshold_path: str = Field("/app/models/optimal_threshold.txt", env="OPTIMAL_THRESHOLD_PATH")

    # Micro-batching / back-pressure (Mongo explanations)
    batch_max_size: int = Field(1000, env="BATCH_MAX_SIZE")
    batch_flush_interval_ms: int = Field(200, env="BATCH_FLUSH_INTERVAL_MS")
    batch_queue_max: int = Field(10000, env="BATCH_QUEUE_MAX")
    batch_block_on_full: bool = Field(True, env="BATCH_BLOCK_ON_FULL")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @root_validator
    def _security_checks(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("enable_tls"):
            for path_key in ("tls_cert_path", "tls_key_path"):
                candidate = values.get(path_key)
                if not candidate or not Path(candidate).exists():
                    raise ValueError(f"{path_key.upper()} must point to an existing file when ENABLE_TLS=true.")

        app_env = (values.get("app_env") or "").lower()

        # Enforce strict security in production; relax for local/dev usage.
        if app_env == "production":
            redis_password = values.get("redis_password")
            if not redis_password:
                raise ValueError("REDIS_PASSWORD must be set to secure Redis connections.")

            if not values.get("redis_use_ssl"):
                raise ValueError("REDIS_USE_SSL must be true to enforce TLS access to Redis.")

            mongo_uri = values.get("mongo_uri", "")
            if not values.get("mongo_ssl"):
                raise ValueError("MONGO_SSL must be true to enforce TLS access to MongoDB.")
            if "@" not in mongo_uri:
                raise ValueError("When MONGO_SSL=true, MONGO_URI must include credentials to avoid anonymous access.")

            ca_path = values.get("mongo_tls_ca_path")
            if ca_path and not Path(ca_path).exists():
                raise ValueError("MONGO_TLS_CA_PATH must point to an existing CA certificate file.")

        return values

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
