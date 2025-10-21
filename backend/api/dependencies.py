from functools import lru_cache

from .config import get_settings
from ..database.mongo import MongoDBClient
from ..database.mongo_batch import MongoBatchWriter
from ..database.redis_batch import RedisBatchWriter
from ..database.redis_client import RedisCache
from ..streaming.kafka_producer import TransactionProducer
from ..utils.model_service import AdvancedFraudModelService


@lru_cache
def get_mongo_client() -> MongoDBClient:
    settings = get_settings()
    return MongoDBClient(
        settings.mongo_uri,
        settings.mongo_db,
        tls_enabled=settings.mongo_ssl,
        tls_verify=True,
        tls_ca_path=settings.mongo_tls_ca_path,
    )


@lru_cache
def get_redis_cache() -> RedisCache:
    settings = get_settings()
    try:
        return RedisCache(
            startup_nodes=settings.redis_cluster_startup_nodes,
            password=settings.redis_password or None,
            use_ssl=settings.redis_use_ssl,
            ca_certs=settings.tls_ca_path if settings.redis_use_ssl else None,
        )
    except Exception:
        class _Dummy:
            async def health(self) -> bool:
                return False

            async def close(self) -> None:
                return None

        return _Dummy()  # type: ignore[return-value]


@lru_cache
def get_explanations_batch_writer() -> MongoBatchWriter:
    settings = get_settings()
    client = get_mongo_client()
    return MongoBatchWriter(
        client=client,
        collection="explanations",
        max_size=settings.batch_max_size,
        flush_interval_ms=settings.batch_flush_interval_ms,
        queue_max=settings.batch_queue_max,
        block_on_full=settings.batch_block_on_full,
    )


@lru_cache
def get_kafka_producer() -> TransactionProducer:
    settings = get_settings()
    return TransactionProducer(settings.kafka_broker_list)


@lru_cache
def get_redis_batch_writer() -> RedisBatchWriter:
    settings = get_settings()
    cache = get_redis_cache()
    # Slightly lower defaults for cache writes
    return RedisBatchWriter(
        cache=cache,
        max_size=min(500, settings.batch_max_size // 2),
        flush_interval_ms=max(50, settings.batch_flush_interval_ms // 2),
        queue_max=max(1000, settings.batch_queue_max // 2),
        block_on_full=not settings.batch_block_on_full,  # prefer drop-on-full to avoid hot path stalls
    )


@lru_cache
def get_model_service() -> AdvancedFraudModelService:
    settings = get_settings()
    return AdvancedFraudModelService(
        model_path=settings.model_path,
        performance_path=settings.model_performance_path,
    )
