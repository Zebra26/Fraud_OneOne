import logging
from typing import Any, Dict, Iterable, Optional

try:
    from redis.asyncio.cluster import RedisCluster
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Redis Cluster support requires redis>=5.0 with asyncio extras."
    ) from exc


logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(
        self,
        startup_nodes: Iterable[Dict[str, Any]],
        password: Optional[str] = None,
        use_ssl: bool = False,
        ca_certs: Optional[str] = None,
    ):
        nodes = list(startup_nodes)
        if not nodes:
            raise ValueError("Redis Cluster requires at least one startup node")

        self._client = RedisCluster(
            startup_nodes=nodes,
            password=password,
            decode_responses=True,
            socket_connect_timeout=2,
            ssl=use_ssl,
            ssl_cert_reqs=None if not use_ssl else "required",
            ssl_ca_certs=ca_certs,
        )

        logger.info(
            "Redis Cluster client initialisé",
            extra={
                "nodes": nodes,
                "ssl": use_ssl,
                "has_password": bool(password),
                "ca_provided": bool(ca_certs),
            },
        )

    async def set_feature_vector(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        await self._client.set(name=key, value=value, ex=ttl_seconds)

    async def get_feature_vector(self, key: str) -> Optional[str]:
        return await self._client.get(name=key)

    async def set_risk_score(self, transaction_id: str, score: float, ttl_seconds: int = 600) -> None:
        await self._client.set(name=f"risk:{transaction_id}", value=score, ex=ttl_seconds)

    async def pipeline_set_many(self, kv_ttl: Iterable[tuple[str, Any, int]]) -> None:
        """Set many keys in a single pipeline (key, value, ttl_seconds)."""
        pipe = self._client.pipeline(transaction=False)
        for key, value, ttl in kv_ttl:
            pipe.set(name=key, value=value, ex=ttl)
        await pipe.execute()

    async def health(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception as exc:  # pragma: no cover - dependent on infra setup
            logger.exception("Redis Cluster ping failed", exc_info=exc)
            return False

    async def close(self) -> None:
        await self._client.close()
