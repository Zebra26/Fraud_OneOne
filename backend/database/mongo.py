from __future__ import annotations
import logging
import os
from typing import Any, Dict

from motor.motor_asyncio import AsyncIOMotorClient


logger = logging.getLogger(__name__)


class MongoDBClient:
    def __init__(self, uri: str, db_name: str, tls_enabled: bool = True, tls_verify: bool = True, tls_ca_path: str | None = None):
        self._uri = uri
        self._db_name = db_name
        client_kwargs: dict = dict(serverSelectionTimeoutMS=2000)
        if tls_enabled:
            client_kwargs.update(tls=True, tlsAllowInvalidCertificates=not tls_verify)
            if tls_ca_path:
                try:
                    from pathlib import Path as _Path
                    text = _Path(tls_ca_path).read_text(encoding="utf-8", errors="ignore")
                    if "BEGIN CERTIFICATE" in text:
                        client_kwargs["tlsCAFile"] = tls_ca_path
                except Exception:
                    pass
            cert_pem = os.getenv("TLS_CERT_PATH")
            if os.getenv("ENABLE_MTLS", "false").strip().lower() in {"1", "true", "yes"} and cert_pem:
                # tlsCertificateKeyFile must contain both client cert and key (PEM)
                client_kwargs["tlsCertificateKeyFile"] = cert_pem
        self._client = AsyncIOMotorClient(self._uri, **client_kwargs)
        logger.info(
            "MongoDB client initialized",
            extra={"db": self._db_name, "tls": tls_enabled, "verify": tls_verify, "ca": bool(tls_ca_path)},
        )

        # Ensure indexes (best-effort)
        try:
            self._ensure_indexes()
        except Exception:  # pragma: no cover
            logger.warning("Failed to ensure MongoDB indexes", exc_info=True)

    @property
    def db(self):
        return self._client[self._db_name]

    async def insert_transaction(self, data: Dict[str, Any]) -> str:
        result = await self.db.transactions.insert_one(data)
        return str(result.inserted_id)

    async def insert_explanation(self, data: Dict[str, Any]) -> str:
        result = await self.db.explanations.insert_one(data)
        return str(result.inserted_id)

    async def health(self) -> bool:
        try:
            await self._client.admin.command("ping")
            return True
        except Exception as exc:  # pragma: no cover - health path
            logger.exception("MongoDB ping failed", exc_info=exc)
            return False

    def close(self) -> None:
        self._client.close()

    def _ensure_indexes(self) -> None:
        """Create TTL and compound indexes used by the app."""
        # TTL on explanations.recorded_at (180 days)
        try:
            self.db.explanations.create_index("recorded_at", expireAfterSeconds=15552000)  # 180*24*3600
        except Exception:
            pass
        # Compound indexes for query speed (tokens used rather than raw PII)
        try:
            self.db.explanations.create_index([("customer_token", 1), ("recorded_at", -1)])
        except Exception:
            pass
        try:
            self.db.explanations.create_index([("device_token", 1), ("recorded_at", -1)])
        except Exception:
            pass
        try:
            self.db.explanations.create_index([("is_fraud", 1), ("recorded_at", -1)])
        except Exception:
            pass
