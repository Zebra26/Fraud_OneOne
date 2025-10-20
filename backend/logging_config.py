import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict


correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
route_var: ContextVar[str | None] = ContextVar("route", default=None)
user_agent_var: ContextVar[str | None] = ContextVar("user_agent", default=None)


_PAN_RE = re.compile(r"\b\d{12,19}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def mask_pii(text: str) -> str:
    if not text:
        return text
    masked = _PAN_RE.sub("<PAN>", text)
    masked = _IBAN_RE.sub("<IBAN>", masked)
    masked = _IP_RE.sub("<IP>", masked)
    return masked


def set_log_context(*, correlation_id: str | None = None, route: str | None = None, user_agent: str | None = None) -> None:
    if correlation_id is not None:
        correlation_id_var.set(correlation_id)
    if route is not None:
        route_var.set(route)
    if user_agent is not None:
        user_agent_var.set(user_agent)


class ContextFilter(logging.Filter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        # Attach contextual fields
        record.service = self.service
        record.correlation_id = correlation_id_var.get()
        record.route = route_var.get()
        record.user_agent = user_agent_var.get()
        # Mask message if it is a string
        if isinstance(record.msg, str):
            record.msg = mask_pii(record.msg)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "service": getattr(record, "service", None),
            "correlation_id": getattr(record, "correlation_id", None),
            "route": getattr(record, "route", None),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include known extras when present
        for key in (
            "decision",
            "score",
            "model_version",
            "hash_verified",
            "sig_verified",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        # Include user_agent masked
        ua = getattr(record, "user_agent", None)
        if ua:
            payload["user_agent"] = mask_pii(str(ua))
        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging(service: str = "backend", level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers once
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter(service))
    root.addHandler(handler)

