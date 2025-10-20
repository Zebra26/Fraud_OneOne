from __future__ import annotations

import threading
import time
from typing import Dict

from prometheus_client import Gauge

SAFE_MODE_GAUGE = Gauge("safe_mode_state", "Safe mode status", labelnames=["service"])


class SafeModeState:
    _lock = threading.RLock()
    _enabled: bool = False
    _reason: str = ""
    _since: float = 0.0

    @classmethod
    def enable(cls, reason: str) -> None:
        with cls._lock:
            cls._enabled = True
            cls._reason = reason
            cls._since = time.time()
            SAFE_MODE_GAUGE.labels(service="ml-inference").set(1)

    @classmethod
    def disable(cls) -> None:
        with cls._lock:
            cls._enabled = False
            cls._reason = ""
            cls._since = 0.0
            SAFE_MODE_GAUGE.labels(service="ml-inference").set(0)

    @classmethod
    def status(cls) -> Dict[str, object]:
        with cls._lock:
            return {
                "enabled": cls._enabled,
                "reason": cls._reason,
                "since": cls._since,
            }

    @classmethod
    def is_enabled(cls) -> bool:
        with cls._lock:
            return cls._enabled
