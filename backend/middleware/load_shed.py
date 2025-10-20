from __future__ import annotations

import os
import time
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Gauge


def _cpu_percent() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=0.0))
    except Exception:
        return 0.0


class LoadShedMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.cpu_threshold = float(os.getenv("SHED_CPU_PCT", "85"))
        self.mongo_queue_max = int(os.getenv("SHED_MONGO_QUEUE_MAX", "8000"))
        self.redis_queue_max = int(os.getenv("SHED_REDIS_QUEUE_MAX", "4000"))
        self.cpu_gauge = Gauge("cpu_utilization_pct", "Process CPU utilization percent (best-effort)")
        self.dropped = Counter("dropped_requests_total", "Total requests dropped due to load shedding")

    async def dispatch(self, request: Request, call_next: Callable):
        # Shed only for scoring endpoints (hot path), allow health/metrics/admin
        if request.url.path.startswith("/predictions/score"):
            # CPU check
            cpu = _cpu_percent()
            try:
                self.cpu_gauge.set(cpu)
            except Exception:
                pass
            if cpu >= self.cpu_threshold:
                try:
                    self.dropped.inc()
                except Exception:
                    pass
                return JSONResponse({"status": "degraded_mode", "reason": "cpu_overload"}, status_code=503)
            # Queue backlog checks (best-effort)
            try:
                from ..api.dependencies import get_explanations_batch_writer, get_redis_batch_writer

                mongo_qsize = getattr(get_explanations_batch_writer(), "_queue", None)
                if hasattr(mongo_qsize, "qsize") and mongo_qsize.qsize() > self.mongo_queue_max:  # type: ignore[attr-defined]
                    try:
                        self.dropped.inc()
                    except Exception:
                        pass
                    return JSONResponse({"status": "degraded_mode", "reason": "mongo_backlog"}, status_code=503)
                redis_qsize = getattr(get_redis_batch_writer(), "_queue", None)
                if hasattr(redis_qsize, "qsize") and redis_qsize.qsize() > self.redis_queue_max:  # type: ignore[attr-defined]
                    try:
                        self.dropped.inc()
                    except Exception:
                        pass
                    return JSONResponse({"status": "degraded_mode", "reason": "redis_backlog"}, status_code=503)
            except Exception:
                pass
        return await call_next(request)
