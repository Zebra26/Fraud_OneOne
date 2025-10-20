import time
from typing import Callable

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


REQUEST_COUNT = Counter(
    "fraud_backend_requests_total",
    "Total number of requests received",
    ["method", "path", "status"],
)
REQUEST_ERRORS = Counter(
    "fraud_backend_request_errors_total",
    "Total number of error responses",
    ["method", "path"],
)
REQUEST_LATENCY = Histogram(
    "fraud_backend_request_latency_seconds",
    "Request latency distribution",
    ["method", "path"],
)


def _canonical_path(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return route.path
    return request.url.path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware recording latency, throughput, and error counters."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        method = request.method
        path = _canonical_path(request)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
            REQUEST_COUNT.labels(method=method, path=path, status="500").inc()
            REQUEST_ERRORS.labels(method=method, path=path).inc()
            raise

        status = str(response.status_code)
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        if response.status_code >= 500:
            REQUEST_ERRORS.labels(method=method, path=path).inc()
        return response
