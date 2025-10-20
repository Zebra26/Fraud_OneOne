from __future__ import annotations

import asyncio
from functools import wraps
from typing import Callable, Iterable

from fastapi import HTTPException, Request


def require_roles(*required_roles: str) -> Callable:
    required = set(required_roles)

    def decorator(endpoint: Callable) -> Callable:
        if asyncio.iscoroutinefunction(endpoint):

            @wraps(endpoint)
            async def async_wrapper(*args, **kwargs):
                request = _extract_request(*args, **kwargs)
                _check_roles(request, required)
                return await endpoint(*args, **kwargs)

            return async_wrapper

        @wraps(endpoint)
        def sync_wrapper(*args, **kwargs):
            request = _extract_request(*args, **kwargs)
            _check_roles(request, required)
            return endpoint(*args, **kwargs)

        return sync_wrapper

    return decorator


def _extract_request(*args, **kwargs) -> Request:
    request = kwargs.get("request")
    if isinstance(request, Request):
        return request
    for arg in args:
        if isinstance(arg, Request):
            return arg
    raise RuntimeError("Request object not found for RBAC enforcement")


def _check_roles(request: Request, required: Iterable[str]) -> None:
    claims = getattr(request.state, "jwt_claims", {}) or {}
    user_roles = set(claims.get("roles", []))
    if not set(required).issubset(user_roles):
        raise HTTPException(status_code=403, detail="Insufficient role")
