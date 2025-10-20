from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from prometheus_client import Counter, Gauge, Histogram

from .redis_client import RedisCache


R_BATCH_FLUSHES = Counter("redis_batch_flushes_total", "Total Redis batch flushes")
R_BATCH_ITEMS = Counter("redis_batch_items_total", "Total Redis ops in batches")
R_BATCH_DROPPED = Counter("redis_batch_dropped_total", "Redis ops dropped due to full queue")
R_BATCH_QUEUE_DEPTH = Gauge("redis_batch_queue_depth", "Redis batch queue depth")
R_BATCH_SIZE = Histogram("redis_batch_size", "Redis batch flush sizes", buckets=[10, 50, 100, 200, 500, 1000, 2000])
R_BATCH_FLUSH_LAT_MS = Histogram(
    "redis_batch_flush_latency_ms",
    "Latency of Redis batch flush (ms)",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500, 1000],
)
R_PIPELINE_LAT_MS = Histogram(
    "redis_pipeline_latency_ms",
    "Latency of Redis pipeline operations (ms)",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500, 1000],
)


class RedisBatchWriter:
    def __init__(
        self,
        cache: RedisCache,
        max_size: int = 500,
        flush_interval_ms: int = 100,
        queue_max: int = 5000,
        block_on_full: bool = True,
    ) -> None:
        self._cache = cache
        self._max = max(1, max_size)
        self._interval = max(20, flush_interval_ms) / 1000.0
        self._queue: asyncio.Queue[Tuple[str, Any, int]] = asyncio.Queue(maxsize=max(1, queue_max))
        self._block_on_full = block_on_full
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        buf: List[Tuple[str, Any, int]] = []
        while not self._stop.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._interval)
                buf.append(item)
                R_BATCH_QUEUE_DEPTH.set(self._queue.qsize())
                if len(buf) >= self._max:
                    await self._flush(buf)
                    buf.clear()
            except asyncio.TimeoutError:
                if buf:
                    await self._flush(buf)
                    buf.clear()
        if buf:
            await self._flush(buf)

    async def _flush(self, items: List[Tuple[str, Any, int]]) -> None:
        try:
            start = time.perf_counter()
            R_BATCH_SIZE.observe(len(items))
            await self._cache.pipeline_set_many(items)
            R_BATCH_FLUSHES.inc()
            R_BATCH_ITEMS.inc(len(items))
            dur_ms = (time.perf_counter() - start) * 1000.0
            R_BATCH_FLUSH_LAT_MS.observe(dur_ms)
            R_PIPELINE_LAT_MS.observe(dur_ms)
        except Exception:
            R_BATCH_DROPPED.inc(len(items))

    async def enqueue_set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        try:
            # Add jitter to TTL to avoid cache stampede
            jitter_pct = float(os.getenv("REDIS_TTL_JITTER_PCT", "0.1"))
            try:
                import random

                jitter = int(ttl_seconds * jitter_pct * random.random())
            except Exception:
                jitter = 0
            ttl = max(1, ttl_seconds + jitter)
            if self._block_on_full:
                await self._queue.put((key, value, ttl))
            else:
                self._queue.put_nowait((key, value, ttl))
            R_BATCH_QUEUE_DEPTH.set(self._queue.qsize())
            return True
        except asyncio.QueueFull:
            R_BATCH_DROPPED.inc()
            return False

    async def close(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                self._task.cancel()
