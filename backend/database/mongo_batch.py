from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram

from .mongo import MongoDBClient


BATCH_FLUSHES = Counter("mongo_batch_flushes_total", "Total number of batch flushes")
BATCH_ITEMS = Counter("mongo_batch_items_total", "Total number of items written in batches")
BATCH_DROPPED = Counter("mongo_batch_dropped_total", "Total number of items dropped due to full queue")
BATCH_QUEUE_DEPTH = Gauge("mongo_batch_queue_depth", "Current queue depth of batcher")
BATCH_SIZE = Histogram("mongo_batch_size", "Sizes of flushed batches", buckets=[10, 50, 100, 200, 500, 1000, 2000])


class MongoBatchWriter:
    def __init__(
        self,
        client: MongoDBClient,
        collection: str,
        max_size: int = 1000,
        flush_interval_ms: int = 200,
        queue_max: int = 10000,
        block_on_full: bool = True,
    ) -> None:
        self._client = client
        self._collection = collection
        self._max_size = max(1, max_size)
        self._interval = max(50, flush_interval_ms) / 1000.0
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max(1, queue_max))
        self._block_on_full = block_on_full
        self._stop = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        buf: List[Dict[str, Any]] = []
        last = time.perf_counter()
        while not self._stop.is_set():
            try:
                timeout = max(self._interval - (time.perf_counter() - last), 0.01)
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                buf.append(item)
                BATCH_QUEUE_DEPTH.set(self._queue.qsize())
                if len(buf) >= self._max_size:
                    await self._flush(buf)
                    buf.clear()
                    last = time.perf_counter()
            except asyncio.TimeoutError:
                if buf:
                    await self._flush(buf)
                    buf.clear()
                last = time.perf_counter()
        # final flush
        if buf:
            await self._flush(buf)

    async def _flush(self, items: List[Dict[str, Any]]) -> None:
        try:
            BATCH_SIZE.observe(len(items))
            await self._client.db[self._collection].insert_many(items, ordered=False)
            BATCH_FLUSHES.inc()
            BATCH_ITEMS.inc(len(items))
        except Exception:
            # On error we drop this batch to avoid blocking hot path
            BATCH_DROPPED.inc(len(items))

    async def enqueue(self, doc: Dict[str, Any]) -> bool:
        try:
            if self._block_on_full:
                await self._queue.put(doc)
            else:
                self._queue.put_nowait(doc)
            BATCH_QUEUE_DEPTH.set(self._queue.qsize())
            return True
        except asyncio.QueueFull:
            BATCH_DROPPED.inc()
            return False

    async def close(self) -> None:
        self._stop.set()
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=2.0)
            except Exception:
                self._worker_task.cancel()

