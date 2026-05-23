"""
Request Scheduler — Intelligent Batching and Queue Management

The scheduler is responsible for:
1. Collecting incoming requests into optimal batches
2. Prioritizing requests based on cache locality and deadlines
3. Dispatching batches to inference workers
4. Managing queue depth and backpressure

Scheduling policies:
- FCFS: First-come, first-served (fair but inefficient)
- PRIORITY: Higher priority requests jump the queue
- CACHE_AWARE: Group requests with shared prefixes together
  (maximizes KV cache hit rate, critical for VLM serving)

Architecture:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Request A   │     │              │     │              │
│  (image: x)  │────▶│  Scheduler   │────▶│  Engine      │
│  Request B   │     │  (Batch +    │     │  Worker      │
│  (image: x)  │────▶│   Schedule)  │     │              │
│  Request C   │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Request     │
                     │  Queue       │
                     │  (Redis)     │
                     └──────────────┘

Production considerations:
- Dynamic batching: wait up to N ms for more requests before dispatching
- Cache-aware scheduling: group requests with same image prefix together
- Starvation prevention: aging mechanism for low-priority requests
- Backpressure: reject when queue exceeds max_depth
- Request coalescing: identical requests share a single computation
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from specvlm.models.base_vlm import VLMInput

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    CACHE_AWARE = "cache_aware"


@dataclass
class ScheduledRequest:
    """A request waiting to be scheduled."""
    request_id: str
    inputs: VLMInput
    arrival_time: float = field(default_factory=time.time)
    priority: int = 0
    cache_key: str = ""  # For cache-aware scheduling: hash of image path
    timeout_ms: float = 30000.0  # Max time before request is dropped


class RequestScheduler:
    """
    Schedules inference requests for optimal throughput.

    Uses dynamic batching: waits for a configurable window to collect
    multiple requests before dispatching to the engine.

    In cache-aware mode, groups requests that share the same image
    to maximize KV cache prefix reuse.
    """

    def __init__(
        self,
        engine=None,
        policy: SchedulingPolicy = SchedulingPolicy.CACHE_AWARE,
        max_batch_size: int = 64,
        max_waiting_ms: float = 10.0,
        max_queue_depth: int = 1000,
    ):
        self.engine = engine
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_waiting_ms = max_waiting_ms
        self.max_queue_depth = max_queue_depth

        # Queues
        self._fcfs_queue: list[ScheduledRequest] = []
        self._priority_queues: dict[int, list[ScheduledRequest]] = defaultdict(list)
        self._cache_aware_buckets: dict[str, list[ScheduledRequest]] = defaultdict(list)

        # Control
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None

        # Stats
        self.total_scheduled = 0
        self.total_batches = 0
        self.total_dropped = 0

    async def submit(self, request: ScheduledRequest) -> bool:
        """Submit a request to the scheduler. Returns True if accepted."""
        if len(self._fcfs_queue) >= self.max_queue_depth:
            self.total_dropped += 1
            logger.warning(f"Queue full, dropping request {request.request_id}")
            return False

        self._fcfs_queue.append(request)
        self.total_scheduled += 1
        return True

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        logger.info(f"Scheduler started (policy={self.policy.value})")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduling_loop(self) -> None:
        """
        Main scheduling loop.

        Collects requests into batches and dispatches them to the engine.
        Uses a time-based window to balance latency vs throughput.
        """
        while self._running:
            batch = await self._collect_batch()

            if batch:
                self.total_batches += 1
                await self._dispatch_batch(batch)
            else:
                # No requests, sleep briefly
                await asyncio.sleep(0.001)

    async def _collect_batch(self) -> list[ScheduledRequest]:
        """
        Collect a batch of requests for processing.

        In cache-aware mode, groups requests by image hash first.
        In priority mode, drains highest priority queues first.
        """
        if not self._fcfs_queue:
            return []

        if self.policy == SchedulingPolicy.CACHE_AWARE:
            return await self._collect_cache_aware_batch()
        elif self.policy == SchedulingPolicy.PRIORITY:
            return await self._collect_priority_batch()
        else:
            return await self._collect_fcfs_batch()

    async def _collect_fcfs_batch(self) -> list[ScheduledRequest]:
        """Simple FCFS batching within the time window."""
        deadline = time.time() + self.max_waiting_ms / 1000

        batch = []
        while len(batch) < self.max_batch_size:
            if self._fcfs_queue:
                request = self._fcfs_queue.pop(0)
                batch.append(request)
            else:
                if time.time() >= deadline:
                    break
                await asyncio.sleep(0.001)

        return batch

    async def _collect_priority_batch(self) -> list[ScheduledRequest]:
        """Priority-based batching: drain highest priorities first."""
        # Move FCFS queue items into priority buckets
        while self._fcfs_queue:
            req = self._fcfs_queue.pop(0)
            self._priority_queues[req.priority].append(req)

        deadline = time.time() + self.max_waiting_ms / 1000
        batch = []

        # Drain from highest priority to lowest
        for priority in sorted(self._priority_queues.keys(), reverse=True):
            while self._priority_queues[priority] and len(batch) < self.max_batch_size:
                batch.append(self._priority_queues[priority].pop(0))

            if len(batch) >= self.max_batch_size:
                break

        return batch

    async def _collect_cache_aware_batch(self) -> list[ScheduledRequest]:
        """
        Cache-aware batching: group requests by cache key.

        This maximizes KV cache reuse by grouping requests that share
        the same image or prompt prefix. The first request prefills
        the cache, and subsequent requests in the same group get
        near-zero prefill latency.
        """
        while self._fcfs_queue:
            req = self._fcfs_queue.pop(0)

            # Compute cache key from image paths
            cache_key = "_".join(req.inputs.image_paths) if req.inputs.image_paths else ""
            req.cache_key = cache_key
            self._cache_aware_buckets[cache_key].append(req)

        deadline = time.time() + self.max_waiting_ms / 1000
        batch = []

        # Pick the largest bucket first (maximizes cache reuse)
        sorted_buckets = sorted(
            self._cache_aware_buckets.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        for cache_key, bucket in sorted_buckets:
            while bucket and len(batch) < self.max_batch_size:
                batch.append(bucket.pop(0))

            if not bucket:
                del self._cache_aware_buckets[cache_key]

            if len(batch) >= self.max_batch_size:
                break

        return batch

    async def _dispatch_batch(self, batch: list[ScheduledRequest]) -> None:
        """
        Dispatch a batch of requests to the inference engine.

        In a production system, this would fan out to multiple
        worker processes/GPUs using Ray or similar.
        """
        if self.engine is None:
            logger.warning("No engine available, dropping batch")
            return

        logger.debug(f"Dispatching batch of {len(batch)} requests")

        # For now, process sequentially (parallel processing comes in Phase 5)
        for request in batch:
            try:
                await self.engine.generate(request.inputs)
            except Exception as e:
                logger.error(f"Request {request.request_id} failed: {e}")

    def get_stats(self) -> dict:
        """Return scheduler statistics."""
        return {
            "policy": self.policy.value,
            "total_scheduled": self.total_scheduled,
            "total_batches": self.total_batches,
            "total_dropped": self.total_dropped,
            "fcfs_queue_depth": len(self._fcfs_queue),
            "cache_buckets": len(self._cache_aware_buckets),
            "max_queue_depth": self.max_queue_depth,
        }
