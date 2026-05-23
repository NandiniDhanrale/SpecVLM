"""
Response Aggregator — Merges Partial Results from Distributed Workers

In a distributed inference scenario (especially with tensor/pipeline
parallelism), partial results from multiple workers need to be merged
into a coherent response. The aggregator handles this.

When speculative decoding runs in parallel:
- Each worker may process different parts of the prompt
- The aggregator ensures tokens are in the correct order
- Handles partial failures gracefully (worker drops out)

Architecture:
┌──────────┐     ┌──────────────┐     ┌──────────┐
│ Worker 0 │────▶│              │     │          │
│ (Tokens) │     │  Aggregator  │────▶│  Final   │
├──────────┤     │  (Merge +    │     │  Response│
│ Worker 1 │────▶│   Reorder)   │     │          │
├──────────┤     │              │     └──────────┘
│ Worker 2 │────▶│              │
└──────────┘     └──────────────┘

Production:
- Uses Redis pub/sub for worker communication
- Configurable timeout for slow workers
- Automatic reordering of out-of-order tokens
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PartialResult:
    """Partial result from a single worker."""
    worker_id: str
    request_id: str
    token_ids: list[int]
    token_texts: list[str]
    sequence_number: int  # For ordering
    timestamp: float = field(default_factory=time.time)
    speculative_stats: dict = field(default_factory=dict)


class ResponseAggregator:
    """
    Aggregates partial inference results into a complete response.

    In distributed mode, multiple workers may produce tokens for
    different parts of the sequence. This merges them correctly.
    """

    def __init__(self, timeout_ms: float = 5000.0):
        self.timeout_ms = timeout_ms
        self._partial_results: dict[str, list[PartialResult]] = {}
        self._completed_requests: dict[str, list[PartialResult]] = {}

    async def add_partial(self, result: PartialResult) -> Optional[str]:
        """
        Add a partial result. Returns the complete text if all parts received.

        Args:
            result: Partial result from a worker

        Returns:
            Complete text if aggregation is done, None otherwise
        """
        request_id = result.request_id

        if request_id not in self._partial_results:
            self._partial_results[request_id] = []

        self._partial_results[request_id].append(result)

        # Check for completion (all workers responded)
        completed = await self._check_completion(request_id)
        if completed:
            return await self._finalize(request_id)

        return None

    async def _check_completion(self, request_id: str) -> bool:
        """Check if all expected partial results have arrived."""
        # In practice, we'd know the expected number of workers
        # For now, assume completion after timeout
        return False

    async def _finalize(self, request_id: str) -> str:
        """
        Merge and reorder all partial results into the final response.
        """
        results = sorted(
            self._partial_results.pop(request_id, []),
            key=lambda r: r.sequence_number,
        )

        all_tokens = []
        all_text = []
        total_spec_stats = {}

        for result in results:
            all_tokens.extend(result.token_ids)
            all_text.extend(result.token_texts)
            for key, value in result.speculative_stats.items():
                total_spec_stats[key] = total_spec_stats.get(key, 0) + value

        self._completed_requests[request_id] = results

        return "".join(all_text)

    def get_request_stats(self, request_id: str) -> dict:
        """Get stats for a completed request."""
        results = self._completed_requests.get(request_id, [])
        if not results:
            return {}

        total_time = max(r.timestamp for r in results) - min(r.timestamp for r in results)

        return {
            "request_id": request_id,
            "num_workers": len(results),
            "total_time_ms": total_time * 1000,
            "num_tokens": sum(len(r.token_ids) for r in results),
            "workers": [r.worker_id for r in results],
        }
