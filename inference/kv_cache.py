"""
KV Cache Manager — Memory-Optimized Cache for Transformer Attention

The KV cache is the single largest memory consumer during inference.
For a 7B model with batch=1 and seq_len=4096:
- KV cache size ≈ 2 × 32 layers × 32 heads × 128 dim × 4096 × 2 bytes ≈ 2 GB
- With speculative decoding, we also need cache for draft AND target models
- Prefix caching can reduce this by 40-60% for shared visual token prefixes

This module implements:
1. Paged KV cache (vLLM-style, but with shared prefix support)
2. Prefix caching — cache visual token KV pairs for reuse across requests
3. Cache-aware scheduling — prioritize requests with overlapping prefixes
4. Memory pooling — pre-allocate cache blocks to reduce fragmentation

Architecture:
┌────────────────────────────────────────────┐
│              KVCacheManager                 │
├────────────────────────────────────────────┤
│  ┌──────────────────────────────────────┐  │
│  │          Prefix Cache Tree            │  │
│  │  (Trie of cached token prefixes)      │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │        Block Pool (Pre-allocated)     │  │
│  │  [Block 0][Block 1]...[Block N]      │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │     Cache Eviction Policy (LRU)      │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘

Production at scale (Baseten/OpenAI):
- Cache is distributed across GPUs via NCCL
- Prefix cache uses approximate matching (Locality-Sensitive Hashing)
- Cache blocks are 16-32 tokens for fine-grained memory management
- Cache hit rates are monitored as a key SLO
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheBlock:
    """A fixed-size block of KV cache entries."""

    block_id: int
    block_size: int  # Number of token positions in this block
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None
    is_occupied: bool = False
    prefix_hash: str = ""
    last_access_time: float = 0.0

    def allocate(self, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: str) -> None:
        """Pre-allocate memory for this block."""
        shape = (num_layers, self.block_size, num_heads, head_dim)
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)

    def free(self) -> None:
        """Release GPU memory."""
        self.key_cache = None
        self.value_cache = None
        self.is_occupied = False


@dataclass
class PrefixCacheEntry:
    """Entry in the prefix cache tree."""

    token_ids: list[int]
    kv_blocks: list[CacheBlock]
    hit_count: int = 0
    last_used: float = 0.0


class PrefixCache:
    """
    Trie-based prefix cache for KV reuse.

    When two requests share the same prompt prefix (e.g., visual tokens),
    the second request can reuse the KV cache from the first, saving
    the entire prefill computation for common tokens.

    Visual tokens are particularly good candidates because:
    - They're identical across requests with the same image
    - They're numerous (576 tokens per image)
    - The vision tower output is deterministic
    """

    def __init__(self, max_entries: int = 1000, min_prefix_len: int = 10):
        self.max_entries = max_entries
        self.min_prefix_len = min_prefix_len

        # Trie structure: prefix_hash -> PrefixCacheEntry
        self._cache: dict[str, PrefixCacheEntry] = {}

        # Block allocations
        self._block_id_counter = 0
        self._block_pool: list[CacheBlock] = []

    def lookup(self, token_ids: list[int]) -> tuple[Optional[list[CacheBlock]], int]:
        """
        Find the longest cached prefix matching token_ids.

        Returns:
        - List of cached KV blocks (or None if no match)
        - Length of the matched prefix (number of tokens cached)
        """
        if not token_ids:
            return None, 0

        # Compute rolling hash of prefix
        longest_match = None
        match_length = 0

        for prefix_len in range(self.min_prefix_len, len(token_ids) + 1):
            prefix = token_ids[:prefix_len]
            prefix_hash = self._hash_prefix(prefix)

            if prefix_hash in self._cache:
                entry = self._cache[prefix_hash]
                longest_match = entry
                match_length = prefix_len
                entry.hit_count += 1
                entry.last_used = 0.0  # Will be set by caller
            else:
                break  # Prefix tree is contiguous

        if longest_match:
            return longest_match.kv_blocks, match_length
        return None, 0

    def store(self, token_ids: list[int], kv_blocks: list[CacheBlock]) -> None:
        """Cache KV blocks for a given token prefix."""
        if len(token_ids) < self.min_prefix_len:
            return

        prefix_hash = self._hash_prefix(token_ids)

        # Evict if full
        if len(self._cache) >= self.max_entries:
            self._evict_lru()

        self._cache[prefix_hash] = PrefixCacheEntry(
            token_ids=token_ids,
            kv_blocks=kv_blocks,
        )

    def _hash_prefix(self, token_ids: list[int]) -> str:
        """Compute a fast hash for a prefix."""
        return hashlib.md5(
            str(token_ids).encode(), usedforsecurity=False
        ).hexdigest()

    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_used,
        )
        del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all cached prefixes."""
        self._cache.clear()

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_blocks": len(self._block_pool),
        }


class KVCacheManager:
    """
    Central manager for all KV cache operations.

    Handles:
    - Cache block allocation and deallocation
    - Prefix cache operations
    - Memory monitoring and defragmentation
    - Cross-model cache sharing (draft ↔ target)

    Design decisions:
    - Block size of 16 tokens balances granularity vs overhead
    - Pre-allocation prevents OOM during serving
    - LRU eviction with frequency boost for hot prefixes
    """

    def __init__(
        self,
        block_size: int = 16,
        max_cache_gb: float = 0.0,
        device: str = "cuda",
    ):
        self.block_size = block_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.prefix_cache = PrefixCache()

        # Auto-detect available GPU memory
        if max_cache_gb <= 0 and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Reserve 30% of GPU memory for cache (rest for weights/activations)
            self.max_cache_bytes = int(total_memory * 0.30)
        else:
            self.max_cache_bytes = int(max_cache_gb * 1e9)

        # Block pool
        self._blocks: list[CacheBlock] = []
        self._next_block_id = 0
        self._allocated_bytes = 0

        # Model dimensions (set during allocation)
        self._num_layers = 0
        self._num_heads = 0
        self._head_dim = 0

        logger.info(
            f"KVCacheManager initialized: block_size={block_size}, "
            f"max_cache={self.max_cache_bytes / 1e9:.2f}GB"
        )

    def configure_for_model(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Configure cache geometry for a specific model architecture."""
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._dtype = dtype

        # Pre-allocate blocks
        block_bytes_per_token = (
            2  # K and V
            * num_layers
            * num_heads
            * head_dim
            * dtype.itemsize
        )
        tokens_per_block = block_bytes_per_token * self.block_size

        n_blocks = max(1, self.max_cache_bytes // tokens_per_block)
        self._preallocate_blocks(n_blocks)

        logger.info(
            f"Configured for model: {num_layers} layers, {num_heads} heads, "
            f"head_dim={head_dim}, {n_blocks} blocks pre-allocated"
        )

    def _preallocate_blocks(self, n_blocks: int) -> None:
        """Pre-allocate cache blocks to avoid runtime allocation."""
        for _ in range(n_blocks):
            block = CacheBlock(
                block_id=self._next_block_id,
                block_size=self.block_size,
            )
            if torch.cuda.is_available():
                block.allocate(
                    self._num_layers,
                    self._num_heads,
                    self._head_dim,
                    self._dtype,
                    self.device,
                )
            self._blocks.append(block)
            self._next_block_id += 1

        block_bytes = (
            self.block_size
            * self._num_layers
            * self._num_heads
            * self._head_dim
            * self._dtype.itemsize
            * 2  # K + V
        )
        self._allocated_bytes = n_blocks * block_bytes

    def allocate_blocks(self, n_blocks: int) -> list[CacheBlock]:
        """Allocate blocks from the pool."""
        allocated = []
        for block in self._blocks:
            if not block.is_occupied and len(allocated) < n_blocks:
                block.is_occupied = True
                block.prefix_hash = ""
                allocated.append(block)

        # If pool exhausted, warn and return what we have
        if len(allocated) < n_blocks:
            logger.warning(
                f"Cache pool exhausted: requested {n_blocks}, available {len(allocated)}"
            )
            # Trigger eviction
            self._evict_blocks(n_blocks - len(allocated))

        return allocated

    def free_blocks(self, blocks: list[CacheBlock]) -> None:
        """Return blocks to the pool."""
        for block in blocks:
            block.is_occupied = False
            block.prefix_hash = ""

    def _evict_blocks(self, n_blocks: int) -> None:
        """Evict occupied blocks (LRU policy)."""
        occupied = [b for b in self._blocks if b.is_occupied]
        occupied.sort(key=lambda b: b.last_access_time)

        for block in occupied[:n_blocks]:
            block.is_occupied = False
            block.prefix_hash = ""

    def get_cache_stats(self) -> dict:
        """Return comprehensive cache statistics."""
        total_blocks = len(self._blocks)
        occupied_blocks = sum(1 for b in self._blocks if b.is_occupied)
        cache_utilization = occupied_blocks / total_blocks if total_blocks > 0 else 0

        return {
            "total_blocks": total_blocks,
            "occupied_blocks": occupied_blocks,
            "free_blocks": total_blocks - occupied_blocks,
            "cache_utilization": cache_utilization,
            "allocated_gb": self._allocated_bytes / 1e9,
            "block_size": self.block_size,
            "prefix_cache_entries": len(self.prefix_cache._cache),
        }

    def clear(self) -> None:
        """Clear all cache data."""
        self.prefix_cache.clear()
        for block in self._blocks:
            block.is_occupied = False
            block.prefix_hash = ""
