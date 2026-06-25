from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

PAGE_SIZE = 16
BLOCK_SIZE = 256


@dataclass
class Page:
    block_id: int
    seq_len: int = 0
    max_len: int = PAGE_SIZE
    kv_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def is_full(self) -> bool:
        return self.seq_len >= self.max_len

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self.kv_data is None:
            self.kv_data = (key, value)
        else:
            old_k, old_v = self.kv_data
            self.kv_data = (torch.cat([old_k, key], dim=-2), torch.cat([old_v, value], dim=-2))
        self.seq_len += key.size(-2)


@dataclass
class BlockTable:
    logical_blocks: List[int] = field(default_factory=list)
    seq_len: int = 0


class PagedKVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, max_blocks: int, device: torch.device):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks
        self.device = device

        self.free_blocks: List[int] = list(range(max_blocks))
        self.block_size = BLOCK_SIZE

        self.key_cache: List[torch.Tensor] = [
            torch.zeros(max_blocks, BLOCK_SIZE, num_heads, head_dim, dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]
        self.value_cache: List[torch.Tensor] = [
            torch.zeros(max_blocks, BLOCK_SIZE, num_heads, head_dim, dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]
        self.block_tables: Dict[str, BlockTable] = {}

    def allocate(self, request_id: str, num_tokens: int) -> BlockTable:
        needed = math.ceil(num_tokens / BLOCK_SIZE)
        if len(self.free_blocks) < needed:
            raise RuntimeError(f"OOM: {needed} blocks needed, {len(self.free_blocks)} free")

        blocks = self.free_blocks[:needed]
        self.free_blocks = self.free_blocks[needed:]

        table = BlockTable(logical_blocks=blocks, seq_len=num_tokens)
        self.block_tables[request_id] = table
        return table

    def write(self, request_id: str, layer: int, token_pos: int, key: torch.Tensor, value: torch.Tensor) -> None:
        table = self.block_tables.get(request_id)
        if table is None:
            raise ValueError(f"Unknown request: {request_id}")

        block_idx = token_pos // BLOCK_SIZE
        offset = token_pos % BLOCK_SIZE

        if block_idx >= len(table.logical_blocks):
            raise IndexError(f"Block index {block_idx} out of range ({len(table.logical_blocks)} blocks)")

        physical_block = table.logical_blocks[block_idx]

        k_dst = self.key_cache[layer][physical_block]
        v_dst = self.value_cache[layer][physical_block]

        k_src = key.squeeze(0) if key.dim() == 3 else key
        v_src = value.squeeze(0) if value.dim() == 3 else value
        token_len = k_src.size(-2)

        k_dst[offset : offset + token_len] = k_src
        v_dst[offset : offset + token_len] = v_src

    def read(self, request_id: str, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        table = self.block_tables.get(request_id)
        if table is None:
            raise ValueError(f"Unknown request: {request_id}")

        keys = []
        values = []
        for block in table.logical_blocks:
            keys.append(self.key_cache[layer][block][:PAGE_SIZE])
            values.append(self.value_cache[layer][block][:PAGE_SIZE])

        return torch.cat(keys, dim=-2), torch.cat(values, dim=-2)

    def free(self, request_id: str) -> None:
        table = self.block_tables.pop(request_id, None)
        if table is not None:
            self.free_blocks.extend(table.logical_blocks)

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        return self.block_tables.get(request_id)

    def usage(self) -> float:
        used = self.max_blocks - len(self.free_blocks)
        return used / self.max_blocks


class PagedAttention:
    @staticmethod
    def attend(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = query.size(-1) ** 0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn, value)

    @staticmethod
    def paged_attend(
        query: torch.Tensor,
        key_blocks: List[torch.Tensor],
        value_blocks: List[torch.Tensor],
        block_table: BlockTable,
    ) -> torch.Tensor:
        keys = torch.cat(key_blocks[: len(block_table.logical_blocks)], dim=-2)
        values = torch.cat(value_blocks[: len(block_table.logical_blocks)], dim=-2)
        return PagedAttention.attend(query, keys, values)
