from __future__ import annotations

import contextlib
import os
from typing import AsyncIterator, Optional, Tuple

import torch

from specvlm.inference.kv_cache import PagedAttention, PagedKVCache

BLOCK_SIZE = 256
PAGE_SIZE = 16


class SpecInferenceEngine:
    def __init__(
        self,
        draft_model: torch.nn.Module,
        target_model: torch.nn.Module,
        visual_encoder: Optional[torch.nn.Module] = None,
        device: str = "cuda",
        max_blocks: int = 4096,
    ):
        self.draft = draft_model.to(device)
        self.target = target_model.to(device)
        self.visual_encoder = visual_encoder.to(device) if visual_encoder is not None else None
        self.device = torch.device(device)

        self.draft_kv_cache = PagedKVCache(
            num_layers=self._num_layers(self.draft),
            num_heads=self._num_heads(self.draft),
            head_dim=self._head_dim(self.draft),
            max_blocks=max_blocks,
            device=self.device,
        )
        self.target_kv_cache = PagedKVCache(
            num_layers=self._num_layers(self.target),
            num_heads=self._num_heads(self.target),
            head_dim=self._head_dim(self.target),
            max_blocks=max_blocks,
            device=self.device,
        )

        self.visual_embedding_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _num_layers(model: torch.nn.Module) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        return 32

    @staticmethod
    def _num_heads(model: torch.nn.Module) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_attention_heads"):
            return model.config.num_attention_heads
        return 32

    @staticmethod
    def _head_dim(model: torch.nn.Module) -> int:
        if hasattr(model, "config") and hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
            return model.config.hidden_size // model.config.num_attention_heads
        return 128

    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor, image_id: str) -> torch.Tensor:
        if image_id in self.visual_embedding_cache:
            return self.visual_embedding_cache[image_id]

        if self.visual_encoder is None:
            raise RuntimeError("No visual encoder available")

        embeds = self.visual_encoder(image.to(self.device))
        self.visual_embedding_cache[image_id] = embeds.cpu()
        return embeds

    @torch.inference_mode()
    def prefill(
        self,
        model: torch.nn.Module,
        kv_cache: PagedKVCache,
        input_ids: torch.Tensor,
        request_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_cache.allocate(request_id, input_ids.size(-1))

        logits = model(input_ids.to(self.device)).logits
        last_logit = logits[0, -1]

        probs = torch.softmax(last_logit, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token, last_logit

    @torch.inference_mode()
    def decode(
        self,
        model: torch.nn.Module,
        kv_cache: PagedKVCache,
        input_id: torch.Tensor,
        request_id: str,
    ) -> torch.Tensor:
        logits = model(input_id.to(self.device)).logits
        probs = torch.softmax(logits[0, -1], dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def speculative_decode(
        self,
        input_ids: torch.Tensor,
        request_id: str,
        spec_length: int = 5,
        max_tokens: int = 256,
    ) -> AsyncIterator[str]:
        draft_tokens = self.prefill(self.draft, self.draft_kv_cache, input_ids, f"{request_id}_draft")
        _ = self.prefill(self.target, self.target_kv_cache, input_ids, f"{request_id}_target")

        generated = 0
        while generated < max_tokens:
            proposals = [draft_tokens]
            for _ in range(spec_length - 1):
                next_tok = self.decode(self.draft, self.draft_kv_cache, proposals[-1], f"{request_id}_draft")
                proposals.append(next_tok)

            proposal_ids = torch.cat(proposals, dim=-1)
            target_logits = self.target(proposal_ids.to(self.device)).logits

            accepted = 0
            for i in range(len(proposals)):
                draft_prob = torch.softmax(self.draft(proposals[i].to(self.device)).logits[0, -1], dim=-1)
                target_prob = torch.softmax(target_logits[0, i], dim=-1)
                ratio = target_prob[proposals[i]] / (draft_prob[proposals[i]] + 1e-8)

                if torch.rand(1, device=self.device) < ratio:
                    accepted += 1
                else:
                    break

            for i in range(accepted):
                token_str = self.draft_tokenizer(proposals[i])
                generated += 1
                yield token_str

            if accepted < len(proposals):
                residual_logits = target_logits[0, accepted]
                residual_prob = torch.softmax(residual_logits, dim=-1)
                corrective = torch.multinomial(residual_prob, num_samples=1)
                token_str = self.draft_tokenizer(corrective)
                generated += 1
                yield token_str

    def draft_tokenizer(self, token_id: torch.Tensor) -> str:
        return f"<tok:{token_id.item()}>"

    def cache_usage(self) -> dict[str, float]:
        return {
            "draft_kv_cache": self.draft_kv_cache.usage(),
            "target_kv_cache": self.target_kv_cache.usage(),
        }

    @torch.inference_mode()
    def reset(self) -> None:
        self.draft_kv_cache = PagedKVCache(
            num_layers=self._num_layers(self.draft),
            num_heads=self._num_heads(self.draft),
            head_dim=self._head_dim(self.draft),
            max_blocks=self.draft_kv_cache.max_blocks,
            device=self.device,
        )
        self.target_kv_cache = PagedKVCache(
            num_layers=self._num_layers(self.target),
            num_heads=self._num_heads(self.target),
            head_dim=self._head_dim(self.target),
            max_blocks=self.target_kv_cache.max_blocks,
            device=self.device,
        )
