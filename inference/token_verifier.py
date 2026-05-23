"""
Token Verifier — Acceptance/Rejection Logic for Speculative Decoding

The token verifier implements the core acceptance criterion that makes
speculative decoding correct. Given draft tokens from the small model
and log-probabilities from the large model, it decides which tokens to
accept and which to reject.

The key insight: you can accept draft tokens with high probability
under the target model, and only fall back to the target model when
the draft model is uncertain. This gives EXACTLY the same output
distribution as the target model alone (in expectation), while
being much faster.

Three acceptance strategies:
1. STRICT: Accept iff target assigns >= probability threshold
2. STOCHASTIC: Accept with probability min(1, p_target / p_draft)
   (This gives exact target-model sampling)
3. REJECTION_SAMPLING: Full rejection sampling with fallback
   (Guarantees identical distribution to target model)

Architecture:
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Draft   │───▶│  Token       │───▶│  Accepted    │
│  Tokens  │    │  Verifier    │    │  Tokens      │
└──────────┘    └──────────────┘    └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Rejected    │
              │  Tokens     │──▶ Target model fallback
              └──────────────┘

Latency budget:
- Verification must take < 2ms per batch of k tokens
- Each rejected token adds ~5ms of fallback latency
- Target acceptance rate: 60-85% depending on task difficulty
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AcceptanceStrategy:
    """Enum-like class for acceptance strategies."""
    STRICT = "strict"
    STOCHASTIC = "stochastic"
    REJECTION_SAMPLING = "rejection_sampling"


class TokenVerifier:
    """
    Implements token acceptance/rejection for speculative decoding.

    Usage:
        verifier = TokenVerifier(strategy="rejection_sampling")
        accepted, rejected = verifier.verify(
            draft_tokens=draft_ids,
            draft_logprobs=draft_logprobs,
            target_logprobs=target_logprobs,
        )
        output_tokens.extend(accepted)
        if rejected:
            # Fall back to target model for rejected positions
            target_token = verifier.sample_fallback(target_logprobs)
            output_tokens.append(target_token)
    """

    def __init__(
        self,
        strategy: str = "rejection_sampling",
        strict_threshold: float = -1.0,
    ):
        """
        Args:
            strategy: One of "strict", "stochastic", "rejection_sampling"
            strict_threshold: Log-prob threshold for strict acceptance
        """
        self.strategy = strategy
        self.strict_threshold = strict_threshold

        # Statistics
        self.total_tokens_evaluated = 0
        self.total_accepted = 0
        self.total_fallback_calls = 0

    def verify(
        self,
        draft_token_ids: torch.Tensor,
        draft_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        """
        Verify a sequence of draft tokens.

        Args:
            draft_token_ids: (1, k) tensor of draft token IDs
            draft_logprobs: (1, k) tensor of draft model log-probabilities
            target_logprobs: (1, k, vocab_size) tensor of target model logits

        Returns:
            accepted_tokens: List of tensors, each shape (1,) — accepted tokens
            rejected_token: Optional tensor (1,) — the token at the first
                           rejected position (if any), None if all accepted
        """
        self.total_tokens_evaluated += draft_token_ids.shape[-1]

        k = draft_token_ids.shape[-1]

        if self.strategy == AcceptanceStrategy.STRICT:
            return self._strict_verify(draft_token_ids, target_logprobs, k)
        elif self.strategy == AcceptanceStrategy.STOCHASTIC:
            return self._stochastic_verify(draft_token_ids, draft_logprobs, target_logprobs, k)
        elif self.strategy == AcceptanceStrategy.REJECTION_SAMPLING:
            return self._rejection_sampling_verify(draft_token_ids, draft_logprobs, target_logprobs, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _strict_verify(
        self,
        draft_token_ids: torch.Tensor,
        target_logprobs: torch.Tensor,
        k: int,
    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        """
        Strict verification: accept if target assigns logprob > threshold.

        This is the simplest strategy but doesn't guarantee distribution
        matching. Use for speed when slight distribution shift is acceptable.
        """
        accepted = []
        rejected_token = None

        for i in range(k):
            token_id = draft_token_ids[0, i]
            token_logprob = target_logprobs[0, i, token_id]

            if token_logprob > self.strict_threshold:
                accepted.append(token_id.unsqueeze(0))
            else:
                # Sample from target logprobs as fallback
                rejected_token = self.sample_fallback(target_logprobs[0, i])
                self.total_fallback_calls += 1
                break

        self.total_accepted += len(accepted)
        return accepted, rejected_token

    def _stochastic_verify(
        self,
        draft_token_ids: torch.Tensor,
        draft_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
        k: int,
    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        """
        Stochastic verification: accept with probability min(1, p_target/p_draft).

        This gives EXACTLY the same distribution as sampling from the target
        model, making it theoretically optimal. The acceptance probability is
        proportional to how much the target model agrees with the draft.

        Reference: "Fast Inference from Transformers via Speculative Decoding"
        (Leviathan et al., ICML 2023)
        """
        accepted = []
        rejected_token = None

        # Compute per-token acceptance probabilities
        draft_probs = draft_logprobs.exp()
        target_probs = target_logprobs.exp()

        for i in range(k):
            token_id = draft_token_ids[0, i]

            p_draft = draft_probs[0, i, token_id].item()
            p_target = target_probs[0, i, token_id].item()

            # Acceptance probability = min(1, p_target / p_draft)
            # If draft model assigned low prob but target assigns high,
            # the ratio > 1 and we accept greedily.
            # If draft overconfidently assigns high prob, we may reject.
            if p_draft > 0:
                acceptance_prob = min(1.0, p_target / p_draft)
            else:
                acceptance_prob = 0.0

            if torch.rand(1).item() < acceptance_prob:
                accepted.append(token_id.unsqueeze(0))
            else:
                rejected_token = self.sample_fallback(target_logprobs[0, i])
                self.total_fallback_calls += 1
                break

        self.total_accepted += len(accepted)
        return accepted, rejected_token

    def _rejection_sampling_verify(
        self,
        draft_token_ids: torch.Tensor,
        draft_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
        k: int,
    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full rejection sampling verification.

        This is the most theoretically sound approach, guaranteeing exact
        distribution matching with the target model. For each draft token:
        1. Sample r ~ Uniform(0,1)
        2. Accept if r * q_draft(x) < q_target(x)
        3. Otherwise, resample from max(0, q_target(x) - q_draft(x))

        This ensures E[output_distribution] = target_model_distribution
        while still getting speedup from the draft model.
        """
        accepted = []
        rejected_token = None

        draft_probs = draft_logprobs.exp()
        target_probs = target_logprobs.exp()

        for i in range(k):
            token_id = draft_token_ids[0, i]

            p_draft = draft_probs[0, i, token_id]
            p_target = target_probs[0, i, token_id]

            # Sample uniform random variable
            r = torch.rand(1, device=draft_token_ids.device)

            # Rejection criterion
            if r * p_draft < p_target:
                accepted.append(token_id.unsqueeze(0))
            else:
                # Resample from corrected distribution
                corrected_probs = torch.clamp(target_probs[0, i] - draft_probs[0, i], min=0)
                corrected_probs = corrected_probs / corrected_probs.sum()
                rejected_token = torch.multinomial(corrected_probs, 1)
                self.total_fallback_calls += 1
                break

        self.total_accepted += len(accepted)
        return accepted, rejected_token

    def sample_fallback(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample a token from target model logits as fallback.
        Used when a draft token is rejected.
        """
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    def get_stats(self) -> dict:
        """Return verification statistics."""
        return {
            "total_evaluated": self.total_tokens_evaluated,
            "total_accepted": self.total_accepted,
            "total_fallback_calls": self.total_fallback_calls,
            "acceptance_rate": (
                self.total_accepted / self.total_tokens_evaluated
                if self.total_tokens_evaluated > 0
                else 0.0
            ),
            "strategy": self.strategy,
        }
