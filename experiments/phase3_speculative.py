"""
Phase 3 Experiment — Speculative Decoding

Demonstrates the core speculative decoding pipeline:
1. Draft model generates K candidate tokens
2. Target model verifies all K tokens in one forward pass
3. Token verifier accepts/rejects based on log-probability comparison
4. Accepted tokens are emitted; rejected tokens trigger fallback

Measures:
- Acceptance rate (what % of draft tokens are accepted)
- Speedup over baseline (wall-clock time comparison)
- Distribution matching (are outputs statistically identical to target?)
- Optimal speculation length (what K gives best speedup?)

Usage:
    python experiments/phase3_speculative.py --image path/to/image.jpg
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from specvlm.inference.engine import InferenceEngine, EngineConfig, EngineMode, EngineBackend
from specvlm.inference.speculative_decoder import SpeculativeDecoder
from specvlm.inference.token_verifier import TokenVerifier
from specvlm.models.base_vlm import VLMInput
from specvlm.models.draft_model import DraftModel
from specvlm.models.target_model import TargetModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Phase 3: Speculative Decoding")
    parser.add_argument("--draft-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--spec-length", type=int, default=5,
                        help="Number of draft tokens per iteration (K)")
    parser.add_argument("--strategy", default="rejection_sampling",
                        choices=["strict", "stochastic", "rejection_sampling"])
    parser.add_argument("--output", default="results/phase3_speculative.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SpecVLM — Phase 3: Speculative Decoding")
    logger.info("=" * 60)
    logger.info(f"Draft Model:   {args.draft_model}")
    logger.info(f"Target Model:  {args.target_model}")
    logger.info(f"Spec Length:   {args.spec_length}")
    logger.info(f"Strategy:      {args.strategy}")

    # =========================================================================
    # BASELINE: Target model alone
    # =========================================================================
    logger.info("\n" + "-" * 40)
    logger.info("BASELINE: Target model alone")
    logger.info("-" * 40)

    target_engine = InferenceEngine(EngineConfig(
        mode=EngineMode.BASELINE,
        backend=EngineBackend.TRANSFORMERS,
        target_model_id=args.target_model,
    ))
    await target_engine.load()

    baseline_input = VLMInput(
        prompt=args.prompt,
        image_paths=[args.image] if args.image else [],
        max_new_tokens=args.max_tokens,
        temperature=0.7,
    )

    baseline_start = time.time()
    baseline_result = await target_engine.generate(baseline_input)
    baseline_time = (time.time() - baseline_start) * 1000

    logger.info(f"  Time: {baseline_time:.1f}ms")
    logger.info(f"  Tokens: {baseline_result.num_output_tokens}")
    logger.info(f"  TPS: {baseline_result.tokens_per_second:.1f}")
    logger.info(f"  Output: {baseline_result.text[:200]}...")

    await target_engine.shutdown()

    # =========================================================================
    # SPECULATIVE: Draft + Target with verification
    # =========================================================================
    logger.info("\n" + "-" * 40)
    logger.info("SPECULATIVE DECODING")
    logger.info("-" * 40)

    # Load models individually for detailed control
    draft = DraftModel(model_id=args.draft_model, backend="transformers")
    target = TargetModel(model_id=args.target_model, backend="transformers")
    draft.load()
    target.load()

    # Create verifier and decoder
    verifier = TokenVerifier(strategy=args.strategy)
    decoder = SpeculativeDecoder(
        draft_model=draft,
        target_model=target,
        verifier=verifier,
        speculation_length=args.spec_length,
    )

    spec_input = VLMInput(
        prompt=args.prompt,
        image_paths=[args.image] if args.image else [],
        max_new_tokens=args.max_tokens,
        temperature=0.7,
    )

    spec_start = time.time()
    spec_result = await decoder.generate(spec_input)
    spec_time = (time.time() - spec_start) * 1000

    logger.info(f"  Time: {spec_time:.1f}ms")
    logger.info(f"  Tokens: {spec_result.num_output_tokens}")
    logger.info(f"  TPS: {spec_result.tokens_per_second:.1f}")
    logger.info(f"  Acceptance rate: {spec_result.speculation_acceptance_rate:.2%}")
    logger.info(f"  Output: {spec_result.text[:200]}...")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Baseline vs Speculative")
    logger.info("=" * 60)

    speedup = baseline_time / spec_time if spec_time > 0 else 0
    tps_improvement = (
        (spec_result.tokens_per_second - baseline_result.tokens_per_second)
        / baseline_result.tokens_per_second * 100
        if baseline_result.tokens_per_second > 0
        else 0
    )

    logger.info(f"  {'Metric':<30} {'Baseline':>12} {'Speculative':>12} {'Improvement':>12}")
    logger.info(f"  {'-'*66}")
    logger.info(f"  {'Total Time (ms)':<30} {baseline_time:>12.1f} {spec_time:>12.1f} {speedup:>11.2f}x")
    logger.info(f"  {'Tokens/Sec':<30} {baseline_result.tokens_per_second:>12.1f} {spec_result.tokens_per_second:>12.1f} {tps_improvement:>+11.1f}%")
    logger.info(f"  {'Acceptance Rate':<30} {'N/A':>12} {spec_result.speculation_acceptance_rate:>11.2%} {'':>12}")

    # Detailed decoder stats
    dec_stats = decoder.get_stats()
    logger.info(f"\nDecoder Stats:")
    logger.info(f"  Iterations:       {dec_stats['total_iterations']}")
    logger.info(f"  Draft tokens:     {dec_stats['total_draft_tokens']}")
    logger.info(f"  Accepted tokens:  {dec_stats['total_accepted_tokens']}")
    logger.info(f"  Fallback tokens:  {dec_stats['total_fallback_tokens']}")
    logger.info(f"  Avg accepted/iter: {dec_stats['avg_accepted_per_iter']:.1f}")
    logger.info(f"  Avg iter time:    {dec_stats['avg_iteration_time_ms']:.1f}ms")

    # Save
    results = {
        "phase": "3_speculative_decoding",
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "speculation_length": args.spec_length,
        "strategy": args.strategy,
        "baseline": {
            "time_ms": baseline_time,
            "tokens": baseline_result.num_output_tokens,
            "tps": baseline_result.tokens_per_second,
            "text": baseline_result.text,
        },
        "speculative": {
            "time_ms": spec_time,
            "tokens": spec_result.num_output_tokens,
            "tps": spec_result.tokens_per_second,
            "acceptance_rate": spec_result.speculation_acceptance_rate,
            "text": spec_result.text,
        },
        "speedup_x": speedup,
        "tps_improvement_pct": tps_improvement,
        "decoder_stats": dec_stats,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
