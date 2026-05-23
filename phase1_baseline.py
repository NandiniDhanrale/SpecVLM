"""
Phase 1 Experiment — Baseline VLM Inference

This script demonstrates basic VLM inference without speculative decoding.
It establishes the baseline performance metrics (TTFT, TPS) that all
subsequent optimizations will be compared against.

What it does:
1. Loads a target VLM (Qwen2-VL-7B or LLaVA-7B)
2. Processes an image + text prompt
3. Generates a response
4. Measures TTFT, TPS, and memory usage
5. Saves benchmark data

Usage:
    python experiments/phase1_baseline.py --image path/to/image.jpg --prompt "Describe this image"
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from specvlm.inference.engine import InferenceEngine, EngineConfig, EngineMode, EngineBackend
from specvlm.models.base_vlm import VLMInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline VLM Inference")
    parser.add_argument("--target-model", default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Target model ID on HuggingFace")
    parser.add_argument("--image", default=None,
                        help="Path to input image")
    parser.add_argument("--prompt", default="Describe this image in detail.",
                        help="Text prompt for the model")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--backend", default="transformers",
                        choices=["vllm", "transformers"],
                        help="Inference backend")
    parser.add_argument("--output", default="results/phase1_baseline.json",
                        help="Output path for benchmark results")
    parser.add_argument("--profile", action="store_true",
                        help="Enable PyTorch profiler")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print system info
    logger.info("=" * 60)
    logger.info("SpecVLM — Phase 1: Baseline VLM Inference")
    logger.info("=" * 60)
    logger.info(f"Target Model:    {args.target_model}")
    logger.info(f"Image:           {args.image or 'None (text-only test)'}")
    logger.info(f"Prompt:          {args.prompt[:80]}...")
    logger.info(f"Max Tokens:      {args.max_tokens}")
    logger.info(f"Backend:         {args.backend}")
    logger.info(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU:             {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Configure engine
    engine_config = EngineConfig(
        mode=EngineMode.BASELINE,
        backend=EngineBackend.VLLM if args.backend == "vllm" else EngineBackend.TRANSFORMERS,
        target_model_id=args.target_model,
        enable_profiling=args.profile,
    )

    # Create and load engine
    logger.info("\nLoading model...")
    engine = InferenceEngine(config=engine_config)
    load_start = time.time()
    await engine.load()
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Warmup
    logger.info("\nWarming up...")
    warmup_input = VLMInput(
        prompt="What do you see?",
        image_paths=[args.image] if args.image else [],
        max_new_tokens=16,
        temperature=0.0,
    )
    await engine.generate(warmup_input)
    logger.info("Warmup complete")

    # Run inference
    logger.info("\nRunning inference...")
    vlm_input = VLMInput(
        prompt=args.prompt,
        image_paths=[args.image] if args.image else [],
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    start_time = time.time()
    result = await engine.generate(vlm_input)
    elapsed = time.time() - start_time

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Generated text ({result.num_output_tokens} tokens):")
    logger.info("-" * 40)
    logger.info(f"{result.text}")
    logger.info("-" * 40)
    logger.info(f"TTFT:                {result.ttft_ms:.1f} ms")
    logger.info(f"Total time:          {elapsed * 1000:.1f} ms")
    logger.info(f"Tokens per second:   {result.tokens_per_second:.1f}")
    logger.info(f"Output tokens:       {result.num_output_tokens}")

    if torch.cuda.is_available():
        logger.info(f"\nGPU Memory:")
        logger.info(f"  Allocated:   {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"  Reserved:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        logger.info(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Engine stats
    logger.info(f"\nEngine Stats:")
    stats = engine.get_stats()
    logger.info(f"  Total requests:  {stats['requests_total']}")

    # Save results
    results = {
        "phase": "1_baseline_inference",
        "model": args.target_model,
        "backend": args.backend,
        "prompt": args.prompt,
        "num_tokens": result.num_output_tokens,
        "ttft_ms": result.ttft_ms,
        "total_time_ms": elapsed * 1000,
        "tokens_per_second": result.tokens_per_second,
        "load_time_s": load_time,
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        "engine_stats": stats,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {args.output}")

    # Cleanup
    await engine.shutdown()
    logger.info("\nDone! Baseline established for future optimization comparison.")


if __name__ == "__main__":
    asyncio.run(main())
