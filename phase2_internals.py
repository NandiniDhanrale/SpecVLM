"""
Phase 2 Experiment — VLM Internals Deep Dive

This script demonstrates and explains the internal architecture of VLMs:
1. Vision tower (ViT) — how images become tokens
2. Projection layer — cross-modal alignment
3. Attention mechanism — how tokens interact
4. KV cache — prefill vs decode
5. Paged attention — vLLM's memory management

Outputs detailed tensor shape traces for educational purposes.

Usage:
    python experiments/phase2_internals.py --image path/to/image.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def explain_vision_encoder():
    """Explain how images are converted to visual tokens."""
    print("\n" + "=" * 70)
    print("VISION ENCODER: Image → Visual Tokens")
    print("=" * 70)
    print("""
    Architecture:
    ┌──────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────┐
    │  Image   │───▶│  Patch       │───▶│  ViT           │───▶│  Visual     │
    │  (336x336)│   │  Embedding   │   │  Transformer   │   │  Tokens     │
    └──────────┘    └──────────────┘    └────────────────┘    └─────────────┘
                      (14x14 patches)    (24 ViT layers)      (576 tokens)

    Why 576 tokens?
    - 336x336 image → 14x14 = 196 patches at 24x24 patch size
    - Some models use 24x24 = 576 patches with 14x14 patch size
    - Visual tokens are FIXED per image (unlike text tokens)
    
    Bottlenecks:
    1. Patch embedding: memory-bound (large image matrix multiply)
    2. ViT self-attention: O(n²) with n=576 patches
    3. Projection: learnable linear layer (no compute bottleneck)
    """)


def explain_projection():
    """Explain cross-modal projection layer."""
    print("\n" + "=" * 70)
    print("PROJECTION LAYER: Visual → LLM Embedding Space")
    print("=" * 70)
    print("""
    The projection layer maps vision tower outputs into the LLM's
    embedding space so the language model can attend to visual tokens.

    Architecture:
    Visual Embedding (ViT hidden_dim=1024)
           │
           ▼
    ┌──────────────────┐
    │  Linear(1024, 4096) │  ← Projection (learned)
    └──────────────────┘
           │
           ▼
    LLM Embedding Space (hidden_dim=4096)

    Key insight: The projection is a SIMPLE linear layer because
    cross-modal alignment is learned during pretraining. No need
    for complex architectures here.

    Tensor shapes:
    - ViT output:     (1, 576, 1024)
    - Projected:      (1, 576, 4096)
    - Text embeddings: (1, 20, 4096)  ← prompt tokens
    - Concatenated:    (1, 596, 4096) ← full input to LLM
    """)


def explain_attention():
    """Explain the attention mechanism in transformer blocks."""
    print("\n" + "=" * 70)
    print("ATTENTION MECHANISM: How Tokens Communicate")
    print("=" * 70)
    print("""
    Scaled Dot-Product Attention:
    
        Attention(Q, K, V) = softmax(Q × K^T / √d) × V
    
    For VLM inference, attention enables:
    1. Visual tokens → attend to each other (self-attention)
    2. Text tokens → attend to visual tokens (cross-attention)
    3. Text tokens → attend to previous text tokens (causal attention)

                    ┌──────┐  ┌──────┐  ┌──────┐
                    │  Q   │  │  K   │  │  V   │
                    └──┬───┘  └──┬───┘  └──┬───┘
                       │         │         │
                       ▼         ▼         │
                 ┌──────────────────┐      │
                 │  Q × K^T / √d   │      │
                 │  (Attention     │      │
                 │   Scores)       │      │
                 └───────┬─────────┘      │
                         │                │
                         ▼                │
                 ┌──────────────────┐      │
                 │    softmax( )    │      │
                 └───────┬─────────┘      │
                         │                │
                         ▼                ▼
                 ┌──────────────────────────┐
                 │    softmax × V           │
                 │    (Weighted sum)        │
                 └──────────────────────────┘

    KV cache optimization:
    - K and V are cached after prefill (Phase 2 detail)
    - Only Q is computed during decode
    - PagedAttention manages K,V as fixed-size blocks
    """)


def explain_kv_cache():
    """Explain the KV cache mechanism."""
    print("\n" + "=" * 70)
    print("KV CACHE: Memory for Recurrent Computation")
    print("=" * 70)
    print("""
    The KV cache stores key (K) and value (V) tensors from previous
    attention computations, avoiding recomputation at each decode step.

    PREPHASE (prefill): Compute K,V for ALL input tokens
    ┌─────────────────────────────────────────────┐
    │ [vis_0][vis_1]...[vis_575][t_0][t_1]...[t_N] │
    │ Compute K,V for all positions → store in cache│
    └─────────────────────────────────────────────┘
    Cost: 1 forward pass (expensive, but done once)

    DECODE PHASE: Generate tokens one at a time
    Step 1: [vis_0..t_N] → generate t_{N+1}
            K,V cache already has [vis_0..t_N]
            Only compute K,V for t_{N+1}
            Append to cache
    ...
    Step M: Only compute K,V for one new token
    Cost: 1 forward pass per token (cheap)

    Memory impact (7B model, 32 layers):
    Per token KV cache = 2 × 32 × 32 × 128 × 2 bytes = 524 KB
    For 4096 tokens: 524 KB × 4096 = 2.15 GB per sequence

    vLLM optimization: PagedAttention manages KV cache in blocks
    Block size: 16 tokens → 16 × 524 KB = 8.4 MB per block
    """)


def explain_prefill_vs_decode():
    """Contrast prefill and decode phases."""
    print("\n" + "=" * 70)
    print("PREPHASE vs DECODE: Two Distinct Compute Regimes")
    print("=" * 70)
    print("""
    ┌─────────────────────┬─────────────────────────────┐
    │    PREPHASE          │         DECODE              │
    ├─────────────────────┼─────────────────────────────┤
    │ Processes ALL tokens │ Processes 1 token at a time │
    │ at once              │                             │
    ├─────────────────────┼─────────────────────────────┤
    │ Compute-bound:       │ Memory-bound:               │
    │ matrix multiply     │ KV cache load dominates    │
    ├─────────────────────┼─────────────────────────────┤
    │ Uses FlashAttention │ Uses CUDA kernels           │
    │ for efficient       │ optimized for single        │
    │ long-sequence attn  │ token decoding              │
    ├─────────────────────┼─────────────────────────────┤
    │ Duration: 50-500ms  │ Duration: 5-50ms per token  │
    │ (depends on prompt) │ (linear in batch size)      │
    ├─────────────────────┼─────────────────────────────┤
    │ GPU Utilization:    │ GPU Utilization:            │
    │ ~70-90%             │ ~20-40% (underutilized)     │
    └─────────────────────┴─────────────────────────────┘

    Why speculative decoding works:
    The decode phase is MEMORY-BOUND (limited by GPU memory bandwidth,
    not compute). The target model's verification step processes K tokens
    in ONE forward pass, making it much more compute-efficient than K
    individual decode steps.
    """)


def explain_paged_attention():
    """Explain vLLM's PagedAttention."""
    print("\n" + "=" * 70)
    print("PAGED ATTENTION: vLLM's Memory Breakthrough")
    print("=" * 70)
    print("""
    PagedAttention is inspired by operating system virtual memory:
    contiguous virtual blocks mapped to non-contiguous physical blocks.

    Traditional KV cache: One contiguous block per sequence
    ┌──────┬──────┬──────┬──────┬──────┐
    │ SeqA │    allocated, but wasteful    │
    │ SeqB │          allocated            │
    ├──────┴───────────────────────────────┤
    │   Fragmentation wastes ~60% of GPU   │
    └──────────────────────────────────────┘

    PagedAttention: Fixed-size blocks, non-contiguous
    ┌──────┬──────┬──────┬──────┬──────┐
    │ A[0]  │ B[0]  │ A[1]  │ B[1]  │ A[2]  │
    ├──────┼──────┼──────┼──────┼──────┤
    │ B[2]  │ A[3]  │ free  │ free  │ free  │
    └──────┴──────┴──────┴──────┴──────┘
    
    Benefits:
    - Near-zero fragmentation
    - 2-4x higher memory utilization
    - Enables larger batch sizes
    - Copy-on-write for shared prefixes
    
    This is WHY vLLM can handle 2-3x more concurrent requests
    than naive Transformer implementations.
    """)


async def main():
    parser = argparse.ArgumentParser(description="Phase 2: VLM Internals Deep Dive")
    parser.add_argument("--image", default=None, help="Path to example image")
    args = parser.parse_args()

    print("=" * 70)
    print("  SpecVLM — Phase 2: Understanding VLM Internals")
    print("  Vision-Language Model Architecture Deep Dive")
    print("=" * 70)

    # Print example image info if provided
    if args.image:
        from PIL import Image
        img = Image.open(args.image)
        print(f"\nUsing image: {args.image}")
        print(f"Image size:  {img.size}")
        print(f"Image mode:  {img.mode}")

    explain_vision_encoder()
    explain_projection()
    explain_attention()
    explain_kv_cache()
    explain_prefill_vs_decode()
    explain_paged_attention()

    print("\n" + "=" * 70)
    print("Key Takeaways for Engineering")
    print("=" * 70)
    print("""
    1. Visual encoding is 30% of VLM latency — cache visual embeddings!
    2. Prefill is compute-bound — FlashAttention helps here
    3. Decode is memory-bound — memory bandwidth is the bottleneck
    4. KV cache is the largest memory consumer — PagedAttention helps
    5. Speculative decoding converts memory-bound decode into
       compute-bound verification — better GPU utilization
    6. The draft model's speedup comes from the verification being
       batched (all K tokens in one pass)
    """)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
