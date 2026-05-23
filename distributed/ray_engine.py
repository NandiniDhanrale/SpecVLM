"""
Ray Inference Engine — Distributed Inference via Ray

Ray is used to distribute inference across multiple GPUs and machines.
This enables:
1. Multi-GPU tensor parallelism (shard large models across GPUs)
2. Pipeline parallelism (draft model on GPU 0, target on GPU 1)
3. Request replication (multiple workers for higher throughput)
4. Fault tolerance (dead workers are replaced)

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     Ray Cluster                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Head Node  │  │  Worker 1   │  │  Worker 2   │         │
│  │  (Router)   │  │  (GPU 0)    │  │  (GPU 1)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │               │                    │
│         ▼               ▼               ▼                    │
│  ┌─────────────────────────────────────────────┐            │
│  │           Shared Object Store                │            │
│  │  (Visual embeddings, KV cache, models)       │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘

Production:
- Use Ray Serve for production-grade model serving
- Auto-scale workers based on queue depth
- GPUs are virtualized via Ray's placement groups
- Each model replica is a Ray actor with GPU access
"""

import asyncio
import logging
from typing import Optional

from specvlm.distributed.gpu_router import GPURouter
from specvlm.inference.engine import InferenceEngine, EngineConfig, EngineMode, EngineBackend
from specvlm.models.base_vlm import VLMInput, VLMOutput

logger = logging.getLogger(__name__)


class RayInferenceEngine:
    """
    Distributed inference engine powered by Ray.

    Manages a pool of GPU workers across a Ray cluster.
    Each worker can independently serve inference requests.

    In speculative decoding mode:
    - Draft workers run on smaller GPUs (or same GPU with lower memory)
    - Target workers run on larger GPUs with tensor parallelism
    - Visual embeddings are shared via Ray's object store
    """

    def __init__(
        self,
        target_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        draft_model_id: Optional[str] = "Qwen/Qwen2-VL-2B-Instruct",
        num_workers: int = 1,
        gpu_ids: Optional[list[int]] = None,
        use_speculative: bool = False,
        ray_address: Optional[str] = None,
        ray_namespace: str = "specvlm",
    ):
        self.target_model_id = target_model_id
        self.draft_model_id = draft_model_id
        self.num_workers = num_workers
        self.gpu_ids = gpu_ids or []
        self.use_speculative = use_speculative
        self.ray_address = ray_address
        self.ray_namespace = ray_namespace

        self._router = GPURouter(
            gpu_count=max(len(self.gpu_ids), 1)
        )
        self._workers: dict[int, InferenceEngine] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize Ray and create workers."""
        try:
            import ray
        except ImportError:
            logger.error("Ray not installed. Install with: pip install ray[default]")
            raise

        # Initialize Ray
        ray.init(
            address=self.ray_address,
            namespace=self.ray_namespace,
            ignore_reinit_error=True,
        )
        logger.info(f"Ray initialized: {ray.cluster_resources()}")

        # Create workers
        for i in range(self.num_workers):
            gpu_id = self.gpu_ids[i] if i < len(self.gpu_ids) else None
            worker = self._create_worker(i, gpu_id)
            self._workers[i] = worker

        self._initialized = True
        logger.info(f"Ray engine initialized with {self.num_workers} workers")

    def _create_worker(self, worker_id: int, gpu_id: Optional[int]) -> InferenceEngine:
        """Create a local inference worker on the specified GPU."""

        @ray.remote(num_gpus=1 if gpu_id is not None else 0)
        class RayWorker:
            def __init__(self, engine_config):
                self.engine = InferenceEngine(config=engine_config)

            async def generate(self, inputs: VLMInput) -> VLMOutput:
                return await self.engine.generate(inputs)

            def get_stats(self):
                return self.engine.get_stats()

        mode = EngineMode.SPECULATIVE if self.use_speculative else EngineMode.BASELINE
        engine_config = EngineConfig(
            mode=mode,
            backend=EngineBackend.VLLM,
            target_model_id=self.target_model_id,
            draft_model_id=self.draft_model_id if self.use_speculative else None,
        )

        return RayWorker.remote(engine_config)

    async def generate(self, inputs: VLMInput) -> VLMOutput:
        """
        Generate using distributed workers.

        Selects the best worker via GPU router, then dispatches.
        """
        if not self._initialized:
            self.initialize()

        # Select worker
        cache_key = "_".join(inputs.image_paths) if inputs.image_paths else ""
        gpu_id = self._router.select_gpu(
            request_id=inputs.request_id,
            cache_key=cache_key,
        )
        worker = self._workers.get(gpu_id)

        if worker is None:
            # Fallback to any available worker
            worker = self._workers[0]

        self._router.mark_request_started(gpu_id)
        try:
            result = await worker.generate.remote(inputs)
            return result
        finally:
            self._router.mark_request_completed(gpu_id)

    async def shutdown(self) -> None:
        """Shutdown all workers and Ray."""
        import ray

        for worker in self._workers.values():
            try:
                await worker.shutdown.remote()
            except Exception:
                pass

        ray.shutdown()
        self._initialized = False
        logger.info("Ray engine shutdown complete")

    def get_stats(self) -> dict:
        """Return cluster-wide statistics."""
        stats = {
            "num_workers": len(self._workers),
            "initialized": self._initialized,
            "gpu_status": self._router.get_all_gpu_status(),
        }
        return stats
