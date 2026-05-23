"""
Inference Worker — Dedicated GPU Worker Process

Each worker manages one or more GPUs and runs inference requests.
In a distributed setting, multiple workers form a pool behind
a load balancer.

Worker responsibilities:
- Load and manage model instances on assigned GPUs
- Execute inference requests (sync or async)
- Report health and metrics to the scheduler
- Handle graceful shutdown and model reloading

Architecture:
┌──────────────────────────────────────────────┐
│              InferenceWorker                  │
├──────────────────────────────────────────────┤
│  GPU 0: Draft Model + Target Model (sharded) │
│  GPU 1: Target Model (tensor parallel)       │
│  ...                                          │
│  GPU N: ...                                   │
├──────────────────────────────────────────────┤
│  Request Queue → Inference Loop → Response    │
└──────────────────────────────────────────────┘

Production:
- One worker per GPU for optimal memory utilization
- Workers communicate via NCCL for tensor parallelism
- Graceful degradation: if one worker fails, requests reroute
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

from specvlm.inference.engine import InferenceEngine, EngineConfig, EngineMode, EngineBackend
from specvlm.models.base_vlm import VLMInput

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Worker configuration."""
    worker_id: str
    gpu_ids: list[int]
    engine_config: EngineConfig
    max_concurrent_requests: int = 4
    health_check_interval: float = 10.0


class InferenceWorker:
    """
    GPU worker that executes inference requests.

    Runs in its own process/thread and manages GPU memory.
    Supports hot-reloading of models and graceful shutdown.
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id = config.worker_id
        self.gpu_ids = config.gpu_ids

        # Set CUDA device visibility
        if config.gpu_ids:
            os_env_set = False
            for i, gpu_id in enumerate(config.gpu_ids):
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    if i == 0:
                        torch.cuda.set_device(gpu_id)
                    os_env_set = True

        self.engine = InferenceEngine(config=config.engine_config)
        self._running = False
        self._health_status = {"status": "initializing", "last_heartbeat": 0.0}

        # Active request tracking
        self.active_requests: dict[str, asyncio.Task] = {}
        self.total_requests_completed = 0

    async def start(self) -> None:
        """Start the worker — load models and begin processing."""
        logger.info(f"Worker {self.worker_id} starting on GPUs {self.gpu_ids}")
        await self.engine.load()
        await self.engine.warmup()

        self._running = True
        self._health_status = {"status": "ready", "last_heartbeat": time.time()}

        # Start health check loop
        asyncio.create_task(self._health_check_loop())

        logger.info(f"Worker {self.worker_id} ready")

    async def execute(self, inputs: VLMInput) -> dict:
        """
        Execute an inference request on this worker.

        Args:
            inputs: The VLM input to process

        Returns:
            dict with results and metadata
        """
        start_time = time.time()

        # Check worker health
        if not self._running:
            raise RuntimeError(f"Worker {self.worker_id} is not running")

        # Track active request
        task = asyncio.create_task(self.engine.generate(inputs))
        self.active_requests[inputs.request_id] = task

        try:
            result = await task
            elapsed_ms = (time.time() - start_time) * 1000
            self.total_requests_completed += 1

            return {
                "worker_id": self.worker_id,
                "request_id": inputs.request_id,
                "text": result.text,
                "tokens": result.tokens,
                "ttft_ms": result.ttft_ms,
                "tokens_per_second": result.tokens_per_second,
                "total_time_ms": elapsed_ms,
                "speculation_acceptance_rate": result.speculation_acceptance_rate,
                "num_tokens": result.num_output_tokens,
                "gpu_memory": self._get_gpu_memory(),
            }
        except Exception as e:
            logger.error(f"Worker {self.worker_id} request failed: {e}")
            raise
        finally:
            self.active_requests.pop(inputs.request_id, None)

    def get_available_capacity(self) -> int:
        """Return number of additional requests this worker can handle."""
        return self.config.max_concurrent_requests - len(self.active_requests)

    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return (
            self._running
            and self._health_status["status"] == "ready"
            and self.get_available_capacity() > 0
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — complete active requests, then unload models."""
        logger.info(f"Worker {self.worker_id} shutting down...")
        self._running = False

        # Wait for active requests with timeout
        if self.active_requests:
            logger.info(f"Waiting for {len(self.active_requests)} active requests...")
            done, pending = await asyncio.wait(
                self.active_requests.values(),
                timeout=30.0,
            )
            for task in pending:
                task.cancel()

        await self.engine.shutdown()
        self._health_status["status"] = "shutdown"
        logger.info(f"Worker {self.worker_id} shutdown complete")

    async def _health_check_loop(self) -> None:
        """Periodic health check and metric reporting."""
        while self._running:
            self._health_status.update({
                "status": "ready",
                "last_heartbeat": time.time(),
                "active_requests": len(self.active_requests),
                "total_completed": self.total_requests_completed,
                "capacity": self.get_available_capacity(),
                "gpu_memory": self._get_gpu_memory(),
            })
            await asyncio.sleep(self.config.health_check_interval)

    def _get_gpu_memory(self) -> dict:
        """Get GPU memory usage for assigned GPUs."""
        memory_stats = {}
        for gpu_id in self.gpu_ids:
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                memory_stats[f"gpu_{gpu_id}"] = {
                    "allocated_gb": torch.cuda.memory_allocated(gpu_id) / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved(gpu_id) / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated(gpu_id) / 1e9,
                }
        return memory_stats
