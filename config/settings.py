"""
SpecVLM Configuration System

Production-grade settings management using pydantic-settings.
All configurations are loaded from environment variables with YAML overrides.
Supports multi-model, multi-GPU, and distributed serving configurations.
"""

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for a single VLM model."""

    model_id: str = Field(description="HuggingFace model ID or local path")
    model_type: Literal["draft", "target"] = "target"
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True

    # Vision-specific
    image_size: int = 336  # CLIP/ViT input resolution
    image_token_limit: int = 576  # Max visual tokens (e.g., 24x24 patches)
    vision_feature_layer: int = -2  # Which ViT layer to extract features from

    # Quantization
    quantization: Optional[Literal["awq", "gptq", "fp8", "int8", "int4"]] = None

    # Draft model specific
    speculation_length: int = 5  # k tokens to speculate
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class InferenceConfig(BaseModel):
    """Inference engine configuration."""

    engine: Literal["vllm", "transformers", "sglang"] = "vllm"
    enforce_eager: bool = False
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    seed: int = 42

    # Scheduling
    scheduling_policy: Literal["fcfs", "priority", "cache_aware"] = "cache_aware"
    max_waiting_requests: int = 1000

    # Speculative decoding
    use_speculative_decoding: bool = False
    draft_model_id: Optional[str] = None
    target_model_id: Optional[str] = None
    acceptance_fn: Literal["strict", "stochastic", "rejection_sampling"] = "strict"

    # KV cache
    enable_prefix_caching: bool = True
    max_kv_cache_size_gb: float = 0.0  # 0 = auto-detect based on GPU
    cache_pool_size: int = 1000

    # Profiling
    enable_profiling: bool = False
    torch_profile_path: Optional[str] = None


class ServingConfig(BaseModel):
    """API serving configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_concurrent_requests: int = 64
    request_timeout_seconds: int = 300
    streaming: bool = True

    # Rate limiting
    rate_limit_rpm: int = 100  # Requests per minute
    rate_limit_burst: int = 20

    # CORS
    allow_origins: list[str] = ["*"]


class DistributedConfig(BaseModel):
    """Distributed inference configuration."""

    backend: Literal["ray", "native", "none"] = "none"
    num_workers: int = 1
    gpu_ids: list[int] = Field(default_factory=list)
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1

    # Ray specific
    ray_address: Optional[str] = None
    ray_namespace: str = "specvlm"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


class Settings(BaseSettings):
    """
    Root settings for SpecVLM.
    Loads from environment variables with YAML file override.
    """

    # Project
    project_name: str = "SpecVLM"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Config file path
    config_file: Optional[str] = Field(
        default=None, env="SPECVLM_CONFIG"
    )

    # Sub-configs
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    serving: ServingConfig = ServingConfig()
    distributed: DistributedConfig = DistributedConfig()

    # Paths
    cache_dir: str = os.path.join(str(Path.home()), ".cache", "specvlm")
    data_dir: str = "data"
    results_dir: str = "results"

    class Config:
        env_prefix = "SPECVLM_"
        env_nested_delimiter = "__"
        extra = "ignore"

    def load_yaml(self, path: str) -> None:
        """Override settings from a YAML config file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    sub = getattr(self, key)
                    for sub_key, sub_val in value.items():
                        if hasattr(sub, sub_key):
                            setattr(sub, sub_key, sub_val)
                else:
                    setattr(self, key, value)


# Singleton settings instance
settings = Settings()

# Try to load YAML config if specified
if settings.config_file and os.path.exists(settings.config_file):
    settings.load_yaml(settings.config_file)
