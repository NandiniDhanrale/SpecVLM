"""Verify all SpecVLM modules import correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

modules = [
    ("specvlm.config.settings", "Settings"),
    ("specvlm.models.base_vlm", "BaseVLM, VLMInput, VLMOutput"),
    ("specvlm.inference.engine", "InferenceEngine, EngineConfig"),
    ("specvlm.inference.visual_encoder", "VisualEncoder"),
    ("specvlm.inference.kv_cache", "KVCacheManager, PrefixCache"),
    ("specvlm.inference.token_verifier", "TokenVerifier"),
    ("specvlm.inference.speculative_decoder", "SpeculativeDecoder"),
    ("specvlm.serving.api", "create_app"),
    ("specvlm.serving.scheduler", "RequestScheduler, SchedulingPolicy"),
    ("specvlm.serving.worker", "InferenceWorker, WorkerConfig"),
    ("specvlm.serving.aggregator", "ResponseAggregator"),
    ("specvlm.distributed.gpu_router", "GPURouter, RoutingStrategy"),
    ("specvlm.monitoring.metrics", "MetricsCollector"),
    ("specvlm.monitoring.profiler", "InferenceProfiler"),
]

success = True
for module_path, names in modules:
    try:
        exec(f"from {module_path} import {names}")
        print(f"OK: {module_path}")
    except Exception as e:
        print(f"FAIL: {module_path}: {e}")
        success = False

if success:
    from specvlm.config.settings import Settings
    s = Settings()
    print(f"\nAll modules verified successfully")
    print(f"Project: {s.project_name} v{s.version}")
    print(f"Target:  {s.model.model_id}")
    print(f"Draft:   {s.inference.draft_model_id}")
    print(f"Engine:  {s.inference.engine}")
    print(f"Serving: {s.serving.host}:{s.serving.port}")
else:
    sys.exit(1)
