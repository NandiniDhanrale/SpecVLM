"""
Metrics Collection — Prometheus Monitoring for SpecVLM

Tracks and exports:
- Request rate, latency, and error count
- Token generation throughput
- GPU utilization and memory
- Speculative decoding acceptance rate
- KV cache hit rate
- Queue depth and scheduling latency

Production at scale:
- Every metric is labeled by model_id, worker_id, and GPU
- Metrics are pushed to Prometheus via pushgateway or scraped
- Grafana dashboards visualize all metrics in real-time
- Alerts configured for P99 latency > 500ms or error rate > 1%
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Per-request metrics."""
    request_id: str
    model_id: str
    mode: str  # 'baseline' or 'speculative'
    ttft_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    tokens_per_second: float = 0.0
    speculation_acceptance_rate: float = 0.0
    kv_cache_hits: int = 0
    gpu_memory_gb: float = 0.0
    success: bool = True
    error_message: str = ""


class MetricsCollector:
    """
    Collects and exports Prometheus metrics for the inference server.

    Metrics are organized into:
    - Request metrics (latency, throughput, errors)
    - GPU metrics (memory, utilization)
    - Speculative decoding metrics (acceptance rate, draft speed)
    - System metrics (queue depth, scheduling latency)
    """

    def __init__(self, prometheus_port: int = 8001):
        self.prometheus_port = prometheus_port
        self._metrics = {
            # Counters
            "requests_total": Counter("specvlm_requests_total", "Total requests received", ["model", "mode"]),
            "requests_success": Counter("specvlm_requests_success_total", "Successful requests", ["model"]),
            "requests_error": Counter("specvlm_requests_error_total", "Failed requests", ["model", "error_type"]),
            "tokens_generated": Counter("specvlm_tokens_generated_total", "Tokens generated", ["model", "mode"]),
            "tokens_accepted": Counter("specvlm_tokens_accepted_total", "Speculative tokens accepted", ["model"]),
            "tokens_rejected": Counter("specvlm_tokens_rejected_total", "Speculative tokens rejected", ["model"]),

            # Histograms
            "latency_ttft": Histogram(
                "specvlm_ttft_ms", "Time to first token (ms)",
                ["model", "mode"],
                buckets=[50, 100, 200, 500, 1000, 2000, 5000],
            ),
            "latency_total": Histogram(
                "specvlm_request_latency_ms", "Total request latency (ms)",
                ["model", "mode"],
                buckets=[100, 200, 500, 1000, 2000, 5000, 10000],
            ),
            "tokens_per_second": Histogram(
                "specvlm_tokens_per_second", "Token generation rate",
                ["model", "mode"],
                buckets=[5, 10, 20, 50, 100, 200, 500],
            ),
            "acceptance_rate": Histogram(
                "specvlm_spec_acceptance_rate", "Speculative token acceptance rate",
                ["model"],
                buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            ),

            # Gauges
            "gpu_memory_used": Gauge("specvlm_gpu_memory_used_gb", "GPU memory used (GB)", ["gpu_id"]),
            "gpu_memory_total": Gauge("specvlm_gpu_memory_total_gb", "GPU memory total (GB)", ["gpu_id"]),
            "kv_cache_usage": Gauge("specvlm_kv_cache_usage_pct", "KV cache utilization %"),
            "queue_depth": Gauge("specvlm_queue_depth", "Current request queue depth"),
            "active_requests": Gauge("specvlm_active_requests", "Currently processing requests"),
        }

        self._recent_requests: list[RequestMetrics] = []

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics server on port {self.prometheus_port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")
        else:
            logger.info("prometheus_client not available; metrics disabled")

    def record_request(self, metrics: RequestMetrics) -> None:
        """Record metrics for a completed request."""
        model = metrics.model_id
        mode = metrics.mode

        # Counters
        self._metrics["requests_total"].labels(model=model, mode=mode).inc()

        if metrics.success:
            self._metrics["requests_success"].labels(model=model).inc()
        else:
            self._metrics["requests_error"].labels(
                model=model,
                error_type=metrics.error_message[:50] if metrics.error_message else "unknown",
            ).inc()

        # Tokens
        self._metrics["tokens_generated"].labels(model=model, mode=mode).inc(metrics.num_output_tokens)

        if mode == "speculative":
            accepted = int(metrics.num_output_tokens * metrics.speculation_acceptance_rate)
            rejected = metrics.num_output_tokens - accepted
            self._metrics["tokens_accepted"].labels(model=model).inc(accepted)
            self._metrics["tokens_rejected"].labels(model=model).inc(rejected)

        # Latency histograms
        self._metrics["latency_ttft"].labels(model=model, mode=mode).observe(metrics.ttft_ms)
        self._metrics["latency_total"].labels(model=model, mode=mode).observe(metrics.total_time_ms)
        self._metrics["tokens_per_second"].labels(model=model, mode=mode).observe(metrics.tokens_per_second)

        if mode == "speculative" and metrics.speculation_acceptance_rate > 0:
            self._metrics["acceptance_rate"].labels(model=model).observe(metrics.speculation_acceptance_rate)

        # Store recent for debugging
        self._recent_requests.append(metrics)
        if len(self._recent_requests) > 1000:
            self._recent_requests.pop(0)

    def update_gpu_metrics(self) -> None:
        """Update GPU memory gauges."""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    used = torch.cuda.memory_allocated(i) / 1e9
                    total = props.total_memory / 1e9
                    self._metrics["gpu_memory_used"].labels(gpu_id=str(i)).set(used)
                    self._metrics["gpu_memory_total"].labels(gpu_id=str(i)).set(total)
        except Exception:
            pass

    def update_queue_depth(self, depth: int) -> None:
        """Update queue depth gauge."""
        if PROMETHEUS_AVAILABLE:
            self._metrics["queue_depth"].set(depth)

    def update_active_requests(self, count: int) -> None:
        """Update active requests gauge."""
        if PROMETHEUS_AVAILABLE:
            self._metrics["active_requests"].set(count)
