"""Thread-safe operational metrics collector for the FastAPI service."""

import threading
import time
from typing import Dict, List, Any
import numpy as np


class ServiceMetrics:
    """Collects and aggregates request counts and latency percentiles."""

    def __init__(self, max_latency_history: int = 1000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.max_latency_history = max_latency_history

        self.total_search_requests = 0
        self.total_rag_requests = 0
        self.error_count = 0

        self._latencies_ms: List[float] = []
        self._stage_latencies_sum: Dict[str, float] = {}
        self._stage_latencies_count: Dict[str, int] = {}

    def record_request(
        self,
        endpoint_type: str,  # "search" or "rag"
        latency_ms: float,
        stage_latencies: Dict[str, float],
        is_error: bool = False,
    ):
        """Records telemetry for an executed request."""
        with self._lock:
            if is_error:
                self.error_count += 1
                return

            if endpoint_type == "search":
                self.total_search_requests += 1
            elif endpoint_type == "rag":
                self.total_rag_requests += 1

            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > self.max_latency_history:
                self._latencies_ms.pop(0)

            for stage, ms in stage_latencies.items():
                self._stage_latencies_sum[stage] = self._stage_latencies_sum.get(stage, 0.0) + ms
                self._stage_latencies_count[stage] = self._stage_latencies_count.get(stage, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated metrics summary."""
        with self._lock:
            total_q = self.total_search_requests + self.total_rag_requests
            if self._latencies_ms:
                p50 = float(np.percentile(self._latencies_ms, 50))
                p95 = float(np.percentile(self._latencies_ms, 95))
                p99 = float(np.percentile(self._latencies_ms, 99))
            else:
                p50 = p95 = p99 = 0.0

            avg_stages = {}
            for stage, s_val in self._stage_latencies_sum.items():
                count = max(1, self._stage_latencies_count.get(stage, 1))
                avg_stages[stage] = round(s_val / count, 2)

            return {
                "total_queries": total_q,
                "total_search_requests": self.total_search_requests,
                "total_rag_requests": self.total_rag_requests,
                "error_count": self.error_count,
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
                "latency_p99_ms": round(p99, 2),
                "average_stage_latency_ms": avg_stages,
            }

