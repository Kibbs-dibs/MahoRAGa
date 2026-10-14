"""
Production Observability & Tracing Infrastructure for Project Aletheia (core/tracing.py)

Captures execution spans, node-level latency breakdowns, AttentionRAG pruning efficiency,
and aggregates running metrics for LangSmith/OpenTelemetry compatible telemetry.
"""

from __future__ import annotations

import logging
import statistics
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("core.tracing")


class Span(BaseModel):
    """Execution span for a specific LangGraph node or subprocess."""
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: float = 0.0
    status: str = "running"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def finish(self, status: str = "ok", extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Mark span as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = round(self.end_time - self.start_time, 4)
        self.status = status
        if extra_meta:
            self.metadata.update(extra_meta)


class TraceRecord(BaseModel):
    """Complete trace of a single multi-agent pipeline query."""
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:10]}")
    query: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    total_latency: float = 0.0
    spans: List[Span] = Field(default_factory=list)
    rubric_score: int = 0
    compression_ratio: float = 1.0
    critic_verdict: str = "Pending"
    error: Optional[str] = None

    def finish(self, rubric_score: int = 0, compression_ratio: float = 1.0, critic_verdict: str = "Approved", error: Optional[str] = None) -> None:
        """Complete the trace record."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        span_total = sum(s.duration for s in self.spans)
        self.total_latency = round(max(elapsed, span_total), 3)
        self.rubric_score = rubric_score
        self.compression_ratio = compression_ratio
        self.critic_verdict = critic_verdict
        self.error = error


class PipelineTracer:
    """In-memory telemetry collector and performance metrics aggregator."""

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self.traces: List[TraceRecord] = []

    def start_trace(self, query: str) -> TraceRecord:
        """Begin tracking a new research pipeline execution."""
        record = TraceRecord(query=query)
        self.traces.append(record)
        if len(self.traces) > self.max_history:
            self.traces.pop(0)
        return record

    def record_span(self, trace: TraceRecord, name: str, duration: float, metadata: Optional[Dict[str, Any]] = None) -> Span:
        """Add a completed span to a trace."""
        span = Span(name=name, start_time=time.time() - duration, end_time=time.time(), duration=duration, status="ok", metadata=metadata or {})
        trace.spans.append(span)
        return span

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Compute running percentiles, averages, and throughput metrics."""
        if not self.traces:
            return {
                "total_queries": 0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "avg_rubric_score": 0.0,
                "avg_compression_ratio": 1.0,
                "success_rate_pct": 100.0,
                "node_durations_avg": {},
            }

        latencies = [t.total_latency for t in self.traces if t.total_latency > 0]
        rubrics = [t.rubric_score for t in self.traces if t.rubric_score > 0]
        compressions = [t.compression_ratio for t in self.traces if t.compression_ratio > 0]
        errors = [t for t in self.traces if t.error]

        # Calculate latency percentiles
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)

        # Average duration per node
        node_times: Dict[str, List[float]] = {}
        for t in self.traces:
            for s in t.spans:
                node_times.setdefault(s.name, []).append(s.duration)

        node_averages = {
            node: round(sum(times) / len(times), 3)
            for node, times in node_times.items()
        }

        success_rate = round(100.0 * (len(self.traces) - len(errors)) / len(self.traces), 1)

        return {
            "total_queries": len(self.traces),
            "latency_p50": round(p50, 2),
            "latency_p95": round(p95, 2),
            "avg_rubric_score": round(sum(rubrics) / len(rubrics), 1) if rubrics else 0.0,
            "avg_compression_ratio": round(sum(compressions) / len(compressions), 2) if compressions else 1.0,
            "success_rate_pct": success_rate,
            "node_durations_avg": node_averages,
        }


# Global singleton instance
tracer = PipelineTracer()
