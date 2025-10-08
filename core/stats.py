"""
Utilities for collecting per-request and session-level usage statistics.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


def _extract_usage_value(usage: Any, key: str) -> int:
    """Safely extract an integer usage value from OpenAI responses."""
    if usage is None:
        return 0

    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class StatRecord:
    """Single operation statistics entry."""

    operation: str
    model: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    tokens_per_sec: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serialisable dict for UI consumption."""
        return {
            "operation": self.operation,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": self.latency_s,
            "tokens_per_sec": self.tokens_per_sec,
            "metadata": self.metadata,
        }


class RequestStats:
    """Collects statistics for a single logical request."""

    def __init__(self, request_type: str):
        self.request_type = request_type
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.events: List[StatRecord] = []

    def finish(self) -> None:
        """Mark the request as finished."""
        if self.ended_at is None:
            self.ended_at = time.time()

    @property
    def duration(self) -> float:
        """Get duration of the request in seconds."""
        end = self.ended_at or time.time()
        return max(0.0, end - self.started_at)

    def add_event(
        self,
        *,
        operation: str,
        start_time: float,
        end_time: float,
        usage: Optional[Any] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StatRecord:
        """Append a new operation event to the request."""
        latency = max(0.0, end_time - start_time)
        prompt_tokens = _extract_usage_value(usage, "prompt_tokens")
        completion_tokens = _extract_usage_value(usage, "completion_tokens")
        total_tokens = _extract_usage_value(usage, "total_tokens")
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        tokens_per_sec: Optional[float]
        if latency > 0.0 and completion_tokens > 0:
            tokens_per_sec = completion_tokens / latency
        else:
            tokens_per_sec = None

        record = StatRecord(
            operation=operation,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=latency,
            tokens_per_sec=tokens_per_sec,
            metadata=metadata or {},
        )
        self.events.append(record)
        if self.ended_at is None or end_time > self.ended_at:
            self.ended_at = end_time
        return record

    def operation_totals(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics grouped by operation."""
        aggregates: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_s": 0.0,
                "models": Counter(),
            }
        )

        for event in self.events:
            agg = aggregates[event.operation]
            agg["count"] += 1
            agg["prompt_tokens"] += event.prompt_tokens
            agg["completion_tokens"] += event.completion_tokens
            agg["total_tokens"] += event.total_tokens
            agg["latency_s"] += event.latency_s
            if event.model:
                agg["models"][event.model] += 1

        return {
            op: {
                **{k: (v if k != "models" else dict(v)) for k, v in agg.items()},
                "avg_tokens_per_sec": (
                    (agg["completion_tokens"] / agg["latency_s"])
                    if agg["latency_s"] > 0 and agg["completion_tokens"] > 0
                    else None
                ),
            }
            for op, agg in aggregates.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the request stats for storage or UI display."""
        return {
            "request_type": self.request_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "operations": self.operation_totals(),
            "events": [event.to_dict() for event in self.events],
        }


def _empty_totals() -> Dict[str, Any]:
    return {
        "count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_s": 0.0,
        "models": {},
    }


def default_session_stats() -> Dict[str, Any]:
    """Create a fresh session stats structure."""
    return {
        "requests": [],
        "totals": _empty_totals(),
        "by_operation": {},
    }


def _increment_totals(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["count"] += source.get("count", 0)
    target["prompt_tokens"] += source.get("prompt_tokens", 0)
    target["completion_tokens"] += source.get("completion_tokens", 0)
    target["total_tokens"] += source.get("total_tokens", 0)
    target["latency_s"] += source.get("latency_s", 0.0)

    models = source.get("models") or {}
    if models:
        if not target["models"]:
            target["models"] = {}
        for model, count in models.items():
            target["models"][model] = target["models"].get(model, 0) + count


def merge_request_into_session(
    session_stats: Dict[str, Any],
    request_stats: RequestStats,
    *,
    history_limit: int = 25,
) -> Dict[str, Any]:
    """Merge a completed request into a session-level accumulator."""
    if session_stats is None:
        session_stats = default_session_stats()

    totals = session_stats.setdefault("totals", _empty_totals())
    overall_increment = {
        "count": 1,
        "prompt_tokens": sum(e.prompt_tokens for e in request_stats.events),
        "completion_tokens": sum(e.completion_tokens for e in request_stats.events),
        "total_tokens": sum(e.total_tokens for e in request_stats.events),
        "latency_s": sum(e.latency_s for e in request_stats.events),
        "models": Counter(
            event.model for event in request_stats.events if event.model
        ),
    }
    # Convert Counter to dict for _increment_totals helper
    overall_increment["models"] = dict(overall_increment["models"])
    _increment_totals(totals, overall_increment)

    by_operation = session_stats.setdefault("by_operation", {})
    for op, aggregate in request_stats.operation_totals().items():
        target = by_operation.setdefault(op, _empty_totals())
        # aggregate already has avg_tokens_per_sec, but we recompute as needed when displaying
        _increment_totals(target, aggregate)

    # Append request summary and trim history
    session_stats.setdefault("requests", []).append(request_stats.to_dict())
    if len(session_stats["requests"]) > history_limit:
        session_stats["requests"] = session_stats["requests"][-history_limit:]

    return session_stats


_CURRENT_REQUEST: contextvars.ContextVar[Optional[RequestStats]] = contextvars.ContextVar(
    "current_stats_request", default=None
)


class StatsTracker:
    """Manages request-scoped statistics collection."""

    @contextlib.contextmanager
    def track_request(self, request_type: str) -> Iterator[RequestStats]:
        request = RequestStats(request_type)
        token = _CURRENT_REQUEST.set(request)
        try:
            yield request
        finally:
            request.finish()
            _CURRENT_REQUEST.reset(token)

    def get_current(self) -> Optional[RequestStats]:
        return _CURRENT_REQUEST.get()

    def record_event(
        self,
        operation: str,
        *,
        start_time: float,
        end_time: float,
        usage: Optional[Any] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request: Optional[RequestStats] = None,
    ) -> Optional[StatRecord]:
        """Record an event against the current or provided request."""
        target_request = request or self.get_current()
        if target_request is None:
            return None
        return target_request.add_event(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            usage=usage,
            model=model,
            metadata=metadata,
        )


# Global tracker instance
stats_tracker = StatsTracker()
