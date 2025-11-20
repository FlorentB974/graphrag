"""Utilities for building per-message and per-session telemetry stats."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.models import ChatMessage


def build_message_stats(
    result: Dict[str, Any],
    request_payload: Dict[str, Any],
    context_documents: List[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Construct a stats payload for a single assistant response."""
    metrics = result.get("metrics") or {}
    metadata = result.get("metadata") or {}

    stage_timings = [dict(entry) for entry in metrics.get("stage_timings", [])]
    pipeline_stats = dict(metrics.get("pipeline", {}))
    pipeline_stats.setdefault("total_duration_ms", pipeline_stats.get("duration_ms"))
    pipeline_stats["stage_timings"] = stage_timings

    retrieval_stats = dict(metrics.get("retrieval", {}))
    retrieval_stats.setdefault("mode", request_payload.get("retrieval_mode"))
    retrieval_stats.setdefault("chunks_requested", request_payload.get("top_k"))
    retrieval_stats.setdefault("context_documents_count", len(context_documents))
    retrieval_stats["chunks_used"] = metadata.get("chunks_used")
    retrieval_stats["chunks_filtered"] = metadata.get("chunks_filtered")
    retrieval_stats["sources_returned"] = len(result.get("sources") or [])

    response_text = result.get("response", "") or ""
    response_stats = {
        "characters": len(response_text),
        "words": len(response_text.split()),
        "sources": len(result.get("sources") or []),
    }

    assistant_turn = 1
    if chat_history:
        assistant_turn = (
            sum(1 for msg in chat_history if msg.get("role") == "assistant") + 1
        )

    stats = {
        "pipeline": pipeline_stats,
        "retrieval": retrieval_stats,
        "response": response_stats,
        "llm": metrics.get("llm"),
        "session": {
            "assistant_turn": assistant_turn,
            "messages_before_response": len(chat_history or []),
        },
    }

    return stats


def append_stage_timing(
    stats: Dict[str, Any], stage: str, duration_ms: float, timestamp: Optional[float] = None
) -> None:
    """Append a new stage timing entry to the stats payload."""
    if stats is None:
        return

    pipeline = stats.setdefault("pipeline", {})
    timings = pipeline.setdefault("stage_timings", [])
    entry: Dict[str, Any] = {
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
    }
    if timestamp is not None:
        entry["timestamp"] = timestamp
    timings.append(entry)
    total = sum(item.get("duration_ms", 0.0) for item in timings)
    pipeline["total_duration_ms"] = round(total, 2)


def compute_session_stats(messages: List[ChatMessage]) -> Optional[Dict[str, Any]]:
    """Aggregate stats across all assistant messages in a conversation."""
    assistant_messages = [
        msg for msg in messages if msg.role == "assistant" and msg.stats
    ]
    if not assistant_messages:
        return None

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    latency_sum = 0.0
    duration_sum = 0.0

    for message in assistant_messages:
        llm_stats = message.stats.get("llm", {}) if message.stats else {}
        total_prompt += int(llm_stats.get("prompt_tokens") or 0)
        total_completion += int(llm_stats.get("completion_tokens") or 0)
        total_cost += float(llm_stats.get("total_cost_usd") or 0.0)
        latency_sum += float(llm_stats.get("latency_ms") or 0.0)

        pipeline = message.stats.get("pipeline", {}) if message.stats else {}
        duration_sum += float(pipeline.get("total_duration_ms") or 0.0)

    total_responses = len(assistant_messages)
    avg_latency = latency_sum / total_responses if latency_sum else None
    avg_duration = duration_sum / total_responses if duration_sum else None

    return {
        "assistant_responses": total_responses,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
        "avg_pipeline_duration_ms": round(avg_duration, 2) if avg_duration else None,
    }
