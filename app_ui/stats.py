"""Utilities for formatting and grouping session statistics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

_OPERATION_LABELS = {
    "embedding": "Embeddings",
    "upload_embedding": "Upload Embeddings",
    "query_analysis": "Query Analysis",
    "query_analysis_follow_up": "Query Analysis",
    "query_analysis_contextualize": "Query Analysis",
    "multi_hop": "Multi-hop Retrieval",
    "llm_answer": "LLM Answer",
    "llm_continue": "LLM Answer",
    "quality_score": "Quality Score",
    "entity_extraction": "Entity Extraction",
    "llm_merge": "LLM Merge",
    "model_introspection": "Model Introspection",
}

_OPERATION_ORDER = [
    "Embeddings",
    "Upload Embeddings",
    "Query Analysis",
    "Multi-hop Retrieval",
    "LLM Answer",
    "Quality Score",
    "Entity Extraction",
    "LLM Merge",
    "Model Introspection",
]


def _aggregate_operation_metrics(operations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group raw operation metrics under human friendly labels."""
    aggregated: Dict[str, Dict[str, Any]] = {}

    for key, metrics in (operations or {}).items():
        label = _OPERATION_LABELS.get(key, key.replace("_", " ").title())
        target = aggregated.setdefault(
            label,
            {
                "count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_s": 0.0,
                "models": {},
            },
        )

        target["count"] += metrics.get("count", 0)
        target["prompt_tokens"] += metrics.get("prompt_tokens", 0)
        target["completion_tokens"] += metrics.get("completion_tokens", 0)
        target["total_tokens"] += metrics.get("total_tokens", 0)
        target["latency_s"] += metrics.get("latency_s", 0.0)

        for model, count in (metrics.get("models") or {}).items():
            target_models = target["models"]
            target_models[model] = target_models.get(model, 0) + count

    for data in aggregated.values():
        completion = data.get("completion_tokens", 0)
        latency = data.get("latency_s", 0.0)
        if latency > 0 and completion > 0:
            data["avg_tokens_per_sec"] = completion / latency
        else:
            data["avg_tokens_per_sec"] = None

    return aggregated


def _order_operations(operations: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return operation labels in a friendly display order."""
    ordered = []
    remaining = set(operations.keys())
    for label in _OPERATION_ORDER:
        if label in remaining:
            ordered.append(label)
            remaining.remove(label)
    ordered.extend(sorted(remaining))
    return ordered


def _format_models(models: Dict[str, int]) -> str:
    if not models:
        return ""
    parts = [
        f"{name} ({count})" for name, count in sorted(models.items(), key=lambda x: (-x[1], x[0]))
    ]
    return ", ".join(parts)


def _format_operation_line(label: str, data: Dict[str, Any]) -> str:
    count = data.get("count", 0)
    prompt_tokens = data.get("prompt_tokens", 0)
    completion_tokens = data.get("completion_tokens", 0)
    total_tokens = data.get("total_tokens", 0)
    latency = data.get("latency_s", 0.0)
    tokens_per_sec = data.get("avg_tokens_per_sec")
    models = _format_models(data.get("models", {}))

    segments = [
        f"- **{label}**",
        f"{count} call{'s' if count != 1 else ''}",
        f"prompt {prompt_tokens:,}",
        f"completion {completion_tokens:,}",
        f"total {total_tokens:,}",
    ]
    if latency > 0:
        segments.append(f"latency {latency:.2f}s")
    if tokens_per_sec:
        segments.append(f"{tokens_per_sec:.1f} tok/s")
    if models:
        segments.append(f"models: {models}")

    return " · ".join(segments)


def _summarize_operation_totals(operations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_s": 0.0,
    }

    for data in operations.values():
        summary["count"] += data.get("count", 0)
        summary["prompt_tokens"] += data.get("prompt_tokens", 0)
        summary["completion_tokens"] += data.get("completion_tokens", 0)
        summary["total_tokens"] += data.get("total_tokens", 0)
        summary["latency_s"] += data.get("latency_s", 0.0)

    return summary


def _render_metric_grid(pairs: Sequence[Tuple[str, str]], columns: int = 2) -> None:
    if not pairs:
        return

    for start in range(0, len(pairs), columns):
        chunk = pairs[start : start + columns]
        cols = st.columns(len(chunk))
        for col, (label, value) in zip(cols, chunk):
            with col:
                st.metric(label, value)


def _format_operation_block(label: str, data: Dict[str, Any]) -> str:
    lines = [f"**{label}**"]
    lines.append(f"- Calls: {data.get('count', 0)}")
    lines.append(f"- Prompt tokens: {data.get('prompt_tokens', 0):,}")
    lines.append(f"- Completion tokens: {data.get('completion_tokens', 0):,}")
    lines.append(f"- Total tokens: {data.get('total_tokens', 0):,}")

    latency = data.get("latency_s", 0.0)
    if latency:
        lines.append(f"- Latency: {latency:.2f}s")

    tokens_per_sec = data.get("avg_tokens_per_sec")
    if tokens_per_sec:
        lines.append(f"- Speed: {tokens_per_sec:.1f} tok/s")

    models = _format_models(data.get("models", {}))
    if models:
        lines.append(f"- Models: {models}")

    return "\n".join(lines)


def _render_request_section(request_summary: Dict[str, Any]) -> None:
    request_ops = _aggregate_operation_metrics(request_summary.get("operations", {}))
    totals = _summarize_operation_totals(request_ops)

    duration = request_summary.get("duration")
    metrics: List[Tuple[str, str]] = []
    if duration is not None:
        metrics.append(("Duration", f"{duration:.2f}s"))
    metrics.append(("Operations", f"{totals['count']:,}"))
    metrics.append(("Prompt tokens", f"{totals['prompt_tokens']:,}"))
    metrics.append(("Completion tokens", f"{totals['completion_tokens']:,}"))
    metrics.append(("Total tokens", f"{totals['total_tokens']:,}"))

    _render_metric_grid(metrics)

    if request_ops:
        # st.caption("Last request breakdown")
        with st.expander("Breakdown", expanded=False):
            for label in _order_operations(request_ops):
                st.markdown(_format_operation_block(label, request_ops[label]))
    else:
        st.caption("No tracked operations for the last request.")


def _render_session_section(session_stats: Dict[str, Any]) -> None:
    totals = session_stats.get("totals", {}) or {}
    session_metrics: List[Tuple[str, str]] = [
        ("Requests", f"{totals.get('count', 0):,}"),
        ("Prompt tokens", f"{totals.get('prompt_tokens', 0):,}"),
        ("Completion tokens", f"{totals.get('completion_tokens', 0):,}"),
        ("Total tokens", f"{totals.get('total_tokens', 0):,}"),
    ]

    latency = totals.get("latency_s")
    if latency:
        session_metrics.append(("Latency", f"{latency:.1f}s"))

    _render_metric_grid(session_metrics)

    session_ops = _aggregate_operation_metrics(session_stats.get("by_operation", {}))
    if session_ops:
        with st.expander("Breakdown", expanded=False):
            for label in _order_operations(session_ops):
                st.markdown(_format_operation_block(label, session_ops[label]))
    else:
        st.caption("No session operations recorded yet.")


def render_stats_panel(
    request_summary: Optional[Dict[str, Any]],
    session_stats: Optional[Dict[str, Any]],
) -> None:
    """Render compact statistics suitable for the sidebar."""

    if not request_summary and not session_stats:
        st.info("No stats available yet. Ask a question to generate usage metrics.")
        return

    if request_summary:
        st.markdown("#### 🔁 Last Request")
        _render_request_section(request_summary)
    else:
        st.caption("No assistant responses yet in this session.")

    has_session_data = (
        session_stats
        and (
            session_stats.get("totals")
            or session_stats.get("by_operation")
            or session_stats.get("requests")
        )
    )

    if has_session_data:
        if request_summary:
            st.divider()
        st.markdown("#### 🧠 Session Overview")
        _render_session_section(session_stats or {})


def build_stats_markdown(
    request_summary: Dict[str, Any],
    session_stats: Optional[Dict[str, Any]] = None,
    *,
    include_session: bool = True,
) -> str:
    """Generate a Markdown summary for request and session statistics."""

    lines: List[str] = ["**Last request stats**"]
    request_ops = _aggregate_operation_metrics(request_summary.get("operations", {}))

    if request_ops:
        for label in _order_operations(request_ops):
            lines.append(_format_operation_line(label, request_ops[label]))
    else:
        lines.append("- No tracked operations")

    if include_session and session_stats:
        totals = session_stats.get("totals", {})
        session_count = totals.get("count") or len(session_stats.get("requests", []))
        lines.append("")
        lines.append("**Session totals**")
        lines.append(f"- Requests: {session_count}")
        lines.append(f"- Prompt tokens: {totals.get('prompt_tokens', 0):,}")
        lines.append(f"- Completion tokens: {totals.get('completion_tokens', 0):,}")
        lines.append(f"- Total tokens: {totals.get('total_tokens', 0):,}")

        session_ops = _aggregate_operation_metrics(session_stats.get("by_operation", {}))
        if session_ops:
            lines.append("")
            lines.append("**Session by operation**")
            for label in _order_operations(session_ops):
                lines.append(_format_operation_line(label, session_ops[label]))

    return "\n".join(lines)
