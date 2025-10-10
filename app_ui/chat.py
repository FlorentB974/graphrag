"""Chat rendering and query handling helpers."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import streamlit as st

from rag.graph_rag import graph_rag

from .search import get_search_mode_config
from .streaming import stream_response


def render_chat_messages(main_col, messages: List[Dict[str, Any]]) -> None:
    """Render existing chat messages inside the main column."""
    for message in messages:
        with main_col.chat_message(message["role"]):
            st.markdown(message["content"])


def process_latest_user_message(main_col) -> None:
    """Process the most recent user message if it lacks an assistant response."""
    if not st.session_state.messages:
        return

    last_message = st.session_state.messages[-1]
    if last_message.get("role") != "user":
        return

    user_query = last_message.get("content", "")
    if not user_query:
        return

    with main_col:
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Generating response..."):
                    search_config = st.session_state.get(
                        "search_config_latest",
                        st.session_state.get(
                            "search_config_default",
                            get_search_mode_config("normal"),
                        ),
                    )

                    chat_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[:-1]
                    ]

                    result = graph_rag.query(
                        user_query,
                        retrieval_mode=search_config.get("retrieval_mode", "hybrid"),
                        top_k=search_config.get("top_k", 5),
                        temperature=search_config.get("temperature", 0.1),
                        chunk_weight=search_config.get("hybrid_chunk_weight", 0.6),
                        graph_expansion=search_config.get(
                            "enable_graph_expansion", True
                        ),
                        use_multi_hop=search_config.get("use_multi_hop", False),
                        chat_history=chat_history,
                    )

                full_response = result["response"]
                quality_score = None
                quality_score_lock = threading.Lock()

                def calculate_quality_async():
                    nonlocal quality_score
                    try:
                        from core.quality_scorer import quality_scorer

                        # Get relevant chunks for scoring
                        context_chunks = result.get("graph_context", [])
                        if not context_chunks:
                            context_chunks = result.get("retrieved_chunks", [])

                        # Filter out chunks with 0.000 similarity
                        relevant_chunks = [
                            chunk
                            for chunk in context_chunks
                            if chunk.get("similarity", chunk.get("hybrid_score", 0.0))
                            > 0.0
                        ]

                        score = quality_scorer.calculate_quality_score(
                            answer=full_response,
                            query=user_query,
                            context_chunks=relevant_chunks,
                            sources=result.get("sources", []),
                        )
                        with quality_score_lock:
                            quality_score = score
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).warning(
                            f"Async quality scoring failed: {e}"
                        )

                # Start scoring thread
                scoring_thread = threading.Thread(
                    target=calculate_quality_async, daemon=True
                )
                scoring_thread.start()

                # Stream response to user (this happens concurrently with scoring)
                st.write_stream(stream_response(full_response, 0.02))

                # Wait for scoring to complete (should be done by now, or very close)
                scoring_thread.join(timeout=5.0)  # Max 5 seconds wait

                # Get final quality score
                with quality_score_lock:
                    final_quality_score = quality_score

                message_data = {
                    "role": "assistant",
                    "content": full_response,
                    "query_analysis": result.get("query_analysis"),
                    "sources": result.get("sources"),
                    "quality_score": final_quality_score,
                }

                try:
                    from core.graph_viz import create_query_result_graph

                    if result.get("sources"):
                        query_fig = create_query_result_graph(
                            result["sources"], user_query
                        )
                        message_data["graph_fig"] = query_fig
                except ImportError:
                    st.warning(
                        "Graph visualization dependencies not installed. Run: pip install plotly networkx"
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    st.warning(f"Could not create query graph: {exc}")

                st.session_state.messages.append(message_data)
                st.rerun()

            except Exception as exc:  # pylint: disable=broad-except
                error_msg = f"❌ I encountered an error processing your request: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
