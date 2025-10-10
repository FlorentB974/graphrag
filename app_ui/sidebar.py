"""Sidebar layout orchestration for the Streamlit app."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import streamlit as st

from .database import display_document_list, display_stats
from .search import get_rag_settings
from .sources import display_sources_detailed
from .upload import display_document_upload


def _find_latest_assistant_message(
    messages: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg
    return None


def render_sidebar(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Render sidebar tabs and return the latest search configuration."""
    search_config: Dict[str, Any] = {}
    container = st.container(width=10)

    with container:
        st.markdown('<div class="floating">', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📚 Sources", "🕸️ Context Graph", "📊 Database", "📁 Upload File", "⚙️"]
        )

        latest_message = _find_latest_assistant_message(messages)

        if latest_message:
            with tab1:
                if "sources" in latest_message:
                    # Display quality score at the top
                    if latest_message.get("quality_score"):
                        score_data = latest_message["quality_score"]
                        total_score = score_data.get("total", 0)
                        breakdown = score_data.get("breakdown", {})
                        confidence = score_data.get("confidence", "medium")

                        if total_score >= 80:
                            color_emoji = "🟢"
                            quality_label = "High Quality"
                        elif total_score >= 60:
                            color_emoji = "🟡"
                            quality_label = "Medium Quality"
                        else:
                            color_emoji = "🔴"
                            quality_label = "Needs Improvement"

                        st.markdown(f"### {color_emoji} Answer Quality Score")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                label="Score",
                                value=f"{total_score:.1f}%",
                                help="LLM evaluation based on multiple factors",
                            )
                        with col2:
                            st.write(f"**{quality_label}**")
                            st.caption(f"Confidence: {confidence.title()}")

                        st.progress(total_score / 100)

                        with st.expander("📊 View Score Breakdown", expanded=False):
                            st.markdown("**Quality Components:**")

                            for component, component_score in breakdown.items():
                                component_name = component.replace("_", " ").title()

                                if component_score >= 80:
                                    icon = "🟢"
                                elif component_score >= 60:
                                    icon = "🟡"
                                else:
                                    icon = "🔴"

                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.write(f"{icon} **{component_name}**")
                                with col_b:
                                    st.write(f"{component_score:.1f}%")

                                st.progress(component_score / 100)

                            st.markdown("---")
                            st.caption(
                                """
                                **Score Components Explained:**
                                - **Context Relevance**: How well the answer uses provided context
                                - **Answer Completeness**: Whether all parts of the query are addressed
                                - **Factual Grounding**: Degree of grounding in source material
                                - **Coherence**: Logical flow and readability
                                - **Citation Quality**: Proper use and attribution of sources
                                """
                            )

                        st.markdown("---")

                    display_sources_detailed(latest_message.get("sources", []))

                    if st.button(
                        "🧹 Clear chat",
                        key="clear_chat_top",
                        help="Clear conversation, graph and sources",
                    ):
                        st.session_state.messages = []
                        for key in [
                            "latest_message",
                            "latest_graph",
                            "latest_sources",
                        ]:
                            st.session_state.pop(key, None)
                        for suffix in ["_latest", "_default"]:
                            st.session_state.pop(f"search_config{suffix}", None)
                        st.rerun()

            with tab2:
                if latest_message.get("graph_fig") is not None:
                    st.plotly_chart(
                        latest_message["graph_fig"], use_container_width=True
                    )
                else:
                    st.info("No graph available yet. Ask a question to generate one!")

            with tab3:
                display_stats()
                display_document_list()

            with tab4:
                display_document_upload()

            with tab5:
                search_config = get_rag_settings(key_suffix="_latest")
                st.session_state["search_config_latest"] = search_config
        else:
            with tab1:
                st.info("💡 Start a conversation to see context information here!")

            with tab2:
                st.info("💡 Start a conversation to see context information here!")

            with tab3:
                display_stats()
                display_document_list()

            with tab4:
                display_document_upload()

            with tab5:
                search_config = get_rag_settings(key_suffix="_default")
                st.session_state["search_config_default"] = search_config

        st.markdown("</div>", unsafe_allow_html=True)

    return search_config
