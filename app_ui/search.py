"""Controls for configuring RAG search parameters in the sidebar."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from config.settings import settings


def get_search_mode_config(search_mode: str) -> Dict[str, Any]:
    """Get configuration parameters for different search modes."""
    configs = {
        "quick": {
            "min_retrieval_similarity": 0.3,
            "hybrid_chunk_weight": 0.8,
            "enable_graph_expansion": False,
            "max_expanded_chunks": 50,
            "max_entity_connections": 5,
            "max_chunk_connections": 3,
            "expansion_similarity_threshold": 0.4,
            "max_expansion_depth": 1,
            "top_k": 3,
            "temperature": 0.1,
            "retrieval_mode": (
                "hybrid" if settings.enable_entity_extraction else "chunk_only"
            ),
            "use_multi_hop": True,
        },
        "normal": {
            "min_retrieval_similarity": 0.1,
            "hybrid_chunk_weight": 0.6,
            "enable_graph_expansion": True,
            "max_expanded_chunks": 500,
            "max_entity_connections": 20,
            "max_chunk_connections": 10,
            "expansion_similarity_threshold": 0.1,
            "max_expansion_depth": 2,
            "top_k": 5,
            "temperature": 0.1,
            "retrieval_mode": (
                "hybrid" if settings.enable_entity_extraction else "chunk_only"
            ),
            "use_multi_hop": True,
        },
        "deep": {
            "min_retrieval_similarity": 0.05,
            "hybrid_chunk_weight": 0.4,
            "enable_graph_expansion": True,
            "max_expanded_chunks": 1000,
            "max_entity_connections": 50,
            "max_chunk_connections": 20,
            "expansion_similarity_threshold": 0.05,
            "max_expansion_depth": 3,
            "top_k": 10,
            "temperature": 0.1,
            "retrieval_mode": (
                "hybrid" if settings.enable_entity_extraction else "chunk_only"
            ),
            "use_multi_hop": True,
        },
    }
    return configs.get(search_mode, configs["normal"])


def get_rag_settings(key_suffix: str = ""):
    """Render RAG settings controls in the sidebar and return their values."""
    st.markdown("### 🧠 Search Settings")

    # Search Mode Selection
    search_modes = ["quick", "normal", "deep"]
    search_mode_labels = ["🚀 Quick Search", "⚖️ Normal Search", "🔍 Deep Search"]

    search_mode = st.selectbox(
        "Search Mode",
        search_modes,
        format_func=lambda mode: search_mode_labels[search_modes.index(mode)],
        index=1,
        key=f"search_mode{key_suffix}",
        help="""
        Choose your search strategy:
        • **Quick**: Fast results, fewer chunks, minimal graph traversal
        • **Normal**: Balanced performance and comprehensiveness (recommended)
        • **Deep**: Comprehensive search, more context, extensive graph exploration
        """,
    )

    # Get base configuration for selected mode
    config = get_search_mode_config(search_mode)

    # Brief explanation of current mode
    mode_explanations = {
        "quick": "🚀 **Quick mode**: Optimized for speed with focused results. Uses fewer chunks and minimal graph expansion. Multi-hop reasoning automatically applied when beneficial.",
        "normal": "⚖️ **Normal mode**: Balanced approach providing good coverage without overwhelming context. Multi-hop reasoning automatically applied when beneficial. Best for most queries.",
        "deep": "🔍 **Deep mode**: Maximum comprehensiveness with extensive graph exploration. Multi-hop reasoning automatically applied when beneficial for complex queries.",
    }

    st.info(mode_explanations[search_mode])

    # Show entity extraction status if not using quick mode
    if search_mode != "quick":
        if settings.enable_entity_extraction:
            st.success("✅ Entity extraction enabled - Enhanced search available")
        else:
            st.warning("⚠️ Entity extraction disabled - Using chunk-only search")
            config["retrieval_mode"] = "chunk_only"

    # Advanced Settings Expander
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.markdown("**Current Configuration:**")

        # Display current configuration in a more readable format
        col1, col2 = st.columns(2)

        with col1:
            retrieval_mode = (
                str(config.get("retrieval_mode", "")).replace("_", " ").title()
            )
            st.write(f"**Retrieval Mode:** {retrieval_mode}")
            st.write(f"**Chunks Retrieved:** {int(config['top_k'])}")
            st.write(f"**Temperature:** {float(config['temperature'])}")
            st.write(f"**Min Similarity:** {float(config['min_retrieval_similarity'])}")

        with col2:
            st.write(
                f"**Graph Expansion:** {'✅' if config['enable_graph_expansion'] else '❌'}"
            )
            st.write(f"**Max Expanded Chunks:** {int(config['max_expanded_chunks'])}")
            st.write(f"**Chunk Weight:** {float(config['hybrid_chunk_weight'])}")
            st.write(f"**Expansion Depth:** {int(config['max_expansion_depth'])}")

        st.markdown("---")
        st.markdown("**Override Settings** (optional):")

        # Allow users to override specific settings
        use_custom = st.checkbox(
            "Customize parameters",
            key=f"use_custom{key_suffix}",
            help="Enable to modify individual parameters",
        )

        if use_custom:
            # Basic settings
            st.markdown("**Basic Settings:**")

            # Update retrieval modes to match hybrid approach
            mode_options = ["chunk_only", "entity_only", "hybrid"]
            mode_labels = [
                "Chunk Only (Traditional)",
                "Entity Only (GraphRAG)",
                "Hybrid (Best of Both)",
            ]

            if not settings.enable_entity_extraction:
                mode_options = ["chunk_only"]
                mode_labels = ["Chunk Only (Traditional)"]

            current_mode_index = 0
            if config["retrieval_mode"] in mode_options:
                current_mode_index = mode_options.index(config["retrieval_mode"])

            config["retrieval_mode"] = st.selectbox(
                "Retrieval Strategy",
                mode_options,
                format_func=lambda mode: (
                    mode_labels[mode_options.index(mode)]
                    if mode in mode_options
                    else mode
                ),
                index=current_mode_index,
                key=f"retrieval_mode_custom{key_suffix}",
            )

            config["top_k"] = st.slider(
                "Number of chunks to retrieve",
                min_value=1,
                max_value=20,
                value=int(config["top_k"]),
                key=f"top_k_custom{key_suffix}",
            )

            config["temperature"] = st.slider(
                "Response creativity (temperature)",
                min_value=0.0,
                max_value=1.0,
                value=float(config["temperature"]),
                step=0.1,
                key=f"temperature_custom{key_suffix}",
            )

            # Advanced retrieval settings
            st.markdown("**Advanced Retrieval:**")

            config["min_retrieval_similarity"] = st.slider(
                "Minimum similarity threshold",
                min_value=0.0,
                max_value=0.5,
                value=float(config["min_retrieval_similarity"]),
                step=0.05,
                key=f"min_similarity_custom{key_suffix}",
                help="Lower values retrieve more chunks, higher values are more selective",
            )

            if config["retrieval_mode"] == "hybrid":
                config["hybrid_chunk_weight"] = st.slider(
                    "Chunk weight (vs Entity weight)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(config["hybrid_chunk_weight"]),
                    step=0.1,
                    key=f"chunk_weight_custom{key_suffix}",
                    help="Higher values favor chunk-based results, lower values favor entity-based results",
                )

            # Graph expansion settings
            st.markdown("**Graph Expansion:**")

            config["enable_graph_expansion"] = st.checkbox(
                "Enable graph expansion",
                value=bool(config["enable_graph_expansion"]),
                key=f"graph_expansion_custom{key_suffix}",
                help="Use entity relationships to expand context",
            )

            if config["enable_graph_expansion"]:
                config["max_expanded_chunks"] = st.number_input(
                    "Max expanded chunks",
                    min_value=50,
                    max_value=2000,
                    value=int(config["max_expanded_chunks"]),
                    step=50,
                    key=f"max_chunks_custom{key_suffix}",
                )

                config["max_expansion_depth"] = st.slider(
                    "Max expansion depth",
                    min_value=1,
                    max_value=5,
                    value=int(config["max_expansion_depth"]),
                    key=f"max_depth_custom{key_suffix}",
                    help="How many hops to follow in the graph",
                )

                config["expansion_similarity_threshold"] = st.slider(
                    "Expansion similarity threshold",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(config["expansion_similarity_threshold"]),
                    step=0.05,
                    key=f"expansion_threshold_custom{key_suffix}",
                    help="Minimum similarity for expanding through relationships",
                )

    return config
