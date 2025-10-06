"""
Streamlit web interface for the GraphRAG pipeline.
"""

from __future__ import annotations

import logging

import streamlit as st

from app_ui import (
    initialize_session_state,
    process_latest_user_message,
    render_chat_messages,
    render_sidebar,
)
from config.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level))


FLOATING_SIDEBAR_STYLES = """
<style>
/* Make the right-side floating container fixed and scrollable when content overflows */
div:has( >.element-container div.floating) {
    display: flex;
    flex-direction: column;
    position: fixed;
    right: 1rem;
    top: 0rem; /* leave space for header */
    width: 33%;
    max-height: calc(100vh - 8rem);
    overflow-y: auto;
    padding-right: 0.5rem; /* avoid clipping scrollbars */
    box-sizing: border-box;
    z-index: 2000;
}

/* Hide deploy button introduced by Streamlit cloud UI if present */
.stAppDeployButton {
    display: none;
}

/* Small top padding for the main app container */
#root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 2rem;}

/* Ensure the inner floating wrapper doesn't collapse and allows scrolling */
div.floating {
    height: auto;
    min-height: 4rem;
}
</style>
"""


def configure_page() -> None:
    """Set Streamlit page options and ensure session defaults."""
    st.set_page_config(
        page_title="GraphRAG Pipeline",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session_state()


def main() -> None:
    """Main Streamlit application entry point."""
    configure_page()
    st.markdown(FLOATING_SIDEBAR_STYLES, unsafe_allow_html=True)

    main_col, sidebar_col = st.columns([2, 1])

    with main_col:
        st.title("💬 Chat with your documents")

    render_chat_messages(main_col, st.session_state.messages)

    with sidebar_col:
        render_sidebar(st.session_state.messages)

    if user_query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()

    process_latest_user_message(main_col)


if __name__ == "__main__":
    main()
