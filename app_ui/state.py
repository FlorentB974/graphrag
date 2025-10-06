"""Session state utilities for the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


_DEFAULT_VALUES: Dict[str, Any] = {
    "messages": [],
    "processing_files": False,
    "show_graph": False,
    "file_uploader_key": 0,
}


def initialize_session_state() -> None:
    """Ensure core session state keys are present with sensible defaults."""
    for key, default in _DEFAULT_VALUES.items():
        if key not in st.session_state:
            # Use copy for mutable defaults to avoid shared state across reruns
            st.session_state[key] = default.copy() if isinstance(default, list) else default
