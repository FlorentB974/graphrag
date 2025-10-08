"""Session state utilities for the Streamlit UI."""

from __future__ import annotations

import copy
from typing import Any, Dict

import streamlit as st

from core.stats import default_session_stats

_DEFAULT_VALUES: Dict[str, Any] = {
    "messages": [],
    "processing_files": False,
    "show_graph": False,
    "file_uploader_key": 0,
    "session_stats": default_session_stats,
    "latest_request_stats": None,
}


def initialize_session_state() -> None:
    """Ensure core session state keys are present with sensible defaults."""
    for key, default in _DEFAULT_VALUES.items():
        if key not in st.session_state:
            if callable(default):
                st.session_state[key] = default()
            elif isinstance(default, (dict, list, set)):
                st.session_state[key] = copy.deepcopy(default)
            else:
                st.session_state[key] = default
