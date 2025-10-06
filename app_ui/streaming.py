"""Utilities for streaming responses in the chat UI."""

from __future__ import annotations

import time
from typing import Generator


def stream_response(text: str, delay: float = 0.02) -> Generator[str, None, None]:
    """Yield a response word-by-word to drive Streamlit's streaming UI."""
    for words in text.split(" "):
        yield words + " "
        time.sleep(delay)
