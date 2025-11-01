"""
History router for managing chat conversation history.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from api.models import ConversationHistory, ConversationSession
from api.services.chat_history_service import chat_history_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions", response_model=List[ConversationSession])
async def list_sessions():
    """
    List all conversation sessions.

    Returns:
        List of conversation sessions with metadata
    """
    try:
        sessions = await chat_history_service.list_sessions()
        return sessions

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """
    Get conversation history for a specific session.

    Args:
        session_id: Session ID

    Returns:
        Conversation history with all messages
    """
    try:
        history = await chat_history_service.get_conversation(session_id)
        return history

    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_conversation(session_id: str):
    """
    Delete a conversation session.

    Args:
        session_id: Session ID to delete

    Returns:
        Deletion result
    """
    try:
        await chat_history_service.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}

    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_history():
    """
    Clear all conversation history.

    Returns:
        Clear operation result
    """
    try:
        await chat_history_service.clear_all()
        return {"status": "success", "message": "All history cleared"}

    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
