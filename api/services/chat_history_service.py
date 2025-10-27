"""Chat history service for managing conversation persistence."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from api.models import ChatMessage, ConversationHistory, ConversationSession
from core.graph_db import graph_db

logger = logging.getLogger(__name__)


class ChatHistoryService:
    """Service for managing chat conversation history."""

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        quality_score: Optional[Dict[str, Any]] = None,
        follow_up_questions: Optional[List[str]] = None,
        context_documents: Optional[List[str]] = None,
        context_document_labels: Optional[List[str]] = None,
        context_hashtags: Optional[List[str]] = None,
    ) -> None:
        """
        Save a message to chat history.

        Args:
            session_id: Conversation session ID
            role: Message role (user/assistant)
            content: Message content
            sources: Optional sources for assistant messages
            quality_score: Optional quality score for assistant messages
            follow_up_questions: Optional follow-up questions
            context_documents: Optional list of context document IDs
            context_document_labels: Optional list of context document labels/names
            context_hashtags: Optional list of context hashtags
        """
        try:
            # Use the machine's local timezone so frontend shows relative times in local context
            timestamp = datetime.now().astimezone().isoformat()

            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            # Create message node in Neo4j
            query = """
            MERGE (s:ConversationSession {session_id: $session_id})
            ON CREATE SET s.created_at = $timestamp, s.updated_at = $timestamp
            ON MATCH SET s.updated_at = $timestamp
            CREATE (m:Message {
                role: $role,
                content: $content,
                timestamp: $timestamp,
                sources: $sources,
                quality_score: $quality_score,
                follow_up_questions: $follow_up_questions,
                context_documents: $context_documents,
                context_document_labels: $context_document_labels,
                context_hashtags: $context_hashtags
            })
            CREATE (s)-[:HAS_MESSAGE]->(m)
            """

            driver.execute_query(
                query,
                session_id=session_id,
                role=role,
                content=content,
                timestamp=timestamp,
                sources=json.dumps(sources or []),
                quality_score=json.dumps(quality_score or {}),
                follow_up_questions=follow_up_questions or [],
                context_documents=context_documents or [],
                context_document_labels=context_document_labels or [],
                context_hashtags=context_hashtags or [],
            )

            logger.info(f"Saved message to session {session_id}")

        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            raise

    async def get_conversation(self, session_id: str) -> ConversationHistory:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Conversation session ID

        Returns:
            Conversation history with all messages
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = """
            MATCH (s:ConversationSession {session_id: $session_id})-[:HAS_MESSAGE]->(m:Message)
            RETURN s, m
            ORDER BY m.timestamp
            """

            result = driver.execute_query(query, session_id=session_id)

            if not result or not result.records:
                raise ValueError(f"Session {session_id} not found")

            messages = []
            session_data: Dict[str, Any] = {}

            for record in result.records:
                if not session_data:
                    session_node = record["s"]
                    session_data = {
                        "created_at": session_node.get("created_at", ""),
                        "updated_at": session_node.get("updated_at", ""),
                    }

                msg_node = record["m"]
                sources_data = msg_node.get("sources")
                if isinstance(sources_data, str):
                    try:
                        sources_data = json.loads(sources_data)
                    except json.JSONDecodeError:
                        sources_data = []

                quality_data = msg_node.get("quality_score")
                if isinstance(quality_data, str):
                    try:
                        quality_data = json.loads(quality_data)
                    except json.JSONDecodeError:
                        quality_data = None

                messages.append(
                    ChatMessage(
                        role=msg_node.get("role", ""),
                        content=msg_node.get("content", ""),
                        timestamp=msg_node.get("timestamp"),
                        sources=sources_data,
                        quality_score=quality_data,
                        follow_up_questions=msg_node.get("follow_up_questions"),
                        context_documents=msg_node.get("context_documents"),
                        context_document_labels=msg_node.get("context_document_labels"),
                        context_hashtags=msg_node.get("context_hashtags"),
                    )
                )

            # Normalize session timestamps if they are numeric epoch ms
            def _normalize_ts(value):
                try:
                    # Numeric epoch (ms) from Neo4j -> convert to local tz ISO
                    if isinstance(value, (int, float)):
                        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).astimezone().isoformat()
                    # If it's already a string, return as-is (assume it includes timezone or local time)
                    if isinstance(value, str) and value:
                        return value
                except Exception:
                    pass
                return ""

            return ConversationHistory(
                session_id=session_id,
                messages=messages,
                created_at=_normalize_ts(session_data.get("created_at", "")),
                updated_at=_normalize_ts(session_data.get("updated_at", "")),
            )

        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            raise

    async def list_sessions(self) -> List[ConversationSession]:
        """
        List all conversation sessions.

        Returns:
            List of conversation sessions
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = (
                "MATCH (s:ConversationSession)-[:HAS_MESSAGE]->(m:Message)\n"
                "WITH s, count(m) as message_count, collect(m)[0] as first_message\n"
                "RETURN s.session_id as session_id,\n"
                "       s.created_at as created_at,\n"
                "       s.updated_at as updated_at,\n"
                "       message_count,\n"
                "       first_message.content as preview\n"
                "ORDER BY s.updated_at DESC"
            )

            result = driver.execute_query(query)

            sessions = []
            if result and result.records:
                for record in result.records:
                    preview = record["preview"]
                    if preview and len(preview) > 100:
                        preview = preview[:100] + "..."

                    # Normalize created_at / updated_at to ISO8601 strings.
                    def _normalize_ts(value):
                        # If Neo4j stored an epoch milliseconds integer, convert it to local tz ISO
                        try:
                            if isinstance(value, (int, float)):
                                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).astimezone().isoformat()
                            if isinstance(value, str) and value:
                                return value
                        except Exception:
                            pass
                        return ""

                    sessions.append(
                        ConversationSession(
                            session_id=record["session_id"],
                            created_at=_normalize_ts(record.get("created_at")),
                            updated_at=_normalize_ts(record.get("updated_at")),
                            message_count=record["message_count"],
                            preview=preview,
                        )
                    )

            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            raise

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a conversation session.

        Args:
            session_id: Session ID to delete
        """
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = """
            MATCH (s:ConversationSession {session_id: $session_id})
            OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
            DETACH DELETE s, m
            """

            driver.execute_query(query, session_id=session_id)
            logger.info(f"Deleted session {session_id}")

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise

    async def clear_all(self) -> None:
        """Clear all conversation history."""
        try:
            driver = graph_db.driver
            if driver is None:
                raise RuntimeError("Neo4j driver is not initialized")

            query = """
            MATCH (s:ConversationSession)
            OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
            DETACH DELETE s, m
            """

            driver.execute_query(query)
            logger.info("Cleared all conversation history")

        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise


# Global service instance
chat_history_service = ChatHistoryService()
