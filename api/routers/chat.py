"""
Chat router for handling chat requests and responses.
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, FollowUpRequest, FollowUpResponse
from api.services.chat_history_service import chat_history_service
from api.services.follow_up_service import follow_up_service
from core.quality_scorer import quality_scorer
from rag.graph_rag import graph_rag

logger = logging.getLogger(__name__)

router = APIRouter()


async def stream_response_generator(
    result: dict, session_id: str, user_query: str, context_documents: List[str]
) -> AsyncGenerator[str, None]:
    """Generate streaming response with SSE format."""
    try:
        response_text = result.get("response", "")

        # Stream response with word-based buffering for smoother rendering
        if response_text:
            # Split into words while preserving whitespace and newlines
            words = []
            current_word = ""
            
            for char in response_text:
                current_word += char
                # Break on space or newline to create natural word boundaries
                if char in {" ", "\n", "\t"}:
                    if current_word:
                        words.append(current_word)
                        current_word = ""
            
            # Add any remaining content
            if current_word:
                words.append(current_word)

            # Stream words with small delay for natural typing effect
            for word in words:
                chunk_data = {
                    "type": "token",
                    "content": word,
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.015)  # Slightly faster for smoother feel

        # Calculate quality score asynchronously
        quality_score = None
        try:
            context_chunks = result.get("graph_context", [])
            if not context_chunks:
                context_chunks = result.get("retrieved_chunks", [])

            relevant_chunks = [
                chunk
                for chunk in context_chunks
                if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
            ]

            quality_score = quality_scorer.calculate_quality_score(
                answer=response_text,
                query=user_query,
                context_chunks=relevant_chunks,
                sources=result.get("sources", []),
            )
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")

        # Generate follow-up questions
        follow_up_questions = []
        try:
            follow_up_questions = await follow_up_service.generate_follow_ups(
                query=user_query,
                response=response_text,
                sources=result.get("sources", []),
                chat_history=[],
            )
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")

        # Save to chat history
        try:
            await chat_history_service.save_message(
                session_id=session_id,
                role="user",
                content=user_query,
                context_documents=context_documents,
            )
            await chat_history_service.save_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
                sources=result.get("sources", []),
                quality_score=quality_score,
                follow_up_questions=follow_up_questions,
                context_documents=context_documents,
            )
            logger.info(f"Saved chat to history for session: {session_id}")
        except Exception as e:
            logger.warning(f"Could not save to chat history: {e}")

        # Send sources
        sources_data = {
            "type": "sources",
            "content": result.get("sources", []),
        }
        yield f"data: {json.dumps(sources_data)}\n\n"

        # Send quality score
        if quality_score:
            quality_data = {
                "type": "quality_score",
                "content": quality_score,
            }
            yield f"data: {json.dumps(quality_data)}\n\n"

        # Send follow-up questions
        if follow_up_questions:
            followup_data = {
                "type": "follow_ups",
                "content": follow_up_questions,
            }
            yield f"data: {json.dumps(followup_data)}\n\n"

        # Send metadata
        metadata_data = {
            "type": "metadata",
            "content": {
                "session_id": session_id,
                "metadata": result.get("metadata", {}),
                "context_documents": result.get("context_documents", []),
            },
        }
        yield f"data: {json.dumps(metadata_data)}\n\n"

        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Error in stream generator: {e}")
        error_data = {
            "type": "error",
            "content": str(e),
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Handle chat query request.

    Args:
        request: Chat request with message and parameters

    Returns:
        Chat response with answer, sources, and metadata
    """
    try:
        # Generate or validate session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Get chat history for this session
        chat_history = []
        if request.session_id:
            try:
                history = await chat_history_service.get_conversation(session_id)
                chat_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in history.messages
                ]
            except Exception as e:
                logger.warning(f"Could not load chat history: {e}")

        context_documents = request.context_documents or []

        # Process query through RAG pipeline
        result = graph_rag.query(
            user_query=request.message,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            temperature=request.temperature,
            use_multi_hop=request.use_multi_hop,
            chat_history=chat_history,
            context_documents=context_documents,
        )

        # If streaming is requested, return SSE stream
        if request.stream:
            return StreamingResponse(
                stream_response_generator(
                    result, session_id, request.message, context_documents
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Calculate quality score
        quality_score = None
        try:
            context_chunks = result.get("graph_context", [])
            if not context_chunks:
                context_chunks = result.get("retrieved_chunks", [])

            relevant_chunks = [
                chunk
                for chunk in context_chunks
                if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
            ]

            quality_score = quality_scorer.calculate_quality_score(
                answer=result.get("response", ""),
                query=request.message,
                context_chunks=relevant_chunks,
                sources=result.get("sources", []),
            )
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")

        # Generate follow-up questions
        follow_up_questions = []
        try:
            follow_up_questions = await follow_up_service.generate_follow_ups(
                query=request.message,
                response=result.get("response", ""),
                sources=result.get("sources", []),
                chat_history=chat_history,
            )
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")

        # Save to chat history
        try:
            await chat_history_service.save_message(
                session_id=session_id,
                role="user",
                content=request.message,
                context_documents=context_documents,
            )
            await chat_history_service.save_message(
                session_id=session_id,
                role="assistant",
                content=result.get("response", ""),
                sources=result.get("sources", []),
                quality_score=quality_score,
                follow_up_questions=follow_up_questions,
                context_documents=context_documents,
            )
        except Exception as e:
            logger.warning(f"Could not save to chat history: {e}")

        return ChatResponse(
            message=result.get("response", ""),
            sources=result.get("sources", []),
            quality_score=quality_score,
            follow_up_questions=follow_up_questions,
            session_id=session_id,
            metadata=result.get("metadata", {}),
            context_documents=result.get("context_documents", context_documents),
        )

    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/follow-ups", response_model=FollowUpResponse)
async def generate_follow_ups(request: FollowUpRequest):
    """
    Generate follow-up questions based on conversation context.

    Args:
        request: Follow-up request with query, response, and context

    Returns:
        List of follow-up questions
    """
    try:
        questions = await follow_up_service.generate_follow_ups(
            query=request.query,
            response=request.response,
            sources=request.sources,
            chat_history=request.chat_history,
        )

        return FollowUpResponse(questions=questions)

    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
