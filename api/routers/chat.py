"""
Chat router for handling chat requests and responses.
"""

import asyncio
import json
import logging
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, FollowUpRequest, FollowUpResponse
from api.services.chat_history_service import chat_history_service
from api.services.follow_up_service import follow_up_service
from config.settings import settings
from core.quality_scorer import quality_scorer
from rag.graph_rag import graph_rag

logger = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for running synchronous RAG pipeline
_executor = ThreadPoolExecutor(max_workers=4)


async def stream_response_generator(
    request: ChatRequest,
    session_id: str,
    chat_history: List[dict],
) -> AsyncGenerator[str, None]:
    """Generate streaming response with real-time token streaming via SSE.
    
    This implementation uses true LLM token streaming when enabled, providing
    significantly better time-to-first-token performance compared to the
    simulated word-splitting approach.
    """
    try:
        context_documents = request.context_documents or []
        context_document_labels = request.context_document_labels or []
        context_hashtags = request.context_hashtags or []
        
        # Check if true streaming is enabled (default: False until stabilized)
        use_true_streaming = getattr(settings, "enable_true_streaming", False)

        if use_true_streaming:
            # Use the new true token streaming approach
            async for item in _stream_with_true_tokens(
                request, session_id, chat_history,
                context_documents, context_document_labels, context_hashtags
            ):
                yield item
        else:
            # Fall back to the legacy word-splitting approach
            async for item in _stream_with_word_splitting(
                request, session_id, chat_history,
                context_documents, context_document_labels, context_hashtags
            ):
                yield item
            
    except Exception as e:
        logger.error(f"Error in stream generator: {e}")
        error_data = {
            "type": "error",
            "content": str(e),
        }
        yield f"data: {json.dumps(error_data)}\n\n"


async def _stream_with_true_tokens(
    request: ChatRequest,
    session_id: str,
    chat_history: List[dict],
    context_documents: List[str],
    context_document_labels: List[str],
    context_hashtags: List[str],
) -> AsyncGenerator[str, None]:
    """Stream response with true LLM token streaming for improved time-to-first-token.
    
    This uses a thread-safe queue to relay tokens from the synchronous LLM
    streaming API to the async SSE response.
    """
    # Use a thread-safe queue for cross-thread communication
    token_queue: queue.Queue = queue.Queue()
    collected_data: dict = {
        "sources": [],
        "metadata": {},
        "graph_context": [],
        "retrieved_chunks": [],
        "query_analysis": {},
        "full_response": "",
    }
    pipeline_error: Optional[Exception] = None
    pipeline_done = threading.Event()
    
    def run_streaming_pipeline() -> None:
        """Run the streaming RAG pipeline in a separate thread."""
        nonlocal pipeline_error
        try:
            for item in graph_rag.query_stream(
                user_query=request.message,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                temperature=request.temperature,
                use_multi_hop=request.use_multi_hop,
                chat_history=chat_history,
                context_documents=context_documents,
            ):
                if isinstance(item, dict):
                    item_type = item.get("type", "")

                    if item_type == "stage":
                        token_queue.put({"type": "stage", "content": item["content"]})
                    elif item_type == "token":
                        token_queue.put({"type": "token", "content": item["content"]})
                    elif item_type == "sources":
                        collected_data["sources"] = item["content"]
                    elif item_type == "metadata":
                        collected_data["metadata"] = item["content"]
                        collected_data["full_response"] = item["content"].get("full_response", "")
                    elif item_type == "graph_context":
                        collected_data["graph_context"] = item["content"]
                    elif item_type == "retrieved_chunks":
                        collected_data["retrieved_chunks"] = item["content"]
                    elif item_type == "query_analysis":
                        collected_data["query_analysis"] = item["content"]
                    elif item_type == "error":
                        token_queue.put({"type": "error", "content": item["content"]})
        except Exception as e:
            logger.error(f"Streaming RAG pipeline error: {e}")
            pipeline_error = e
        finally:
            pipeline_done.set()
    
    # Start pipeline in a daemon thread
    pipeline_thread = threading.Thread(target=run_streaming_pipeline, daemon=True)
    pipeline_thread.start()
    
    # Stream data as it arrives - poll queue with short timeout
    while not pipeline_done.is_set() or not token_queue.empty():
        try:
            item = token_queue.get(timeout=0.01)
            yield f"data: {json.dumps(item)}\n\n"
        except queue.Empty:
            # Allow other async tasks to run
            await asyncio.sleep(0.005)
            continue
    
    # Check for pipeline error
    if pipeline_error:
        error_data = {"type": "error", "content": str(pipeline_error)}
        yield f"data: {json.dumps(error_data)}\n\n"
        return
    
    response_text = collected_data["full_response"]
    
    # Emit quality calculation stage (AFTER response is done)
    quality_score = None
    try:
        stage_data = {
            "type": "stage",
            "content": "quality_calculation",
        }
        yield f"data: {json.dumps(stage_data)}\n\n"

        context_chunks = collected_data.get("graph_context", [])
        if not context_chunks:
            context_chunks = collected_data.get("retrieved_chunks", [])

        relevant_chunks = [
            chunk
            for chunk in context_chunks
            if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
        ]

        quality_score = quality_scorer.calculate_quality_score(
            answer=response_text,
            query=request.message,
            context_chunks=relevant_chunks,
            sources=collected_data.get("sources", []),
        )
    except Exception as e:
        logger.warning(f"Quality scoring failed: {e}")

    # Generate follow-up questions and emit suggestions stage
    follow_up_questions = []
    try:
        stage_data = {
            "type": "stage",
            "content": "suggestions",
        }
        yield f"data: {json.dumps(stage_data)}\n\n"

        follow_up_questions = await follow_up_service.generate_follow_ups(
            query=request.message,
            response=response_text,
            sources=collected_data.get("sources", []),
            chat_history=[],
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
            context_document_labels=context_document_labels,
        )
        await chat_history_service.save_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            sources=collected_data.get("sources", []),
            quality_score=quality_score,
            follow_up_questions=follow_up_questions,
            context_documents=context_documents,
            context_document_labels=context_document_labels,
            context_hashtags=context_hashtags,
        )
        logger.info(f"Saved chat to history for session: {session_id}")
    except Exception as e:
        logger.warning(f"Could not save to chat history: {e}")

    # Send sources
    sources_data = {
        "type": "sources",
        "content": collected_data.get("sources", []),
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
            "metadata": collected_data.get("metadata", {}),
            "context_documents": context_documents,
        },
    }
    yield f"data: {json.dumps(metadata_data)}\n\n"

    # Send done signal
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_with_word_splitting(
    request: ChatRequest,
    session_id: str,
    chat_history: List[dict],
    context_documents: List[str],
    context_document_labels: List[str],
    context_hashtags: List[str],
) -> AsyncGenerator[str, None]:
    """Legacy streaming implementation using word-splitting (simulated streaming)."""
    # Create async queue for real-time stage updates
    stage_queue: asyncio.Queue[str] = asyncio.Queue()
    result_holder: dict = {}
    pipeline_error: Optional[Exception] = None
    pipeline_complete = asyncio.Event()
    
    # Capture the event loop BEFORE spawning threads
    # This is critical - asyncio.get_event_loop() doesn't work from threads
    main_loop = asyncio.get_running_loop()
    
    def stage_callback(stage: str) -> None:
        """Callback invoked by RAG pipeline when a stage completes."""
        try:
            # Use call_soon_threadsafe with the captured main loop
            main_loop.call_soon_threadsafe(stage_queue.put_nowait, stage)
        except Exception as e:
            logger.warning(f"Failed to queue stage update: {e}")
    
    def run_pipeline() -> None:
        """Run the RAG pipeline in a separate thread."""
        nonlocal pipeline_error
        try:
            result = graph_rag.query(
                user_query=request.message,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                temperature=request.temperature,
                use_multi_hop=request.use_multi_hop,
                chat_history=chat_history,
                context_documents=context_documents,
                stage_callback=stage_callback,
            )
            result_holder.update(result)
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            pipeline_error = e
        finally:
            # Signal completion using the captured main loop
            try:
                main_loop.call_soon_threadsafe(pipeline_complete.set)
            except Exception as e:
                logger.warning(f"Failed to signal pipeline completion: {e}")
    
    # Start pipeline in thread pool
    main_loop.run_in_executor(_executor, run_pipeline)
    
    # Stream stage updates as they arrive
    emitted_stages = set()
    while not pipeline_complete.is_set() or not stage_queue.empty():
        try:
            # Wait for stage with timeout to check completion
            stage = await asyncio.wait_for(stage_queue.get(), timeout=0.1)
            if stage not in emitted_stages:
                emitted_stages.add(stage)
                logger.info(f"Emitting real-time stage: {stage}")
                stage_data = {"type": "stage", "content": stage}
                yield f"data: {json.dumps(stage_data)}\n\n"
        except asyncio.TimeoutError:
            # No stage available, check if pipeline is complete
            continue
    
    # Check for pipeline error
    if pipeline_error:
        error_data = {"type": "error", "content": str(pipeline_error)}
        yield f"data: {json.dumps(error_data)}\n\n"
        return
    
    result = result_holder
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

    # Emit quality calculation stage (AFTER response is done)
    quality_score = None
    try:
        # Emit quality calculation stage
        stage_data = {
            "type": "stage",
            "content": "quality_calculation",
        }
        yield f"data: {json.dumps(stage_data)}\n\n"

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
            query=request.message,
            context_chunks=relevant_chunks,
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.warning(f"Quality scoring failed: {e}")

    # Generate follow-up questions and emit suggestions stage
    follow_up_questions = []
    try:
        # Emit suggestions stage LAST
        stage_data = {
            "type": "stage",
            "content": "suggestions",
        }
        yield f"data: {json.dumps(stage_data)}\n\n"

        follow_up_questions = await follow_up_service.generate_follow_ups(
            query=request.message,
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
            content=request.message,
            context_documents=context_documents,
            context_document_labels=context_document_labels,
        )
        await chat_history_service.save_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            sources=result.get("sources", []),
            quality_score=quality_score,
            follow_up_questions=follow_up_questions,
            context_documents=context_documents,
            context_document_labels=context_document_labels,
            context_hashtags=context_hashtags,
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
        context_document_labels = request.context_document_labels or []
        context_hashtags = request.context_hashtags or []

        # If streaming is requested, return SSE stream with real-time stages
        if request.stream:
            return StreamingResponse(
                stream_response_generator(
                    request, session_id, chat_history
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: Process query through RAG pipeline synchronously
        result = graph_rag.query(
            user_query=request.message,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            temperature=request.temperature,
            use_multi_hop=request.use_multi_hop,
            chat_history=chat_history,
            context_documents=context_documents,
        )
        
        # Log the stages for debugging
        stages = result.get("stages", [])
        logger.info(f"RAG pipeline completed with stages: {stages}")

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
                context_document_labels=context_document_labels,
                context_hashtags=context_hashtags,
            )
            await chat_history_service.save_message(
                session_id=session_id,
                role="assistant",
                content=result.get("response", ""),
                sources=result.get("sources", []),
                quality_score=quality_score,
                follow_up_questions=follow_up_questions,
                context_documents=context_documents,
                context_document_labels=context_document_labels,
                context_hashtags=context_hashtags,
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
