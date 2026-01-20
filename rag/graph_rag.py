"""
LangGraph-based RAG pipeline implementation.
"""

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from langgraph.graph import END, StateGraph

from rag.nodes.generation import generate_response
from rag.nodes.graph_reasoning import reason_with_graph
from rag.nodes.query_analysis import analyze_query
from rag.nodes.retrieval import retrieve_documents

logger = logging.getLogger(__name__)


class RAGState:
    """State management for the RAG pipeline."""

    def __init__(self):
        """Initialize RAG state."""
        self.query: str = ""
        self.query_analysis: Dict[str, Any] = {}
        self.retrieved_chunks: List[Dict[str, Any]] = []
        self.graph_context: List[Dict[str, Any]] = []
        self.response: str = ""
        self.sources: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.quality_score: Optional[Dict[str, Any]] = None
        self.context_documents: List[str] = []
        self.stages: List[str] = []  # Track stages for UI
        self.stage_callback: Optional[Callable[[str], None]] = None  # Real-time stage callback


class GraphRAG:
    """LangGraph-based RAG pipeline orchestrator."""

    def __init__(self):
        """Initialize the GraphRAG pipeline."""
        # workflow may be a LangGraph compiled graph — keep as Any to avoid static typing issues
        self.workflow: Any = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow for RAG."""
        # Use plain dict as the runtime state type for LangGraph.
        # StateGraph accepts a dict type annotation for dynamic state management.
        workflow: Any = StateGraph(dict)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("reason_with_graph", self._reason_with_graph_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Add edges
        workflow.add_edge("analyze_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "reason_with_graph")
        workflow.add_edge("reason_with_graph", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        return workflow.compile()

    def _analyze_query_node(self, state) -> Any:
        """Analyze the user query (dict-based state for LangGraph)."""
        try:
            query = state.get("query", "")
            chat_history = state.get("chat_history", [])
            logger.info(f"Analyzing query: {query}")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage and notify callback for real-time streaming
            state["stages"].append("query_analysis")
            if state.get("stage_callback"):
                state["stage_callback"]("query_analysis")
            logger.info(f"Stage query_analysis completed, current stages: {state['stages']}")
            
            state["query_analysis"] = analyze_query(query, chat_history)
            return state
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["query_analysis"] = {"error": str(e)}
            return state

    def _retrieve_documents_node(self, state) -> Any:
        """Retrieve relevant documents (dict-based state for LangGraph)."""
        try:
            logger.info("Retrieving relevant documents")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track retrieval stage and notify callback for real-time streaming
            state["stages"].append("retrieval")
            if state.get("stage_callback"):
                state["stage_callback"]("retrieval")
            logger.info(f"Stage retrieval completed, current stages: {state['stages']}")
            
            # Pass additional retrieval tuning parameters from state
            chunk_weight = state.get("chunk_weight", 0.5)
            graph_expansion = state.get("graph_expansion", True)
            use_multi_hop = state.get("use_multi_hop", False)

            state["retrieved_chunks"] = retrieve_documents(
                state.get("query", ""),
                state.get("query_analysis", {}),
                state.get("retrieval_mode", "graph_enhanced"),
                state.get("top_k", 5),
                chunk_weight=chunk_weight,
                graph_expansion=graph_expansion,
                use_multi_hop=use_multi_hop,
                context_documents=state.get("context_documents", []),
            )
            
            return state
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_chunks"] = []
            return state

    def _reason_with_graph_node(self, state) -> Any:
        """Perform graph-based reasoning (dict-based state for LangGraph)."""
        try:
            logger.info("Performing graph reasoning")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage and notify callback for real-time streaming
            state["stages"].append("graph_reasoning")
            if state.get("stage_callback"):
                state["stage_callback"]("graph_reasoning")
            logger.info(f"Stage graph_reasoning completed, current stages: {state['stages']}")
            
            state["graph_context"] = reason_with_graph(
                state.get("query", ""),
                state.get("retrieved_chunks", []),
                state.get("query_analysis", {}),
                state.get("retrieval_mode", "graph_enhanced"),
            )
            return state
        except Exception as e:
            logger.error(f"Graph reasoning failed: {e}")
            state["graph_context"] = state.get("retrieved_chunks", [])
            return state

    def _generate_response_node(self, state) -> Any:
        """Generate the final response (dict-based state for LangGraph)."""
        try:
            logger.info("Generating response")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage and notify callback for real-time streaming
            state["stages"].append("generation")
            if state.get("stage_callback"):
                state["stage_callback"]("generation")
            logger.info(f"Stage generation completed, current stages: {state['stages']}")
            
            response_data = generate_response(
                state.get("query", ""),
                state.get("graph_context", []),
                state.get("query_analysis", {}),
                state.get("temperature", 0.7),
                state.get("chat_history", []),
            )

            state["response"] = response_data.get("response", "")
            state["sources"] = response_data.get("sources", [])
            state["metadata"] = response_data.get("metadata", {})
            # Capture quality score computed during generation (if available)
            state["quality_score"] = response_data.get("quality_score", None)

            return state
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["response"] = f"I apologize, but I encountered an error: {str(e)}"
            state["sources"] = []
            state["metadata"] = {"error": str(e)}
            return state

    def query(
        self,
        user_query: str,
        retrieval_mode: str = "graph_enhanced",
        top_k: int = 5,
        temperature: float = 0.7,
        chunk_weight: float = 0.5,
        graph_expansion: bool = True,
        use_multi_hop: bool = False,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        context_documents: Optional[List[str]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: User's question or request
            retrieval_mode: Retrieval strategy ("simple", "graph_enhanced", "hybrid")
            top_k: Number of chunks to retrieve
            temperature: LLM temperature for response generation
            chunk_weight: Weight for chunk-based results in hybrid mode
            graph_expansion: Whether to use graph expansion
            use_multi_hop: Whether to use multi-hop reasoning
            chat_history: Optional conversation history for follow-up questions

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Initialize state object and convert to dict for the workflow
            state_obj = RAGState()
            state_obj.query = user_query
            state = state_obj.__dict__.copy()

            # Add RAG parameters to state
            state["retrieval_mode"] = retrieval_mode
            state["top_k"] = top_k
            state["temperature"] = temperature
            # Include hybrid tuning options provided by caller
            state["chunk_weight"] = chunk_weight
            state["graph_expansion"] = graph_expansion
            state["use_multi_hop"] = use_multi_hop
            # Add chat history for follow-up questions
            state["chat_history"] = chat_history or []
            state["context_documents"] = context_documents or []
            # Add stage callback for real-time streaming
            state["stage_callback"] = stage_callback

            # Run the workflow with a dict-based state
            logger.info(f"Processing query through RAG pipeline: {user_query}")
            final_state_dict = self.workflow.invoke(state)

            # Rebuild RAGState object from returned dict for backward compatibility
            final_state = RAGState()
            for k, v in (final_state_dict or {}).items():
                setattr(final_state, k, v)

            context_docs = getattr(final_state, "context_documents", [])
            metadata = getattr(final_state, "metadata", {}) or {}
            if context_docs:
                metadata = {**metadata, "context_documents": context_docs}
                setattr(final_state, "metadata", metadata)

            # Return results
            return {
                "query": user_query,
                "response": getattr(final_state, "response", ""),
                "sources": getattr(final_state, "sources", []),
                "retrieved_chunks": getattr(final_state, "retrieved_chunks", []),
                "graph_context": getattr(final_state, "graph_context", []),
                "query_analysis": getattr(final_state, "query_analysis", {}),
                "metadata": getattr(final_state, "metadata", {}),
                "quality_score": getattr(final_state, "quality_score", None),
                "context_documents": context_docs,
                "stages": getattr(final_state, "stages", []),
            }

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "sources": [],
                "retrieved_chunks": [],
                "graph_context": [],
                "query_analysis": {},
                "metadata": {"error": str(e)},
                "quality_score": None,
                "context_documents": context_documents or [],
                "stages": [],
            }

    def query_stream(
        self,
        user_query: str,
        retrieval_mode: str = "graph_enhanced",
        top_k: int = 5,
        temperature: float = 0.7,
        chunk_weight: float = 0.5,
        graph_expansion: bool = True,
        use_multi_hop: bool = False,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        context_documents: Optional[List[str]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[Union[Dict[str, Any], str]]:
        """
        Process a user query through the RAG pipeline with true token streaming.
        
        This method yields stage updates and then streams tokens from the LLM
        in real-time, providing a much better time-to-first-token experience.

        Args:
            user_query: User's question or request
            retrieval_mode: Retrieval strategy ("simple", "graph_enhanced", "hybrid")
            top_k: Number of chunks to retrieve
            temperature: LLM temperature for response generation
            chunk_weight: Weight for chunk-based results in hybrid mode
            graph_expansion: Whether to use graph expansion
            use_multi_hop: Whether to use multi-hop reasoning
            chat_history: Optional conversation history for follow-up questions
            context_documents: Optional list of document IDs to restrict retrieval
            stage_callback: Optional callback for stage notifications

        Yields:
            - Dict with "type": "stage" for stage updates
            - Dict with "type": "token" containing streaming tokens
            - Dict with "type": "sources" at the end with source information
            - Dict with "type": "metadata" at the end with metadata
        """
        from core.llm import llm_manager
        
        try:
            # Step 1: Query Analysis
            if stage_callback:
                stage_callback("query_analysis")
            yield {"type": "stage", "content": "query_analysis"}
            
            query_analysis = analyze_query(user_query, chat_history or [])
            
            # Step 2: Retrieval
            if stage_callback:
                stage_callback("retrieval")
            yield {"type": "stage", "content": "retrieval"}
            
            retrieved_chunks = retrieve_documents(
                user_query,
                query_analysis,
                retrieval_mode,
                top_k,
                chunk_weight=chunk_weight,
                graph_expansion=graph_expansion,
                use_multi_hop=use_multi_hop,
                context_documents=context_documents or [],
            )
            
            # Step 3: Graph Reasoning
            if stage_callback:
                stage_callback("graph_reasoning")
            yield {"type": "stage", "content": "graph_reasoning"}
            
            graph_context = reason_with_graph(
                user_query,
                retrieved_chunks,
                query_analysis,
                retrieval_mode,
            )
            
            # Step 4: Generation (streaming)
            if stage_callback:
                stage_callback("generation")
            yield {"type": "stage", "content": "generation"}
            
            # Filter out chunks with 0.000 similarity before processing
            relevant_chunks = [
                chunk
                for chunk in graph_context
                if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
            ]
            
            if not relevant_chunks:
                yield {"type": "token", "content": "I couldn't find any relevant information to answer your question."}
                yield {"type": "sources", "content": []}
                yield {"type": "metadata", "content": {"chunks_used": 0}}
                return

            # Stream tokens from the LLM
            full_response = []
            for token in llm_manager.generate_rag_response_stream(
                query=user_query,
                context_chunks=relevant_chunks,
                include_sources=True,
                temperature=temperature,
                chat_history=chat_history if query_analysis.get("is_follow_up") else [],
            ):
                full_response.append(token)
                yield {"type": "token", "content": token}
            
            # Prepare sources information
            sources = []
            for i, chunk in enumerate(relevant_chunks):
                source_info = {
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "content": chunk.get("content", ""),
                    "similarity": chunk.get("similarity", chunk.get("hybrid_score", 0.0)),
                    "document_name": chunk.get("document_name", "Unknown Document"),
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get(
                        "filename", chunk.get("document_name", "Unknown Document")
                    ),
                    "metadata": chunk.get("metadata", {}),
                    "chunk_index": chunk.get("chunk_index"),
                }
                
                # Add entity information if available
                contained_entities = chunk.get("contained_entities", [])
                relevant_entities = chunk.get("relevant_entities", [])
                entities = relevant_entities or contained_entities
                
                if entities:
                    source_info["contained_entities"] = entities
                    source_info["entity_enhanced"] = True
                
                sources.append(source_info)
            
            # Yield sources
            yield {"type": "sources", "content": sources}
            
            # Yield metadata
            query_type = query_analysis.get("query_type", "factual")
            complexity = query_analysis.get("complexity", "simple")
            
            metadata = {
                "chunks_used": len(relevant_chunks),
                "chunks_filtered": len(graph_context) - len(relevant_chunks),
                "query_type": query_type,
                "complexity": complexity,
                "requires_reasoning": query_analysis.get("requires_reasoning", False),
                "key_concepts": query_analysis.get("key_concepts", []),
                "full_response": "".join(full_response),
            }
            
            yield {"type": "metadata", "content": metadata}
            yield {"type": "graph_context", "content": graph_context}
            yield {"type": "retrieved_chunks", "content": retrieved_chunks}
            yield {"type": "query_analysis", "content": query_analysis}
            
        except Exception as e:
            logger.error(f"RAG streaming pipeline failed: {e}")
            yield {"type": "error", "content": str(e)}

    async def aquery(self, user_query: str) -> Dict[str, Any]:
        """
        Async version of query processing.

        Args:
            user_query: User's question or request

        Returns:
            Dictionary containing response and metadata
        """
        # For now, just call the sync version
        # Future enhancement: implement full async pipeline
        return self.query(user_query)


# Global GraphRAG instance
graph_rag = GraphRAG()
