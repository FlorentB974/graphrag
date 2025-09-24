"""
LangGraph-based RAG pipeline implementation.
"""

import logging
from typing import Any, Dict, List

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


class GraphRAG:
    """LangGraph-based RAG pipeline orchestrator."""

    def __init__(self):
        """Initialize the GraphRAG pipeline."""
        # workflow may be a LangGraph compiled graph — keep as Any to avoid static typing issues
        self.workflow: Any = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow for RAG."""
        # Use plain dict as the runtime state type for LangGraph. Keep as Any to silence type checkers.
        workflow: Any = StateGraph(dict)  # type: ignore

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
            logger.info(f"Analyzing query: {query}")
            state["query_analysis"] = analyze_query(query)
            return state
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["query_analysis"] = {"error": str(e)}
            return state

    def _retrieve_documents_node(self, state) -> Any:
        """Retrieve relevant documents (dict-based state for LangGraph)."""
        try:
            logger.info("Retrieving relevant documents")
            # Pass additional retrieval tuning parameters from state
            chunk_weight = state.get("chunk_weight", 0.5)
            graph_expansion = state.get("graph_expansion", True)

            state["retrieved_chunks"] = retrieve_documents(
                state.get("query", ""),
                state.get("query_analysis", {}),
                state.get("retrieval_mode", "graph_enhanced"),
                state.get("top_k", 5),
                chunk_weight=chunk_weight,
                graph_expansion=graph_expansion,
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
            response_data = generate_response(
                state.get("query", ""),
                state.get("graph_context", []),
                state.get("query_analysis", {}),
                state.get("temperature", 0.7),
            )

            state["response"] = response_data.get("response", "")
            state["sources"] = response_data.get("sources", [])
            state["metadata"] = response_data.get("metadata", {})

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
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: User's question or request
            retrieval_mode: Retrieval strategy ("simple", "graph_enhanced", "hybrid")
            top_k: Number of chunks to retrieve
            temperature: LLM temperature for response generation

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

            # Run the workflow with a dict-based state
            logger.info(f"Processing query through RAG pipeline: {user_query}")
            final_state_dict = self.workflow.invoke(state)

            # Rebuild RAGState object from returned dict for backward compatibility
            final_state = RAGState()
            for k, v in (final_state_dict or {}).items():
                setattr(final_state, k, v)

            # Return results
            return {
                "query": user_query,
                "response": getattr(final_state, "response", ""),
                "sources": getattr(final_state, "sources", []),
                "retrieved_chunks": getattr(final_state, "retrieved_chunks", []),
                "graph_context": getattr(final_state, "graph_context", []),
                "query_analysis": getattr(final_state, "query_analysis", {}),
                "metadata": getattr(final_state, "metadata", {}),
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
            }

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
