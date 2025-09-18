"""
Query analysis node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict

from core.llm import llm_manager

logger = logging.getLogger(__name__)


def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analyze user query to extract intent and key concepts.

    Args:
        query: User query string

    Returns:
        Dictionary containing query analysis
    """
    try:
        # Use LLM to analyze the query
        analysis_result = llm_manager.analyze_query(query)

        # Extract key information (simplified version)
        analysis = {
            "original_query": query,
            "query_type": "factual",  # Default type
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": analysis_result.get("analysis", ""),
            "requires_reasoning": False,
            "requires_multiple_sources": False,
        }

        # Simple heuristics to enhance analysis
        query_lower = query.lower()

        # Detect question types
        if any(
            word in query_lower for word in ["compare", "difference", "vs", "versus"]
        ):
            analysis["query_type"] = "comparative"
            analysis["requires_multiple_sources"] = True
        elif any(word in query_lower for word in ["why", "how", "explain", "reason"]):
            analysis["query_type"] = "analytical"
            analysis["requires_reasoning"] = True
        elif any(word in query_lower for word in ["what", "who", "when", "where"]):
            analysis["query_type"] = "factual"

        # Detect complexity
        if len(query.split()) > 10 or "and" in query_lower or "or" in query_lower:
            analysis["complexity"] = "complex"
            analysis["requires_multiple_sources"] = True

        # Extract potential key concepts (simple keyword extraction)
        # Skip common words
        stop_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "that",
            "this",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        words = query_lower.replace("?", "").replace("!", "").replace(",", "").split()
        key_concepts = [
            word for word in words if len(word) > 2 and word not in stop_words
        ]
        analysis["key_concepts"] = key_concepts[:5]  # Limit to top 5 concepts

        logger.info(
            f"Query analysis completed: {analysis['query_type']}, {len(key_concepts)} concepts"
        )
        return analysis

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {
            "original_query": query,
            "query_type": "factual",
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": "",
            "requires_reasoning": False,
            "requires_multiple_sources": False,
            "error": str(e),
        }
