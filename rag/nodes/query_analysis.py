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
            word in query_lower for word in ["compare", "difference", "vs", "versus", "contrast"]
        ):
            analysis["query_type"] = "comparative"
            analysis["requires_multiple_sources"] = True
            analysis["requires_reasoning"] = True
        elif any(word in query_lower for word in ["why", "how", "explain", "reason", "analyze", "relationship", "connection"]):
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

        # Determine if multi-hop reasoning would be beneficial
        multi_hop_beneficial = False
        
        # Multi-hop is beneficial for:
        # 1. Comparative queries (need to connect multiple entities)
        if analysis["query_type"] == "comparative":
            multi_hop_beneficial = True
        
        # 2. Analytical queries that need reasoning (relationships, explanations)
        elif analysis["query_type"] == "analytical" and analysis["requires_reasoning"]:
            multi_hop_beneficial = True
            
        # 3. Complex queries with multiple concepts
        elif analysis["complexity"] == "complex" and len(key_concepts) >= 3:
            multi_hop_beneficial = True
            
        # 4. Queries explicitly asking for relationships or connections
        elif any(word in query_lower for word in ["relationship", "connection", "related", "link", "connect", "between"]):
            multi_hop_beneficial = True
            
        # 5. Queries asking about trends, patterns, or implications
        elif any(word in query_lower for word in ["trend", "pattern", "impact", "effect", "influence", "implication"]):
            multi_hop_beneficial = True

        # Multi-hop is NOT beneficial for:
        # 1. Simple factual lookups (addresses, names, single facts)
        # 2. Direct "what is X" questions about specific entities
        # 3. Simple definition requests
        if (analysis["query_type"] == "factual"
            and analysis["complexity"] == "simple"
            and len(key_concepts) <= 2
            and not analysis["requires_multiple_sources"]):
            multi_hop_beneficial = False

        analysis["multi_hop_recommended"] = multi_hop_beneficial

        logger.info(
            f"Query analysis completed: {analysis['query_type']}, {len(key_concepts)} concepts, "
            f"multi-hop recommended: {multi_hop_beneficial}"
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
