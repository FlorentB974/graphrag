"""
Query analysis node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict, List, Optional

from core.llm import llm_manager

logger = logging.getLogger(__name__)


def analyze_query(
    query: str, chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Analyze user query to extract intent and key concepts.

    Args:
        query: User query string
        chat_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]

    Returns:
        Dictionary containing query analysis
    """
    try:
        # Check if this is a follow-up question
        is_follow_up = False
        needs_context = False
        context_query = query  # The enriched query with context if needed

        if chat_history and len(chat_history) >= 2:
            # Detect follow-up questions using LLM
            follow_up_detection = _detect_follow_up_question(query, chat_history)
            is_follow_up = follow_up_detection.get("is_follow_up", False)
            needs_context = follow_up_detection.get("needs_context", False)

            if is_follow_up and needs_context:
                # Create a contextualized version of the query
                context_query = _create_contextualized_query(query, chat_history)
                logger.info(
                    f"Follow-up question detected. Original: '{query}' -> Contextualized: '{context_query}'"
                )


        # Extract key information using heuristics only
        analysis = {
            "original_query": query,
            "contextualized_query": context_query,
            "is_follow_up": is_follow_up,
            "needs_context": needs_context,
            "query_type": "factual",  # Default type
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": "", 
            "requires_reasoning": False,
            "requires_multiple_sources": False,
        }

        # Simple heuristics to enhance analysis (use contextualized query for better analysis)
        query_lower = context_query.lower()

        # Detect question types
        if any(
            word in query_lower
            for word in ["compare", "difference", "vs", "versus", "contrast"]
        ):
            analysis["query_type"] = "comparative"
            analysis["requires_multiple_sources"] = True
            analysis["requires_reasoning"] = True
        elif any(
            word in query_lower
            for word in [
                "why",
                "how",
                "explain",
                "reason",
                "analyze",
                "relationship",
                "connection",
            ]
        ):
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
        elif any(
            word in query_lower
            for word in [
                "relationship",
                "connection",
                "related",
                "link",
                "connect",
                "between",
            ]
        ):
            multi_hop_beneficial = True

        # 5. Queries asking about trends, patterns, or implications
        elif any(
            word in query_lower
            for word in [
                "trend",
                "pattern",
                "impact",
                "effect",
                "influence",
                "implication",
            ]
        ):
            multi_hop_beneficial = True

        # Multi-hop is NOT beneficial for:
        # 1. Simple factual lookups (addresses, names, single facts)
        # 2. Direct "what is X" questions about specific entities
        # 3. Simple definition requests
        if (
            analysis["query_type"] == "factual"
            and analysis["complexity"] == "simple"
            and len(key_concepts) <= 2
            and not analysis["requires_multiple_sources"]
        ):
            multi_hop_beneficial = False

        analysis["multi_hop_recommended"] = multi_hop_beneficial

        logger.info(
            f"Query analysis completed: {analysis['query_type']}, {len(key_concepts)} concepts, "
            f"multi-hop recommended: {multi_hop_beneficial}, is_follow_up: {is_follow_up}"
        )
        return analysis

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {
            "original_query": query,
            "contextualized_query": query,
            "is_follow_up": False,
            "needs_context": False,
            "query_type": "factual",
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": "",
            "requires_reasoning": False,
            "requires_multiple_sources": False,
            "error": str(e),
        }


def _detect_follow_up_question(
    query: str, chat_history: List[Dict[str, str]]
) -> Dict[str, bool]:
    """
    Detect if the current query is a follow-up question that requires previous context.

    Args:
        query: Current user query
        chat_history: List of previous messages

    Returns:
        Dictionary with is_follow_up and needs_context flags
    """
    try:
        # Quick heuristic checks first (for efficiency)
        query_lower = query.lower().strip()

        # Strong follow-up indicators
        follow_up_indicators = [
            "tell me more",
            "what about",
            "and",
            "also",
            "additionally",
            "his ",
            "her ",
            "their ",
            "its ",
            "this ",
            "that ",
            "these ",
            "those ",
            "he ",
            "she ",
            "they ",
            "it ",
            "more about",
            "explain",
            "clarify",
            "elaborate",
            "same",
            "similar",
            "different",
            "compared to",
        ]

        # Pronouns and references that likely need context
        context_references = [
            "he",
            "she",
            "they",
            "it",
            "this",
            "that",
            "these",
            "those",
            "him",
            "her",
            "them",
            "his",
            "her",
            "their",
            "its",
        ]

        # Check if query starts with these indicators (strong signal)
        starts_with_indicator = any(
            query_lower.startswith(indicator) for indicator in follow_up_indicators
        )

        # Check if query contains context references
        contains_reference = any(
            f" {ref} " in f" {query_lower} " or query_lower.startswith(f"{ref} ")
            for ref in context_references
        )

        # If quick checks suggest it's NOT a follow-up, return early
        if not (starts_with_indicator or contains_reference):
            # Check for questions without specific entities (likely need context)
            if len(query.split()) < 4 and "?" in query:
                # Short question, might be follow-up
                pass
            else:
                return {"is_follow_up": False, "needs_context": False}

        # Use LLM for more sophisticated detection
        # Get last 2-4 exchanges for context (limit to avoid token overflow)
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

        history_text = "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in recent_history]
        )

        detection_prompt = f"""Analyze if the current question is a follow-up question that requires context from the previous conversation.

Previous conversation:
{history_text}

Current question: {query}

A follow-up question is one that:
1. Uses pronouns (he, she, it, they, this, that) referring to previous entities
2. Asks for more details about something just discussed
3. Makes implicit references to previous topics
4. Would be ambiguous or unclear without the previous context

Answer with JSON format:
{{"is_follow_up": true/false, "needs_context": true/false, "reason": "brief explanation"}}"""

        system_message = "You are a query analyzer. Respond only with valid JSON."

        result = llm_manager.generate_response(
            prompt=detection_prompt,
            system_message=system_message,
            temperature=0.1,
            max_tokens=150,
        )

        # Parse the response
        import json

        try:
            # Try to extract JSON from the response
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            analysis = json.loads(result)
            logger.info(f"Follow-up detection: {analysis}")
            return {
                "is_follow_up": analysis.get("is_follow_up", False),
                "needs_context": analysis.get("needs_context", False),
            }
        except json.JSONDecodeError:
            # Fallback to heuristic if JSON parsing fails
            logger.warning(f"Failed to parse follow-up detection result: {result}")
            return {
                "is_follow_up": starts_with_indicator or contains_reference,
                "needs_context": starts_with_indicator or contains_reference,
            }

    except Exception as e:
        logger.error(f"Follow-up detection failed: {e}")
        # Default to safe fallback
        return {"is_follow_up": False, "needs_context": False}


def _create_contextualized_query(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Create a contextualized version of the query by incorporating relevant previous context.

    Args:
        query: Current user query
        chat_history: List of previous messages

    Returns:
        Contextualized query string
    """
    try:
        # Get last 2-4 exchanges for context
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

        history_text = "\n".join(
            [
                f"{msg['role'].title()}: {msg['content'][:500]}"  # Limit length
                for msg in recent_history
            ]
        )

        contextualization_prompt = f"""Given the conversation history and the current follow-up question, rewrite the question to be self-contained and clear without the previous context.

Previous conversation:
{history_text}

Current follow-up question: {query}

Rewrite this question to include necessary context from the conversation. The rewritten question should:
1. Be clear and specific without needing to read the previous messages
2. Replace pronouns (he, she, it, they, this, that) with specific entities or concepts
3. Include relevant context that makes the question unambiguous
4. Maintain the original intent and scope of the question

Rewritten question:"""

        system_message = "You are a query rewriter. Provide only the rewritten question without explanation."

        contextualized = llm_manager.generate_response(
            prompt=contextualization_prompt,
            system_message=system_message,
            temperature=0.1,
            max_tokens=200,
        )

        # Clean up the response
        contextualized = contextualized.strip()
        # Remove quotes if present
        if contextualized.startswith('"') and contextualized.endswith('"'):
            contextualized = contextualized[1:-1]
        if contextualized.startswith("'") and contextualized.endswith("'"):
            contextualized = contextualized[1:-1]

        logger.info(f"Contextualized query: {query} -> {contextualized}")
        return contextualized

    except Exception as e:
        logger.error(f"Query contextualization failed: {e}")
        # Return original query as fallback
        return query
