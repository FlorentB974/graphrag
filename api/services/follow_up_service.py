"""
Follow-up question generation service.
"""

import logging
from typing import Any, Dict, List

from core.llm import llm_manager

logger = logging.getLogger(__name__)


class FollowUpService:
    """Service for generating follow-up questions."""

    async def generate_follow_ups(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        max_questions: int = 3,
    ) -> List[str]:
        """
        Generate follow-up questions based on conversation context.

        Args:
            query: User's original query
            response: Assistant's response
            sources: Sources used in the response
            chat_history: Previous conversation messages
            max_questions: Maximum number of questions to generate

        Returns:
            List of follow-up questions
        """
        try:
            # Build context for follow-up generation
            context_info = ""
            if sources:
                # Get unique document names
                doc_names = set()
                for source in sources[:5]:  # Limit to top 5 sources
                    doc_name = source.get("document_name", source.get("filename"))
                    if doc_name:
                        doc_names.add(doc_name)

                if doc_names:
                    context_info = f"\n\nAvailable documents: {', '.join(doc_names)}"

            # Create prompt for follow-up generation
            prompt = f"""Based on this conversation, generate {max_questions} relevant follow-up questions that the user might want to ask next.

User Question: {query}

Assistant Response: {response}
{context_info}

Generate questions that:
1. Dig deeper into the topics discussed
2. Explore related concepts mentioned in the response
3. Help the user understand the subject better
4. Are specific and actionable

Return ONLY the questions, one per line, without numbering or additional text."""

            # Generate follow-up questions using LLM
            result = llm_manager.generate_response(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )

            # Parse questions from response
            questions_text = result
            questions = [
                q.strip()
                for q in questions_text.split("\n")
                if q.strip() and not q.strip().startswith("#")
            ]

            # Filter and clean questions
            valid_questions = []
            for q in questions[:max_questions]:
                # Remove numbering if present
                q = q.strip()
                if q and len(q) > 10:  # Minimum question length
                    # Remove common prefixes
                    for prefix in ["1.", "2.", "3.", "- ", "• ", "* "]:
                        if q.startswith(prefix):
                            q = q[len(prefix) :].strip()

                    # Ensure question ends with ?
                    if not q.endswith("?"):
                        q += "?"

                    valid_questions.append(q)

            logger.info(f"Generated {len(valid_questions)} follow-up questions")
            return valid_questions[:max_questions]

        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return []


# Global service instance
follow_up_service = FollowUpService()
