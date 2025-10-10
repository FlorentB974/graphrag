"""
Quality scoring system for evaluating LLM-generated answers.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.llm import llm_manager

logger = logging.getLogger(__name__)


class QualityScorer:
    """Evaluates the quality of RAG-generated answers."""

    def __init__(self):
        """Initialize the quality scorer."""
        self.enabled = getattr(settings, "enable_quality_scoring", True)
        self.weights = getattr(
            settings,
            "quality_score_weights",
            {
                "context_relevance": 0.30,
                "answer_completeness": 0.25,
                "factual_grounding": 0.25,
                "coherence": 0.10,
                "citation_quality": 0.10,
            },
        )

    def calculate_quality_score(
        self,
        answer: str,
        query: str,
        context_chunks: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive quality score for an answer.

        Args:
            answer: The generated answer text
            query: The original user query
            context_chunks: List of context chunks used for generation
            sources: List of source information

        Returns:
            Dictionary with quality score and breakdown, or None if disabled
        """
        if not self.enabled or not answer:
            return None

        try:
            # Try a single LLM call to get all component scores at once
            scores = self._score_with_single_llm(answer, query, context_chunks, sources)

            # If single-LLM parsing failed, fall back to heuristic functions
            if not scores:
                logger.info("Single LLM call failed, falling back to heuristic scoring")
                scores = {}
                scores["context_relevance"] = self._heuristic_context_relevance(
                    answer, context_chunks
                )
                scores["answer_completeness"] = self._heuristic_completeness(
                    answer, query
                )
                scores["factual_grounding"] = self._heuristic_context_relevance(
                    answer, context_chunks  # Use same heuristic as context relevance
                )
                scores["coherence"] = self._heuristic_coherence(answer)

            # Always calculate citation quality (already heuristic-based)
            scores["citation_quality"] = self._score_citation_quality(answer, sources)

            # Calculate weighted total
            total_score = sum(
                scores[component] * self.weights.get(component, 0.0)
                for component in scores
            )

            # Determine confidence level based on score variance
            confidence = self._calculate_confidence(list(scores.values()))

            result = {
                "total": round(total_score, 1),
                "breakdown": {k: round(v, 1) for k, v in scores.items()},
                "confidence": confidence,
            }

            logger.info(f"Quality score calculated: {total_score:.1f}%")
            return result

        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return None

    def _score_with_single_llm(
        self,
        answer: str,
        query: str,
        context_chunks: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> Optional[Dict[str, float]]:
        """
        Try to get all metric scores in one LLM call.
        Expects the LLM to return JSON with keys:
          context_relevance, answer_completeness, factual_grounding, coherence, citation_quality
        Each value should be a number 0-10 (preferred) or 0-100.
        Returns None on failure (caller will fall back).
        """
        try:
            # Build a compact context (limit size to avoid huge prompts)
            max_chunks = 1000
            context_text = "\n\n".join(
                chunk.get("content", "")
                for chunk in (context_chunks or [])[:max_chunks]
            )
            sources_text = "\n".join(
                f"- {s.get('title', '')}: {s.get('id', '')}" for s in (sources or [])
            )

            prompt = f"""You will evaluate the QUALITY of an answer on four dimensions.
Return a single JSON object with the following keys and numeric values:
  context_relevance: Evaluate how well this answer uses the provided context,
  answer_completeness: Evaluate if this answer fully addresses the question,
  factual_grounding: Evaluate how well the answer's claims are supported by the context,
  coherence: Evaluate the coherence and clarity of this answer,
Each value must be a number between 0 and 10 (decimals allowed). Do NOT include any text outside the JSON.

Question:
{query}

Answer:
{answer}

Context (top {max_chunks} chunks):
{context_text}

Sources:
{sources_text}

Respond with only JSON, e.g. {{"context_relevance": 8.5, "answer_completeness": 9, ...}}.
"""
            response = llm_manager.generate_response(prompt=prompt, temperature=0.0)
            text = response.strip()

            # Try to extract JSON directly
            json_text = text
            # If the model added surrounding text, try to extract {...}
            if not (text.startswith("{") and text.endswith("}")):
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    json_text = m.group(0)

            try:
                parsed = json.loads(json_text)
                normalized = {}
                for k in (
                    "context_relevance",
                    "answer_completeness",
                    "factual_grounding",
                    "coherence",
                ):
                    v = parsed.get(k)
                    if v is None:
                        # If any key missing, consider this a failure to trigger fallback
                        logger.warning(f"Single-LLM result missing key: {k}")
                        return None
                    # Accept 0-10 or 0-100; normalize to 0-100
                    v = float(v)
                    if 0 <= v <= 10:
                        v = v * 10.0
                    normalized[k] = min(max(v, 0.0), 100.0)

                logger.info("Successfully used single LLM call for quality scoring")
                return normalized
            except Exception as e:
                logger.warning(
                    f"Failed parsing single-LLM JSON: {e} -- response: {text}"
                )
                return None

        except Exception as e:
            logger.warning(f"Single-LLM scoring failed: {e}")
            return None

    def _heuristic_context_relevance(
        self, answer: str, context_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Fallback heuristic-based scoring for context relevance.
        """
        # Simple heuristic: check overlap between answer and context
        answer_words = set(answer.lower().split())
        context_words = set()
        for chunk in context_chunks[:5]:
            context_words.update(chunk.get("content", "").lower().split())

        if not context_words or not answer_words:
            return 50.0

        overlap = len(answer_words & context_words) / len(answer_words)
        # Scale to 0-100, boost score since word overlap is a rough metric
        return min(overlap * 150, 100)

    def _heuristic_completeness(self, answer: str, query: str) -> float:
        """Fallback heuristic for completeness scoring."""
        # Check if answer is reasonably long and contains query terms
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Query term coverage
        coverage = len(query_words & answer_words) / max(len(query_words), 1)

        # Length score (reasonable answers should be substantial)
        length_score = min(len(answer) / 500, 1.0)  # Max at 500 chars

        # Combined score
        return ((coverage * 0.6) + (length_score * 0.4)) * 100

    def _heuristic_coherence(self, answer: str) -> float:
        """Fallback heuristic for coherence scoring."""
        # Check for reasonable length and sentence structure
        sentences = [s.strip() for s in answer.split(".") if s.strip()]

        if not sentences:
            return 40.0

        # Factors: reasonable length, multiple sentences, not too short/long
        length_score = min(len(answer) / 500, 1.0) * 30
        sentence_count_score = min(len(sentences) / 3, 1.0) * 30
        avg_sentence_length = len(answer) / max(len(sentences), 1)
        sentence_length_score = 40 if 20 < avg_sentence_length < 200 else 20

        return length_score + sentence_count_score + sentence_length_score

    def _score_citation_quality(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> float:
        """
        Score how well sources are used and attributed.
        Returns score 0-100.
        """
        if not sources:
            return 50.0  # Neutral if no sources

        # Heuristic scoring based on source count and answer length
        source_count = len(sources)

        # More sources generally indicates better grounding
        # But also consider if answer is proportional to sources
        base_score = min(source_count * 15, 80)

        # Bonus if answer length is reasonable for source count
        expected_length = source_count * 100  # ~100 chars per source
        length_ratio = len(answer) / max(expected_length, 1)

        if 0.5 <= length_ratio <= 2.0:
            base_score += 20  # Good proportion
        elif 0.3 <= length_ratio < 0.5 or 2.0 < length_ratio <= 3.0:
            base_score += 10  # Acceptable proportion

        return min(base_score, 100)

    def _calculate_confidence(self, scores: List[float]) -> str:
        """
        Calculate confidence level based on score variance.

        Args:
            scores: List of individual component scores

        Returns:
            'high', 'medium', or 'low'
        """
        if not scores:
            return "low"

        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        if variance < 100:  # Low variance - scores are consistent
            return "high"
        elif variance < 400:  # Moderate variance
            return "medium"
        else:  # High variance - inconsistent scores
            return "low"


# Global instance
quality_scorer = QualityScorer()
