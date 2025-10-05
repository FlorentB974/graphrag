"""
Quality scoring system for evaluating LLM-generated answers.
"""

import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache
import hashlib

from core.llm import llm_manager
from config.settings import settings

logger = logging.getLogger(__name__)


class QualityScorer:
    """Evaluates the quality of RAG-generated answers."""
    
    def __init__(self):
        """Initialize the quality scorer."""
        self.enabled = getattr(settings, 'enable_quality_scoring', True)
        self.weights = getattr(settings, 'quality_score_weights', {
            'context_relevance': 0.30,
            'answer_completeness': 0.25,
            'factual_grounding': 0.25,
            'coherence': 0.10,
            'citation_quality': 0.10
        })
        self.cache_enabled = getattr(settings, 'quality_score_cache_enabled', True)
        self._score_cache = {}
    
    def calculate_quality_score(
        self,
        answer: str,
        query: str,
        context_chunks: List[Dict[str, Any]],
        sources: List[Dict[str, Any]]
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
            # Check cache first
            if self.cache_enabled:
                cache_key = self._generate_cache_key(answer, query)
                cached_score = self._get_cached_score(cache_key)
                if cached_score:
                    logger.info("Using cached quality score")
                    return cached_score
            
            # Calculate individual component scores
            scores = {}
            scores['context_relevance'] = self._score_context_relevance(
                answer, context_chunks
            )
            scores['answer_completeness'] = self._score_answer_completeness(
                answer, query
            )
            scores['factual_grounding'] = self._score_factual_grounding(
                answer, context_chunks
            )
            scores['coherence'] = self._score_coherence(answer)
            scores['citation_quality'] = self._score_citation_quality(
                answer, sources
            )
            
            # Calculate weighted total
            total_score = sum(
                scores[component] * self.weights.get(component, 0.0)
                for component in scores
            )
            
            # Determine confidence level based on score variance
            confidence = self._calculate_confidence(list(scores.values()))
            
            result = {
                'total': round(total_score, 1),
                'breakdown': {k: round(v, 1) for k, v in scores.items()},
                'confidence': confidence
            }
            
            # Cache the result
            if self.cache_enabled:
                self._cache_score(cache_key, result)
            
            logger.info(f"Quality score calculated: {total_score:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return None
    
    def _score_context_relevance(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Score how well the answer uses the provided context.
        Returns score 0-100.
        """
        if not context_chunks:
            return 50.0  # Neutral score if no context
        
        try:
            # LLM-based evaluation
            context_text = "\n\n".join([
                # chunk.get('content', '')[:500]  # Limit context length
                chunk.get('content', '')  # Limit context length
                # for chunk in context_chunks[:5]  # Use top 5 chunks
                for chunk in context_chunks  # Use top 5 chunks
            ])
            
            prompt = f"""Evaluate how well this answer uses the provided context.
Score from 0-10 where:
- 10: Perfect use of context, all claims directly supported
- 7-9: Good use of context, mostly grounded
- 4-6: Moderate use, some unsupported claims
- 1-3: Poor use, mostly ignores context
- 0: Completely ungrounded or contradicts context

Context:
{context_text}

Answer:
{answer}

Respond with ONLY a number from 0-10."""

            response = llm_manager.generate_response(
                prompt=prompt,
                temperature=0.0
                # max_tokens=10
            )
            # Parse score
            try:
                score = float(response.strip()) * 10  # Convert 0-10 to 0-100
                return min(max(score, 0), 100)  # Clamp to 0-100
            except ValueError:
                logger.warning(f"Could not parse LLM score response: {response}")
                return self._heuristic_context_relevance(answer, context_chunks)
            
        except Exception as e:
            logger.warning(f"Context relevance scoring failed, using heuristic: {e}")
            return self._heuristic_context_relevance(answer, context_chunks)
    
    def _heuristic_context_relevance(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Fallback heuristic-based scoring for context relevance.
        """
        # Simple heuristic: check overlap between answer and context
        answer_words = set(answer.lower().split())
        context_words = set()
        for chunk in context_chunks[:5]:
            context_words.update(chunk.get('content', '').lower().split())
        
        if not context_words or not answer_words:
            return 50.0
        
        overlap = len(answer_words & context_words) / len(answer_words)
        # Scale to 0-100, boost score since word overlap is a rough metric
        return min(overlap * 150, 100)
    
    def _score_answer_completeness(self, answer: str, query: str) -> float:
        """
        Score whether the answer fully addresses the query.
        Returns score 0-100.
        """
        try:
            prompt = f"""Evaluate if this answer fully addresses the question.
Score from 0-10 where:
- 10: Completely answers all aspects of the question
- 7-9: Addresses main points, minor gaps
- 4-6: Partially answers, significant gaps
- 1-3: Barely addresses the question
- 0: Does not answer the question

Question:
{query}

Answer:
{answer}

Respond with ONLY a number from 0-10."""

            response = llm_manager.generate_response(
                prompt=prompt,
                temperature=0.0
                # max_tokens=10
            )
            
            try:
                score = float(response.strip()) * 10
                return min(max(score, 0), 100)
            except ValueError:
                logger.warning(f"Could not parse completeness score: {response}")
                return self._heuristic_completeness(answer, query)
            
        except Exception as e:
            logger.warning(f"Completeness scoring failed, using heuristic: {e}")
            return self._heuristic_completeness(answer, query)
    
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
    
    def _score_factual_grounding(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Score how well claims are grounded in the source material.
        Returns score 0-100.
        """
        if not context_chunks:
            return 50.0
        
        try:
            context_text = "\n\n".join([
                chunk.get('content', '')
                for chunk in context_chunks
            ])
            
            prompt = f"""Evaluate how well the answer's claims are supported by the context.
Score from 0-10 where:
- 10: Every claim is directly supported by context
- 7-9: Most claims supported, minor unsupported details
- 4-6: Some claims supported, some speculation
- 1-3: Few claims supported, mostly unsupported
- 0: Claims contradict or ignore context

Context:
{context_text}

Answer:
{answer}

Respond with ONLY a number from 0-10."""

            response = llm_manager.generate_response(
                prompt=prompt,
                temperature=0.0
                # max_tokens=10
            )
            
            try:
                score = float(response.strip()) * 10
                return min(max(score, 0), 100)
            except ValueError:
                return self._heuristic_context_relevance(answer, context_chunks)
            
        except Exception as e:
            logger.warning(f"Factual grounding scoring failed: {e}")
            # Use context relevance as fallback
            return self._heuristic_context_relevance(answer, context_chunks)
    
    def _score_coherence(self, answer: str) -> float:
        """
        Score the logical flow and readability of the answer.
        Returns score 0-100.
        """
        try:
            prompt = f"""Evaluate the coherence and clarity of this answer.
Score from 0-10 where:
- 10: Exceptionally clear, logical, well-structured
- 7-9: Clear and coherent
- 4-6: Somewhat clear, minor issues
- 1-3: Confusing or poorly structured
- 0: Incoherent

Answer:
{answer}

Respond with ONLY a number from 0-10."""

            response = llm_manager.generate_response(
                prompt=prompt,
                temperature=0.0
                # max_tokens=10
            )
            
            try:
                score = float(response.strip()) * 10
                return min(max(score, 0), 100)
            except ValueError:
                return self._heuristic_coherence(answer)
            
        except Exception as e:
            logger.warning(f"Coherence scoring failed, using heuristic: {e}")
            return self._heuristic_coherence(answer)
    
    def _heuristic_coherence(self, answer: str) -> float:
        """Fallback heuristic for coherence scoring."""
        # Check for reasonable length and sentence structure
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if not sentences:
            return 40.0
        
        # Factors: reasonable length, multiple sentences, not too short/long
        length_score = min(len(answer) / 500, 1.0) * 30
        sentence_count_score = min(len(sentences) / 3, 1.0) * 30
        avg_sentence_length = len(answer) / max(len(sentences), 1)
        sentence_length_score = 40 if 20 < avg_sentence_length < 200 else 20
        
        return length_score + sentence_count_score + sentence_length_score
    
    def _score_citation_quality(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
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
            return 'low'
        
        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        if variance < 100:  # Low variance - scores are consistent
            return 'high'
        elif variance < 400:  # Moderate variance
            return 'medium'
        else:  # High variance - inconsistent scores
            return 'low'
    
    def _generate_cache_key(self, answer: str, query: str) -> str:
        """Generate cache key for answer-query pair."""
        content = f"{query}:::{answer}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_score(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached score if available."""
        return self._score_cache.get(cache_key)
    
    def _cache_score(self, cache_key: str, score: Dict[str, Any]) -> None:
        """Cache a quality score."""
        # Simple in-memory cache with size limit
        if len(self._score_cache) > 100:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._score_cache.keys())[:20]
            for key in oldest_keys:
                del self._score_cache[key]
        
        self._score_cache[cache_key] = score


# Global instance
quality_scorer = QualityScorer()
