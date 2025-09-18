"""
OpenAI LLM integration for the RAG pipeline.
"""
import logging
from typing import Optional, Dict, Any
import openai
from config.settings import settings

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.openai_api_key
openai.base_url = settings.openai_base_url

if settings.openai_proxy:
    openai.http_client = settings.openai_proxy


class LLMManager:
    """Manages interactions with OpenAI language models."""
    
    def __init__(self):
        """Initialize the LLM manager."""
        self.model = settings.openai_model
    
    def generate_response(self, prompt: str, system_message: Optional[str] = None,
                          temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message to set context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise
    
    def generate_rag_response(self, query: str, context_chunks: list,
                              include_sources: bool = True) -> Dict[str, Any]:
        """
        Generate a RAG response using retrieved context chunks.
        
        Args:
            query: User query
            context_chunks: List of relevant document chunks
            include_sources: Whether to include source information
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Build context from chunks
            context = "\n\n".join([
                f"[Chunk {i+1}]: {chunk.get('content', '')}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            system_message = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the given context to answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses."""
            
            prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""
            
            response = self.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3  # Lower temperature for more focused responses
            )
            
            result = {
                "answer": response,
                "query": query,
                "context_chunks": context_chunks if include_sources else [],
                "num_chunks_used": len(context_chunks)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            raise
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to extract intent and key concepts.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary containing query analysis
        """
        try:
            system_message = """Analyze the user query and extract:
1. Intent (question, request for information, etc.)
2. Key concepts and entities
3. Query type (factual, analytical, comparative, etc.)

Return your analysis in a structured format."""
            
            prompt = f"Query to analyze: {query}"
            
            analysis = self.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1  # Very low temperature for consistent analysis
            )
            
            return {
                "query": query,
                "analysis": analysis,
                "timestamp": "2024-01-01"  # You might want to add actual timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            raise


# Global LLM manager instance
llm_manager = LLMManager()