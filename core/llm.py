"""
OpenAI LLM integration for the RAG pipeline.
"""

import logging
from typing import Any, Dict, Optional
import httpx
import requests

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.openai_api_key
openai.base_url = settings.openai_base_url

if settings.openai_proxy:
    openai.http_client = httpx.Client(verify=False, base_url=settings.openai_proxy)


class LLMManager:
    """Manages interactions with language models (OpenAI and Ollama)."""

    def __init__(self):
        """Initialize the LLM manager."""
        self.provider = getattr(settings, 'llm_provider').lower()
        
        if self.provider == 'openai':
            self.model = settings.openai_model
        else:  # ollama
            self.model = getattr(settings, 'ollama_model')
            self.ollama_base_url = getattr(settings, 'ollama_base_url')

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> str:
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
            if self.provider == 'ollama':
                return self._generate_ollama_response(prompt, system_message, temperature, max_tokens)
            else:
                return self._generate_openai_response(prompt, system_message, temperature, max_tokens)

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise

    def _generate_openai_response(self, prompt: str, system_message: Optional[str], temperature: float, max_tokens: int) -> str:
        """Generate response using OpenAI."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    
    def _generate_ollama_response(self, prompt: str, system_message: Optional[str], temperature: float, max_tokens: int) -> str:
        """Generate response using Ollama."""
        full_prompt = ""
        if system_message:
            full_prompt += f"System: {system_message}\n\n"
        full_prompt += f"Human: {prompt}\n\nAssistant:"
        
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get('response', '')

    def generate_rag_response(
        self,
        query: str,
        context_chunks: list,
        include_sources: bool = True,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
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
            context = "\n\n".join(
                [
                    f"[Chunk {i + 1}]: {chunk.get('content', '')}"
                    for i, chunk in enumerate(context_chunks)
                ]
            )

            system_message = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the given context to answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses.

Formatting rules: return plain text only (no HTML). If the model would normally use HTML tags such as <br> or <p>, convert them to plain-text equivalents. When presenting markdown-style tables (rows with `|`), do not insert HTML tags — keep each table cell's content together on the same row. Replace `<br>` inside markdown table rows with a single space so the cell stays on one line; outside tables, replace `<br>` with a newline.

Math/LaTeX: remove common LaTeX delimiters like $...$, $$...$$, `\\(...\\)`, and `\\[...\\]` but preserve the mathematical content.
"""

            prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""

            response = self.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,  # Use the passed temperature
            )

            # Post-processing: remove HTML tags like <br> and strip LaTeX wrappers
            def _clean_text(text: str) -> str:
                import re

                if not isinstance(text, str):
                    return text

                # Process line-by-line: table rows (with '|') will be treated specially

                def _process_line(line: str) -> str:
                    # If line looks like a table row, replace <br> with a space
                    if '|' in line:
                        line = re.sub(r'(?i)<br\s*/?>', ' ', line)
                        line = re.sub(r'(?i)<p\s*/?>', '', line)
                        line = re.sub(r'(?i)</p>', '', line)
                    else:
                        line = re.sub(r'(?i)<br\s*/?>', '\n', line)
                        line = re.sub(r'(?i)<p\s*/?>', '\n', line)
                        line = re.sub(r'(?i)</p>', '\n', line)
                    return line

                # Apply line-wise processing to preserve table-row behavior
                lines = text.splitlines()
                processed_lines = [_process_line(ln) for ln in lines]
                text = '\n'.join(processed_lines)

                # Collapse excessive newlines
                text = re.sub(r'\n{3,}', '\n\n', text)

                # Strip LaTeX delimiters but keep content
                text = re.sub(r"\$\$(.*?)\$\$", lambda m: m.group(1), text, flags=re.S)
                text = re.sub(r"\$(.*?)\$", lambda m: m.group(1), text, flags=re.S)
                text = re.sub(r"\\\\\((.*?)\\\\\)", lambda m: m.group(1), text, flags=re.S)
                text = re.sub(r"\\\\\[(.*?)\\\\\]", lambda m: m.group(1), text, flags=re.S)
                text = re.sub(r"\\begin\{([a-zA-Z*]+)\}(.*?)\\end\{\1\}", lambda m: m.group(2), text, flags=re.S)

                return text.strip()

            cleaned = response
            try:
                cleaned = _clean_text(response)
            except Exception:
                cleaned = response

            result = {
                "answer": cleaned,
                "query": query,
                "context_chunks": context_chunks if include_sources else [],
                "num_chunks_used": len(context_chunks),
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
                temperature=0.1,  # Very low temperature for consistent analysis
            )

            return {
                "query": query,
                "analysis": analysis,
                "timestamp": "2024-01-01",  # You might want to add actual timestamp
            }

        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            raise


# Global LLM manager instance
llm_manager = LLMManager()
