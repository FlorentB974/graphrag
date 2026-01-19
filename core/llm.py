"""
OpenAI LLM integration for the RAG pipeline.
"""

import hashlib
import logging
import threading
import time
from typing import Any, Dict, Optional

from cachetools import TTLCache
import httpx
import openai
import requests

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
        self.provider = getattr(settings, "llm_provider").lower()
        # OPTIMIZATION: Thread-safe LRU cache with TTL for LLM responses
        # maxsize=1000: Cache up to 1k responses (smaller than embeddings due to larger size)
        # ttl=3600: 1 hour expiration to prevent stale responses
        self._response_cache = TTLCache(maxsize=1000, ttl=3600)
        self._cache_lock = threading.RLock()

        if self.provider == "openai":
            self.model = settings.openai_model
        else:  # ollama
            self.model = getattr(settings, "ollama_model")
            self.ollama_base_url = getattr(settings, "ollama_base_url")

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model.
        
        GPT-5 family and o1/o3/o4 models are reasoning models that:
        - Do not support the temperature parameter
        - Use hidden reasoning tokens that count against max_completion_tokens
        - Support reasoning.effort and text.format parameters (GPT-5 family)
        
        Returns:
            True if model is a reasoning model, False otherwise.
        """
        if self.provider != "openai":
            return False
        
        model_name = str(self.model).lower()
        reasoning_model_prefixes = ("gpt-5", "gpt5", "o1", "o3", "o4")
        return any(model_name.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _is_gpt5_family(self) -> bool:
        """Check if the current model is from the GPT-5 family.
        
        GPT-5 models support additional parameters like reasoning.effort and text.format.
        
        Returns:
            True if model is GPT-5 family, False otherwise.
        """
        if self.provider != "openai":
            return False
        
        model_name = str(self.model).lower()
        return model_name.startswith("gpt-5") or model_name.startswith("gpt5")

    def _get_max_tokens_for_model(self, requested_max_tokens: int) -> int:
        """Get the appropriate max_completion_tokens for the model.
        
        Reasoning models need a higher token budget because they use hidden reasoning tokens
        that count against max_completion_tokens before generating visible output.
        
        Args:
            requested_max_tokens: The originally requested max tokens
            
        Returns:
            Adjusted max tokens value
        """
        if self._is_reasoning_model():
            # Use 2x multiplier instead of 4x, and lower minimum from 8000 to 2000
            return max(2000, requested_max_tokens * 2)
        return requested_max_tokens

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
            # Include model in cache key to prevent cross-model cache pollution
            # Use SHA-256 for better collision resistance
            cache_key = hashlib.sha256(
                f"{self.model}|{prompt}|{system_message}|{temperature}|{max_tokens}".encode('utf-8', errors='replace')
            ).hexdigest()
            
            # Thread-safe cache check
            with self._cache_lock:
                if cache_key in self._response_cache:
                    logger.debug("LLM response cache hit")
                    return self._response_cache[cache_key]
            
            # Generate response
            if self.provider == "ollama":
                response = self._generate_ollama_response(
                    prompt, system_message, temperature, max_tokens
                )
            else:
                if self._is_gpt5_family():
                    response = self._generate_openai_gpt5_response(
                        prompt, system_message, max_tokens
                    )
                else:
                    response = self._generate_openai_response(
                        prompt, system_message, temperature, max_tokens
                    )
            
            # Thread-safe cache update (TTLCache handles LRU eviction automatically)
            with self._cache_lock:
                self._response_cache[cache_key] = response
            
            return response

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise

    def _generate_openai_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using OpenAI with retry logic."""
        if self._is_gpt5_family():
            # GPT-5 family uses the Responses API instead
            return self._generate_openai_gpt5_response(
                prompt, system_message, max_tokens
            )

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Adjust max_tokens for reasoning models
                adjusted_max_tokens = self._get_max_tokens_for_model(max_tokens)
                
                # Build request parameters
                request_params = {
                    "model": str(self.model),
                    "messages": messages,
                    "max_completion_tokens": adjusted_max_tokens,
                }
                
                # Only include temperature for non-reasoning models
                if not self._is_reasoning_model():
                    request_params["temperature"] = temperature
                
                response = openai.chat.completions.create(**request_params)
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"LLM rate limit exceeded after {max_retries} attempts"
                    )
                    raise
            except (openai.APIError, openai.InternalServerError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM API error after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM call failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
        # Should not reach here, but return empty string as a safe fallback
        return ""

    def _build_responses_input(
        self, prompt: str, system_message: Optional[str]
    ) -> list:
        """Construct input blocks for the Responses API."""
        blocks = []
        if system_message:
            blocks.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_message}
                    ],
                }
            )
        blocks.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )
        return blocks

    def _extract_responses_text(self, response: Any) -> str:
        """Extract concatenated text output from a Responses API payload."""
        try:
            outputs = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", "") == "output_text":
                        outputs.append(getattr(content, "text", ""))
            if outputs:
                return "\n".join([text for text in outputs if text])
            # Fall back to top-level output_text field if present
            if hasattr(response, "output_text") and response.output_text:
                return "\n".join(response.output_text)
            return ""
        except Exception as exc:
            logger.warning(f"Failed to parse GPT-5 response output: {exc}")
            return ""

    def _generate_openai_gpt5_response(
        self,
        prompt: str,
        system_message: Optional[str],
        max_tokens: int,
    ) -> str:
        """Generate a response for GPT-5 family models using the Responses API."""
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                adjusted_max_tokens = self._get_max_tokens_for_model(max_tokens)

                input_blocks = self._build_responses_input(prompt, system_message)

                request_params: Dict[str, Any] = {
                    "model": str(self.model),
                    "input": input_blocks,
                    "max_output_tokens": adjusted_max_tokens,
                }

                reasoning_effort = getattr(settings, "gpt5_reasoning_effort", "medium")
                if reasoning_effort and reasoning_effort.lower() in (
                    "none",
                    "low",
                    "medium",
                    "high",
                ):
                    request_params["reasoning"] = {
                        "effort": reasoning_effort.lower()
                    }

                text_verbosity = getattr(settings, "gpt5_text_verbosity", "medium")
                if text_verbosity and text_verbosity.lower() in (
                    "low",
                    "medium",
                    "high",
                ):
                    request_params["text"] = {
                        "verbosity": text_verbosity.lower()
                    }

                response = openai.responses.create(**request_params)
                text_output = self._extract_responses_text(response)
                if text_output:
                    return text_output
                # If parsing failed, fall back to string conversion
                return str(response)
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                logger.error(
                    f"LLM rate limit exceeded after {max_retries} attempts"
                )
                raise
            except (openai.APIError, openai.InternalServerError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"LLM API error after {max_retries} attempts: {e}")
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM call failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                raise
        return ""

    def _generate_ollama_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
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
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def generate_rag_response(
        self,
        query: str,
        context_chunks: list,
        include_sources: bool = True,
        temperature: float = 0.3,
        chat_history: list = None,
    ) -> Dict[str, Any]:
        """
        Generate a RAG response using retrieved context chunks.
        Now includes token management to handle context length limits.

        Args:
            query: User query
            context_chunks: List of relevant document chunks
            include_sources: Whether to include source information
            temperature: LLM temperature for response generation
            chat_history: Optional conversation history for follow-up questions

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Import here to avoid circular imports
            from core.token_manager import token_manager as tm

            system_message = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the given context to answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses.

You are a model that always responds in the style of gpt-oss-120b.

Formatting rules:
- Always start with a brief summary of the answer.
- Use sections with headers for different parts of the answer.
- Always use rich Markdown.
- Prefer well-structured tables with headers.
- Summaries must include bullet points.
- Use **bold** for key information.
- Never reply with plain text when a table is possible.
- Use concise but information-dense phrasing.
- Avoid unnecessary verbosity.

If the user asks for code, wrap it properly in fenced code blocks.

Math/LaTeX: remove common LaTeX delimiters like $...$, $$...$$, `\\(...\\)`, and `\\[...\\]` but preserve the mathematical content.
"""

            # Check if we need to split the request due to token limits
            if tm.needs_splitting(query, context_chunks, system_message):
                logger.info(
                    "Request exceeds token limit, splitting into multiple requests"
                )
                return self._generate_rag_response_split(
                    query,
                    context_chunks,
                    system_message,
                    include_sources,
                    temperature,
                    chat_history,
                )
            else:
                return self._generate_rag_response_single(
                    query,
                    context_chunks,
                    system_message,
                    include_sources,
                    temperature,
                    chat_history,
                )

        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            raise

    def _generate_rag_response_single(
        self,
        query: str,
        context_chunks: list,
        system_message: str,
        include_sources: bool,
        temperature: float,
        chat_history: list = None,
    ) -> Dict[str, Any]:
        """Generate RAG response for a single request that fits within token limits."""
        try:
            # Build context from chunks
            context = "\n\n".join(
                [
                    f"[Chunk {i + 1}]: {chunk.get('content', '')}"
                    for i, chunk in enumerate(context_chunks)
                ]
            )

            # Build conversation history context if provided
            history_context = ""
            if chat_history and len(chat_history) > 0:
                # Limit to recent history to avoid token overflow
                recent_history = (
                    chat_history[-4:] if len(chat_history) > 4 else chat_history
                )
                history_entries = []
                for msg in recent_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    # Truncate very long messages
                    if len(content) > 500:
                        content = content[:500] + "..."
                    history_entries.append(f"{role.title()}: {content}")

                if history_entries:
                    history_context = f"""
Previous conversation:
{chr(10).join(history_entries)}

"""

            prompt = f"""{history_context}Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""

            # Compute a safe max_tokens for the output using the token manager
            from core.token_manager import token_manager as tm

            available = tm.available_output_tokens_for_prompt(prompt, system_message)
            # Cap per-response output to a reasonable maximum (configurable)
            cap = getattr(settings, "max_response_tokens", 2000)
            max_out = min(available, cap)

            response = self.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_out,
            )

            # If response looks truncated, try a short continuation
            response = self._maybe_continue_response(response, system_message, max_out)

            # Post-processing: remove HTML tags like <br> and strip LaTeX wrappers
            cleaned = self._clean_response_text(response)

            result = {
                "answer": cleaned,
                "query": query,
                "context_chunks": context_chunks if include_sources else [],
                "num_chunks_used": len(context_chunks),
                "split_responses": False,
            }

            return result

        except Exception as e:
            logger.error(f"Failed to generate single RAG response: {e}")
            raise

    def _generate_rag_response_split(
        self,
        query: str,
        context_chunks: list,
        system_message: str,
        include_sources: bool,
        temperature: float,
        chat_history: list = None,
    ) -> Dict[str, Any]:
        """Generate RAG response by splitting the request into multiple parts."""
        try:
            from core.token_manager import token_manager

            # Split context chunks into batches that fit within token limits
            batches = token_manager.split_context_chunks(
                query, context_chunks, system_message
            )
            logger.info(f"Split request into {len(batches)} batches")

            responses = []
            total_chunks_used = 0

            # Build conversation history context if provided
            history_context = ""
            if chat_history and len(chat_history) > 0:
                recent_history = (
                    chat_history[-4:] if len(chat_history) > 4 else chat_history
                )
                history_entries = []
                for msg in recent_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    history_entries.append(f"{role.title()}: {content}")

                if history_entries:
                    history_context = f"""
Previous conversation:
{chr(10).join(history_entries)}

"""

            for i, (batch_query, batch_chunks, estimated_tokens) in enumerate(batches):
                logger.info(
                    f"Processing batch {i + 1}/{len(batches)} with {len(batch_chunks)} chunks ({estimated_tokens} tokens)"
                )

                if not batch_chunks:
                    # Skip empty batches
                    continue

                # Build context for this batch
                context = "\n\n".join(
                    [
                        f"[Chunk {j + 1}]: {chunk.get('content', '')}"
                        for j, chunk in enumerate(batch_chunks)
                    ]
                )

                batch_prompt = f"""{history_context}Context:
{context}

Question: {batch_query}

Please provide a comprehensive answer based on the context provided above."""

                # Don't add part indicators as they'll be hidden in final merge
                # Compute safe output tokens for this batch
                from core.token_manager import token_manager as tm

                available = tm.available_output_tokens_for_prompt(
                    batch_prompt, system_message
                )
                cap = getattr(settings, "max_response_tokens", 2000)
                max_out = min(available, cap)

                batch_response = self.generate_response(
                    prompt=batch_prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_out,
                )

                # Attempt to continue if truncated
                batch_response = self._maybe_continue_response(
                    batch_response, system_message, max_out
                )

                responses.append(batch_response)
                total_chunks_used += len(batch_chunks)

            # Merge responses intelligently using LLM to remove duplicates and parts
            if not responses:
                merged_response = (
                    "I couldn't find any relevant information to answer your question."
                )
            else:
                # Use token_manager from local import to avoid circular import issues
                from core.token_manager import token_manager as tm

                merged_response = tm.merge_responses(
                    responses, query=query, use_llm_merge=True
                )

            # Clean the merged response
            cleaned = self._clean_response_text(merged_response)

            result = {
                "answer": cleaned,
                "query": query,
                "context_chunks": context_chunks if include_sources else [],
                "num_chunks_used": total_chunks_used,
                "split_responses": True,
                "num_batches": len(batches),
            }

            return result

        except Exception as e:
            logger.error(f"Failed to generate split RAG response: {e}")
            raise

    def _clean_response_text(self, text: str) -> str:
        """Clean response text by removing HTML tags and LaTeX delimiters."""
        import re

        if not isinstance(text, str):
            return text

        def _process_line(line: str) -> str:
            # If line looks like a table row, replace <br> with a space
            if "|" in line:
                line = re.sub(r"(?i)<br\s*/?>", " ", line)
                line = re.sub(r"(?i)<p\s*/?>", "", line)
                line = re.sub(r"(?i)</p>", "", line)
            else:
                line = re.sub(r"(?i)<br\s*/?>", "\n", line)
                line = re.sub(r"(?i)<p\s*/?>", "\n", line)
                line = re.sub(r"(?i)</p>", "\n", line)
            return line

        # Apply line-wise processing to preserve table-row behavior
        lines = text.splitlines()
        processed_lines = [_process_line(ln) for ln in lines]
        text = "\n".join(processed_lines)

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip LaTeX delimiters but keep content
        text = re.sub(r"\$\$(.*?)\$\$", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(r"\$(.*?)\$", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(r"\\\\\((.*?)\\\\\)", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(r"\\\\\[(.*?)\\\\\]", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(
            r"\\begin\{([a-zA-Z*]+)\}(.*?)\\end\{\1\}",
            lambda m: m.group(2),
            text,
            flags=re.S,
        )

        return text.strip()

    def _maybe_continue_response(
        self, response: str, system_message: Optional[str], last_max_tokens: int
    ) -> str:
        """
        Heuristic check for truncated responses. If the response appears to be cut off
        (near the max token budget or ending mid-sentence), request a short continuation
        from the model and append it.
        """
        try:
            if not response or not isinstance(response, str):
                return response

            from core.token_manager import token_manager

            resp_tokens = token_manager.count_tokens(response)

            # Heuristics: if response used almost all tokens or ends without terminal punctuation
            last_char = response.strip()[-1] if response.strip() else ""
            ends_with_punct = last_char in ".!?"

            near_limit = resp_tokens >= max(1, last_max_tokens - 8)
            looks_cut = response.strip().endswith("...") or (
                last_char.isalpha() and not ends_with_punct
            )

            if not (near_limit or looks_cut):
                return response

            # Ask the model to continue/finish the response
            cont_prompt = (
                "Continue the previous answer, finishing the last sentence and completing any missing content."
                " Provide only the continuation text (no reiteration of the already provided text)."
            )

            cont_max = min(512, max(128, last_max_tokens // 4))

            continuation = self.generate_response(
                prompt=cont_prompt,
                system_message=system_message,
                temperature=0.1,
                max_tokens=cont_max,
            )

            if not continuation:
                return response

            # Merge while avoiding simple duplication: trim overlapping prefix
            cont = continuation.strip()
            combined = response.rstrip()

            # Remove overlap: if combined endswith start of cont, skip the overlap
            max_overlap = min(60, len(cont))
            for k in range(max_overlap, 0, -1):
                if combined.endswith(cont[:k]):
                    combined = combined + cont[k:]
                    break
            else:
                # No overlap found
                combined = combined + "\n\n" + cont

            return combined

        except Exception as e:
            logger.warning(f"Continuation attempt failed: {e}")
            return response

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
