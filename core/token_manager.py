"""
Token management for LLM requests to handle context length limits.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None

from config.settings import settings

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages token counting and context splitting for LLM requests."""

    # Model context sizes (in tokens) - conservative estimates
    MODEL_CONTEXT_SIZES = {
        # OpenAI models
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-1106-preview": 128000,
        "gpt-oss-120b": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-3.5-turbo-0125": 16385,
        # Common Ollama models (approximate)
        "llama2": 4096,
        "llama2:7b": 4096,
        "llama2:13b": 4096,
        "llama2:70b": 4096,
        "codellama": 16384,
        "mistral": 8192,
        "mixtral": 32768,
        "gemma": 8192,
        "qwen": 262000,
        "dolphin-mixtral": 32768,
        "llama3": 8192,
        "llama3:8b": 8192,
        "llama3:70b": 8192,
        "tinyllama-64": 64000,
        # Default fallback
        "default": 4096,
    }

    def __init__(self):
        """Initialize the token manager."""
        self.provider = getattr(settings, "llm_provider", "openai").lower()
        self.model = self._get_model_name()
        self.context_size = self._get_context_size()
        self.encoding = self._get_encoding()

        # Reserve tokens for system message, response, and safety margin
        self.reserved_tokens = 1000
        # Safety margin specifically reserved for output (to avoid truncation)
        self.output_safety_tokens = 128
        # Maximum tokens we should use for chunked context (input only)
        self.max_chunk_tokens = self.context_size - self.reserved_tokens

        logger.info(
            f"TokenManager initialized: model={self.model}, context_size={self.context_size}, max_chunk_tokens={self.max_chunk_tokens}"
        )

    def _get_model_name(self) -> str:
        """Get the current model name."""
        if self.provider == "openai":
            return getattr(settings, "openai_model", "gpt-3.5-turbo")
        else:  # ollama
            return getattr(settings, "ollama_model", "llama2")

    def _get_context_size(self) -> int:
        """Get the context size for the current model."""
        model_key = self.model.lower()

        # Try exact match first
        if model_key in self.MODEL_CONTEXT_SIZES:
            return self.MODEL_CONTEXT_SIZES[model_key]

        # Try partial matches for OpenAI models
        for known_model, context_size in self.MODEL_CONTEXT_SIZES.items():
            if known_model in model_key or model_key.startswith(known_model):
                return context_size

        # Default fallback
        logger.warning(
            f"Unknown model '{self.model}', using default context size of {self.MODEL_CONTEXT_SIZES['default']} tokens"
        )
        return self.MODEL_CONTEXT_SIZES["default"]

    def _get_encoding(self):
        """Get the appropriate encoding for token counting."""
        if not HAS_TIKTOKEN:
            logger.warning(
                "tiktoken not available, using approximation for token counting"
            )
            return None

        try:
            # Try to get encoding for specific model
            if self.provider == "openai":
                if "gpt-4" in self.model.lower():
                    return tiktoken.encoding_for_model("gpt-4")  # type: ignore
                elif "gpt-3.5" in self.model.lower():
                    return tiktoken.encoding_for_model("gpt-3.5-turbo")  # type: ignore
                else:
                    # Default to cl100k_base for most OpenAI models
                    return tiktoken.get_encoding("cl100k_base")  # type: ignore
            else:
                # For Ollama and other providers, use a general encoding
                return tiktoken.get_encoding("cl100k_base")  # type: ignore
        except Exception as e:
            logger.warning(
                f"Could not get tiktoken encoding for model {self.model}: {e}"
            )
            return tiktoken.get_encoding("cl100k_base") if HAS_TIKTOKEN else None  # type: ignore

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0

        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed, using approximation: {e}")

        # Fallback approximation: ~4 characters per token
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages (OpenAI format)."""
        total_tokens = 0

        for message in messages:
            # Count tokens for role and content
            role_tokens = self.count_tokens(message.get("role", ""))
            content_tokens = self.count_tokens(message.get("content", ""))

            # Add some overhead for message formatting
            total_tokens += (
                role_tokens + content_tokens + 4
            )  # 4 tokens per message overhead

        # Add some tokens for the overall conversation structure
        total_tokens += 3

        return total_tokens

    def available_output_tokens_for_messages(
        self, messages: List[Dict[str, str]]
    ) -> int:
        """Compute available tokens for the model's output given the message list."""
        # tokens used by input messages
        used = self.count_message_tokens(messages)
        available = self.context_size - used - self.output_safety_tokens
        # Minimum of 32 tokens to avoid tiny responses
        return max(32, available)

    def available_output_tokens_for_prompt(
        self, prompt: str, system_message: Optional[str] = None
    ) -> int:
        """Convenience wrapper: build message list and compute available output tokens."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return self.available_output_tokens_for_messages(messages)

    async def get_model_context_size(self, llm_manager) -> int:
        """
        Try to ask the model what its context size is if not known.
        This is a fallback for unknown models.
        """
        if self.model.lower() in self.MODEL_CONTEXT_SIZES:
            return self.context_size

        try:
            query = "What is your maximum context length in tokens? Please respond with just the number."

            # Use a very simple request to avoid token issues
            response = llm_manager.generate_response(
                prompt=query, temperature=0.0, max_tokens=50
            )

            # Extract number from response
            numbers = re.findall(r"\b(\d+)\b", response)
            if numbers:
                context_size = int(numbers[0])
                # Sanity check - should be between 1K and 1M tokens
                if 1000 <= context_size <= 1000000:
                    logger.info(f"Model reported context size: {context_size} tokens")
                    self.context_size = context_size
                    self.max_chunk_tokens = context_size - self.reserved_tokens
                    return context_size

            logger.warning(
                f"Could not determine context size from model response: {response}"
            )

        except Exception as e:
            logger.warning(f"Failed to query model for context size: {e}")

        return self.context_size

    def split_context_chunks(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_message: Optional[str] = None,
    ) -> List[Tuple[str, List[Dict[str, Any]], int]]:
        """
        Split context chunks into batches that fit within token limits.

        Args:
            query: User query
            context_chunks: List of context chunks
            system_message: Optional system message

        Returns:
            List of tuples: (query, chunk_batch, estimated_tokens)
        """
        # Count tokens for fixed parts
        query_tokens = self.count_tokens(query)
        system_tokens = self.count_tokens(system_message) if system_message else 0

        # Calculate available tokens for context
        available_tokens = self.max_chunk_tokens - query_tokens - system_tokens

        if available_tokens <= 0:
            logger.error(
                f"Query and system message too long: {query_tokens + system_tokens} tokens exceed limit"
            )
            return [(query, [], query_tokens + system_tokens)]

        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in context_chunks:
            chunk_content = chunk.get("content", "")
            chunk_tokens = self.count_tokens(chunk_content)

            # If single chunk is too large, truncate it
            if chunk_tokens > available_tokens:
                logger.warning(f"Chunk too large ({chunk_tokens} tokens), truncating")
                truncated_content = self._truncate_text(chunk_content, available_tokens)
                truncated_chunk = chunk.copy()
                truncated_chunk["content"] = truncated_content
                truncated_chunk["truncated"] = True

                # Create batch with just this truncated chunk
                if current_batch:
                    batches.append(
                        (
                            query,
                            current_batch,
                            current_tokens + query_tokens + system_tokens,
                        )
                    )
                    current_batch = []
                    current_tokens = 0

                batches.append(
                    (
                        query,
                        [truncated_chunk],
                        available_tokens + query_tokens + system_tokens,
                    )
                )
                continue

            # If adding this chunk would exceed limit, start new batch
            if current_tokens + chunk_tokens > available_tokens:
                if current_batch:
                    batches.append(
                        (
                            query,
                            current_batch,
                            current_tokens + query_tokens + system_tokens,
                        )
                    )
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_tokens += chunk_tokens

        # Add final batch if any
        if current_batch:
            batches.append(
                (query, current_batch, current_tokens + query_tokens + system_tokens)
            )

        # If no batches (no chunks), create empty batch
        if not batches:
            batches.append((query, [], query_tokens + system_tokens))

        logger.info(f"Split context into {len(batches)} batches")
        return batches

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return text

        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Binary search to find the right length
        left, right = 0, len(text)
        best_text = ""

        while left <= right:
            mid = (left + right) // 2
            candidate = text[:mid]
            tokens = self.count_tokens(candidate)

            if tokens <= max_tokens:
                best_text = candidate
                left = mid + 1
            else:
                right = mid - 1

        # Try to end at a word boundary
        if best_text and not best_text.endswith(" "):
            last_space = best_text.rfind(" ")
            if last_space > len(best_text) * 0.8:  # Only if we don't lose too much
                best_text = best_text[:last_space]

        return best_text + "..." if best_text != text else best_text

    def merge_responses(
        self, responses: List[str], query: str = "", use_llm_merge: bool = True
    ) -> str:
        """
        Merge multiple LLM responses into a single coherent response.
        Handles deduplication and removes part indicators for seamless user experience.

        Args:
            responses: List of response strings from different batches
            query: Original user query for context in LLM merging
            use_llm_merge: Whether to use LLM for intelligent merging (recommended)
        """
        if not responses:
            return ""

        if len(responses) == 1:
            return responses[0].strip()

        # Clean responses first
        cleaned_responses = [
            response.strip() for response in responses if response.strip()
        ]

        if not cleaned_responses:
            return ""

        if len(cleaned_responses) == 1:
            return cleaned_responses[0]

        # Try LLM-based intelligent merging first
        if use_llm_merge:
            try:
                merged = self._llm_merge_responses(cleaned_responses, query)
                if merged and len(merged.strip()) > 50:  # Sanity check
                    return merged
            except Exception as e:
                logger.warning(f"LLM merge failed, falling back to simple merge: {e}")

        # Fallback to simple concatenation without part indicators
        return self._simple_merge_responses(cleaned_responses)

    def _llm_merge_responses(self, responses: List[str], query: str = "") -> str:
        """Use LLM to intelligently merge and deduplicate responses."""
        # Import here to avoid circular imports
        from core.llm import llm_manager

        # Prepare the responses for merging
        numbered_responses = []
        for i, response in enumerate(responses, 1):
            numbered_responses.append(f"Response {i}:\n{response}")

        combined_responses = "\n\n" + "=" * 50 + "\n\n".join(numbered_responses)

        system_message = """You are an expert at merging and consolidating multiple related responses into a single, coherent answer.

Your task:
1. Merge the provided responses into ONE comprehensive answer
2. Remove ANY duplicate information or repetitive content
3. Organize the information logically and coherently
4. Preserve all unique facts, insights, and details
5. Use proper markdown formatting
6. Do NOT include any part numbers, separators, or references to "Response 1/2/3" etc.
7. Write as if this was a single, original response

Guidelines:
- If responses contradict each other, include both perspectives clearly
- Combine related points into unified sections
- Remove redundant phrases and duplicate facts
- Maintain the original tone and style
- Ensure the final answer directly addresses the user's query"""

        query_context = f"\n\nOriginal user query: {query}" if query else ""

        merge_prompt = f"""Please merge these multiple responses into a single, comprehensive answer that removes all duplication and flows naturally:

{combined_responses}{query_context}

Provide your merged response below (no explanations, just the final answer):"""

        try:
            merged_response = llm_manager.generate_response(
                prompt=merge_prompt,
                system_message=system_message,
                temperature=0.2,  # Low temperature for consistency
                max_tokens=2000,  # Generous limit for merged response
            )

            return merged_response.strip()

        except Exception as e:
            logger.error(f"Failed to merge responses with LLM: {e}")
            raise

    def _simple_merge_responses(self, responses: List[str]) -> str:
        """Simple merge without LLM - just concatenate and clean up."""
        # Join responses with double line breaks
        merged = "\n\n".join(responses)

        # Basic deduplication - remove exact duplicate paragraphs
        paragraphs = merged.split("\n\n")
        unique_paragraphs = []
        seen_paragraphs = set()

        for para in paragraphs:
            para_clean = para.strip().lower()
            if para_clean and para_clean not in seen_paragraphs:
                unique_paragraphs.append(para.strip())
                seen_paragraphs.add(para_clean)

        merged = "\n\n".join(unique_paragraphs)

        # Clean up markdown formatting
        merged = self._clean_merged_markdown(merged)

        return merged

    def _clean_merged_markdown(self, text: str) -> str:
        """Clean up markdown formatting in merged responses."""
        if not text:
            return text

        # Remove excessive whitespace but preserve paragraph breaks
        lines = text.split("\n")
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            line_stripped = line.strip()
            is_empty = not line_stripped

            if is_empty:
                if not prev_empty:  # Keep only one empty line
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(line_stripped)
                prev_empty = False

        # Join back and handle specific markdown patterns
        result = "\n".join(cleaned_lines)

        # Fix common markdown issues
        result = re.sub(r"\n{3,}", "\n\n", result)  # Max 2 consecutive newlines
        result = re.sub(
            r"(\*\*[^*]+\*\*)\n{2,}(\*\*[^*]+\*\*)", r"\1\n\n\2", result
        )  # Headers spacing

        return result.strip()

    def needs_splitting(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_message: Optional[str] = None,
    ) -> bool:
        """Check if the request needs to be split due to token limits."""
        total_tokens = self.count_tokens(query)

        if system_message:
            total_tokens += self.count_tokens(system_message)

        for chunk in context_chunks:
            total_tokens += self.count_tokens(chunk.get("content", ""))

        # Add some overhead for message formatting
        total_tokens += len(context_chunks) * 10

        return total_tokens > self.max_chunk_tokens


# Global token manager instance
token_manager = TokenManager()
