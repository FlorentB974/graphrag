"""
Entity and relationship extraction using LLM for GraphRAG pipeline.
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from core.llm import llm_manager

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Decorator for retrying LLM calls with exponential backoff on rate limiting errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


@dataclass
class Entity:
    """Represents an extracted entity."""

    name: str
    type: str
    description: str
    importance_score: float = 0.5
    source_chunks: Optional[List[str]] = None

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []


@dataclass
class Relationship:
    """Represents a relationship between two entities."""

    source_entity: str
    target_entity: str
    description: str
    strength: float = 0.5
    source_chunks: Optional[List[str]] = None

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    # Default entity types based on nano-graphrag
    DEFAULT_ENTITY_TYPES = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "EVENT",
        "CONCEPT",
        "TECHNOLOGY",
        "PRODUCT",
        "DOCUMENT",
        "DATE",
        "MONEY",
    ]

    # Patterns for low-value entities that should be filtered out
    LOW_VALUE_PATTERNS = [
        # Common articles, prepositions, conjunctions
        r"^(?:the|and|or|but|with|from|for|at|by|on|in|to|of|a|an)$",
        # Common pronouns and demonstratives
        r"^(?:this|that|these|those|here|there|where|when|what|who|how|why)$",
        # Generic business terms
        r"^(?:company|organization|group|team|department|division|system|process|method|approach|way|means)$",
        # Generic data terms
        r"^(?:data|information|content|text|document|report|file|item|thing|stuff)$",
        # Common adjectives
        r"^(?:new|old|first|last|next|previous|current|recent|good|bad|big|small|high|low|major|minor)$",
        # Just numbers or single characters
        r"^\d{1,3}$",
        r"^[a-zA-Z]$",
        # Very short meaningless strings
        r"^.{1,2}$",
        # Common file extensions or codes
        r"^\.[a-z]{2,4}$",
        r"^[A-Z]{1,3}\d*$",
    ]

    # Entity type normalization mapping
    ENTITY_TYPE_MAPPING = {
        # Normalize inconsistent typing
        "PERSON (ROLE)": "PERSON",
        "PERSON (COLLECTIVE)": "PERSON",
        "PERSON (LEGAL ROLE)": "PERSON",
        "PERSON/CONCEPT": "PERSON",
        "PERSON/ORGANIZATION": "PERSON",
        "PERSON (OR ORGANIZATION)": "PERSON",
        "**PERSON**": "PERSON",
        "**ORGANIZATION**": "ORGANIZATION",
        "ORGANIZATION/TECHNOLOGY": "ORGANIZATION",
        "**LOCATION**": "LOCATION",
        "LOCATION/CONCEPT": "LOCATION",
        "**CONCEPT**": "CONCEPT",
        "CONCEPT / PRODUCT": "CONCEPT",
        "TECHNOLOGY / CONCEPT": "CONCEPT",
        "CONCEPT/DOCUMENT": "CONCEPT",
        "CONCEPT (LEGISLATION)": "CONCEPT",
        "CONCEPT (LAND PARCEL)": "CONCEPT",
        "CONCEPT (PROPERTY INTEREST)": "CONCEPT",
        "CONCEPT (OWNERSHIP TYPE)": "CONCEPT",
        "CONCEPT (CLAUSE NUMBER)": "CONCEPT",
        "MATERIAL (CLASSIFIED UNDER CONCEPT)": "CONCEPT",
        "SECTION (TREATED AS CONCEPT)": "CONCEPT",
        "ACTION (TREATED AS CONCEPT FOR EXTRACTION)": "CONCEPT",
        "TIME (TREATED AS CONCEPT)": "CONCEPT",
        "PROPERTY (TREATED AS CONCEPT)": "CONCEPT",
        "**DOCUMENT**": "DOCUMENT",
        "DOCUMENT (TAX/IDENTIFIER)": "DOCUMENT",
        "DOCUMENT (AUSTRALIAN BUSINESS NUMBER)": "DOCUMENT",
        "DOCUMENT (CONTACT INFORMATION)": "DOCUMENT",
        "PRODUCT (WEBPAGE)": "PRODUCT",
        "PRODUCT/TECHNOLOGY": "PRODUCT",
        "TRADEMARK (TREATED AS PRODUCT)": "PRODUCT",
        "PHONE NUMBER (TECHNOLOGY)": "TECHNOLOGY",
        "CONTACT (TREATED AS TECHNOLOGY FOR PHONE CONTACT)": "TECHNOLOGY",
        "EMAIL (TREATED AS A PRODUCT/TECHNOLOGY IDENTIFIER)": "TECHNOLOGY",
        "EMAIL (PRODUCT/TECHNOLOGY IDENTIFIER)": "TECHNOLOGY",
        "WEBSITE (TECHNOLOGY)": "TECHNOLOGY",
        "EMAIL (TECHNOLOGY)": "TECHNOLOGY",
        "CONTACT (TREATED AS TECHNOLOGY FOR THIS LIST)": "TECHNOLOGY",
        "CONTACT (TREATED AS TECHNOLOGY FOR EMAIL CONTACT)": "TECHNOLOGY",
        "**MONEY**": "MONEY",
        "MONEY (CONCEPT)": "MONEY",
        "**EVENT**": "EVENT",
        "SECTION (EVENT)": "EVENT",
        "DATE (DURATION)": "DATE",
        "**NUMBER**": "CONCEPT",  # Numbers as concepts
        # Consolidate contact information
        "CONTACT": "TECHNOLOGY",
        "CONTACT INFORMATION": "TECHNOLOGY",
        "CONTACT INFORMATION (ASSOCIATED WITH ORGANIZATION)": "TECHNOLOGY",
        "CONTACT (TREATED AS A TELEPHONE NUMBER)": "TECHNOLOGY",
        "CONTACT (PHONE NUMBER)": "TECHNOLOGY",
        "CONTACT (PHONE)": "TECHNOLOGY",
        # Consolidate materials
        "MATERIAL": "CONCEPT",
        # Consolidate sections
        "SECTION": "CONCEPT",
        # Other consolidations
        "SOFTWARE": "TECHNOLOGY",
        "PROGRAM": "TECHNOLOGY",
        "SERVICE": "PRODUCT",
        "PROJECT": "CONCEPT",
        "EDUCATION": "CONCEPT",
        "LANGUAGE": "CONCEPT",
    }

    def __init__(self, entity_types: Optional[List[str]] = None):
        """Initialize entity extractor."""
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name to reduce duplicates."""
        # Remove extra whitespace and normalize case
        normalized = re.sub(r"\s+", " ", name.strip())

        # Remove unnecessary punctuation but keep meaningful ones
        normalized = re.sub(r"[^\w\s\-\.\(\)\/]", "", normalized)

        # Normalize common variations
        normalized = re.sub(
            r"\b(?:sub[\-\s]?floor)\b", "subfloor", normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"\b(?:sub[\-\s]?structure)\b",
            "substructure",
            normalized,
            flags=re.IGNORECASE,
        )

        # Remove redundant parenthetical content
        normalized = re.sub(r"\s*\([^)]*\)\s*", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type using mapping."""
        # Clean up the type string
        cleaned_type = entity_type.strip().upper()

        # Apply mapping if available
        if cleaned_type in self.ENTITY_TYPE_MAPPING:
            return self.ENTITY_TYPE_MAPPING[cleaned_type]

        # Default normalization for unmapped types
        # Remove excessive descriptors
        if "(" in cleaned_type and ")" in cleaned_type:
            base_type = cleaned_type.split("(")[0].strip()
            if base_type in self.DEFAULT_ENTITY_TYPES:
                return base_type

        # Remove prefixed modifiers like "**TYPE**"
        cleaned_type = re.sub(r"^\*+([A-Z]+)\*+$", r"\1", cleaned_type)

        # Default to CONCEPT for unclear types
        if cleaned_type not in self.DEFAULT_ENTITY_TYPES:
            return "CONCEPT"

        return cleaned_type

    def _is_low_value_entity(
        self, name: str, entity_type: str, importance: float
    ) -> bool:
        """Check if entity is low value and should be filtered out."""
        # Filter by importance threshold
        if importance < 0.3:
            return True

        # Filter by name patterns
        name_lower = name.lower().strip()
        for pattern in self.LOW_VALUE_PATTERNS:
            if re.match(pattern, name_lower, re.IGNORECASE):
                return True

        # Filter very generic concepts
        if entity_type == "CONCEPT" and importance < 0.6:
            generic_patterns = [
                r"^(?:management|system|program|process|method|approach|solution)$",
                r"^(?:inspection|treatment|damage|condition|presence|lack)$",
                r"^(?:area|areas|location|locations|structure|structures)$",
            ]
            for pattern in generic_patterns:
                if re.match(pattern, name_lower, re.IGNORECASE):
                    return True

        return False

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on normalized names."""
        seen_entities = {}
        deduplicated = []

        for entity in entities:
            normalized_name = self._normalize_entity_name(entity.name)
            normalized_type = self._normalize_entity_type(entity.type)

            # Create a key for deduplication
            key = (normalized_name.lower(), normalized_type)

            if key not in seen_entities:
                # Create new entity with normalized values
                normalized_entity = Entity(
                    name=normalized_name,
                    type=normalized_type,
                    description=entity.description,
                    importance_score=entity.importance_score,
                    source_chunks=(
                        entity.source_chunks.copy() if entity.source_chunks else []
                    ),
                )
                seen_entities[key] = normalized_entity
                deduplicated.append(normalized_entity)
            else:
                # Merge with existing entity
                existing = seen_entities[key]
                if entity.source_chunks:
                    existing.source_chunks.extend(entity.source_chunks)
                # Use better description if available
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
                # Average importance scores
                existing.importance_score = (
                    existing.importance_score + entity.importance_score
                ) / 2

        return deduplicated

    def _get_extraction_prompt(self, text: str) -> str:
        """Generate prompt for entity and relationship extraction."""
        entity_types_str = ", ".join(self.entity_types)

        return f"""You are an expert at extracting entities and relationships from text.

**Task**: Extract all relevant entities and relationships from the given text.

**Entity Types**: Focus on these types: {entity_types_str}

**Instructions**:
1. Extract entities with: name, type, description, importance (0.0-1.0)
2. Extract relationships with: source entity, target entity, description, strength (0.0-1.0)
3. Use exact entity names from the text
4. Provide detailed descriptions
5. Rate importance/strength based on context significance

**Output Format**:
ENTITIES:
- Name: [entity_name] | Type: [entity_type] | Description: [description] | Importance: [0.0-1.0]

RELATIONSHIPS:
- Source: [source_entity] | Target: [target_entity] | Description: [description] | Strength: [0.0-1.0]

**Text to analyze**:
{text}

**Output**:"""

    def _parse_extraction_response(
        self, response: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response to extract entities and relationships."""
        entities = []
        relationships = []

        try:
            # Split response into entities and relationships sections
            # Handle different formats: "RELATIONSHIPS:" or "**RELATIONSHIPS**"
            if "**RELATIONSHIPS**" in response:
                sections = response.split("**RELATIONSHIPS**")
                entities_section = sections[0].replace("**ENTITIES**", "").strip()
                relationships_section = sections[1].strip() if len(sections) > 1 else ""
            else:
                sections = response.split("RELATIONSHIPS:")
                entities_section = sections[0].replace("ENTITIES:", "").strip()
                relationships_section = sections[1].strip() if len(sections) > 1 else ""

            # Parse entities
            entity_pattern = r"- Name: ([^|]+) \| Type: ([^|]+) \| Description: ([^|]+) \| Importance: ([\d.]+)"
            for match in re.finditer(entity_pattern, entities_section):
                name = match.group(1).strip()
                entity_type = match.group(2).strip().upper()
                description = match.group(3).strip()
                importance = float(match.group(4))

                # Apply normalization
                normalized_name = self._normalize_entity_name(name)
                normalized_type = self._normalize_entity_type(entity_type)

                # Filter low-value entities
                if self._is_low_value_entity(
                    normalized_name, normalized_type, importance
                ):
                    continue

                entity = Entity(
                    name=normalized_name,
                    type=normalized_type,
                    description=description,
                    importance_score=min(max(importance, 0.0), 1.0),
                    source_chunks=[chunk_id],
                )
                entities.append(entity)

            # Parse relationships
            relationship_pattern = r"- Source: ([^|]+) \| Target: ([^|]+) \| Description: ([^|]+) \| Strength: ([\d.]+)"
            for match in re.finditer(relationship_pattern, relationships_section):
                source = match.group(1).strip()
                target = match.group(2).strip()
                description = match.group(3).strip()
                strength = float(match.group(4))

                # Normalize entity names in relationships
                normalized_source = self._normalize_entity_name(source)
                normalized_target = self._normalize_entity_name(target)

                relationship = Relationship(
                    source_entity=normalized_source,
                    target_entity=normalized_target,
                    description=description,
                    strength=min(max(strength, 0.0), 1.0),
                    source_chunks=[chunk_id],
                )
                relationships.append(relationship)

        except Exception as e:
            logger.error(f"Error parsing extraction response for chunk {chunk_id}: {e}")
            logger.debug(f"Response was: {response}")

        # Apply deduplication
        entities = self._deduplicate_entities(entities)

        logger.info(
            f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}"
        )
        return entities, relationships

    async def extract_from_chunk(
        self, text: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single text chunk with retry logic."""

        @retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
        def _generate_response_with_retry(prompt):
            return llm_manager.generate_response(
                prompt=prompt, max_tokens=4000, temperature=0.1
            )

        try:
            prompt = self._get_extraction_prompt(text)

            # Offload synchronous/blocking LLM call to a thread executor with retry
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: _generate_response_with_retry(prompt)
            )

            return self._parse_extraction_response(response, chunk_id)

        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk_id}: {e}")
            return [], []

    async def extract_from_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Entity], Dict[str, List[Relationship]]]:
        """
        Extract entities and relationships from multiple chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'content' keys

        Returns:
            Tuple of (consolidated_entities, relationships_by_entity_pair)
        """
        logger.info(f"Starting entity extraction from {len(chunks)} chunks")

        # Concurrency limit from settings (use embedding_concurrency to match pattern)
        concurrency = getattr(settings, "llm_concurrency")
        sem = asyncio.Semaphore(concurrency)

        async def _sem_extract(chunk):
            async with sem:
                try:
                    # Add small delay to prevent API flooding
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                    return await self.extract_from_chunk(
                        chunk["content"], chunk["chunk_id"]
                    )
                except Exception as e:
                    logger.error(
                        f"Extraction failed for chunk {chunk.get('chunk_id')}: {e}"
                    )
                    return [], []

        # Schedule tasks with semaphore control
        extraction_tasks = [
            asyncio.create_task(_sem_extract(chunk)) for chunk in chunks
        ]

        results = []
        for coro in asyncio.as_completed(extraction_tasks):
            try:
                res = await coro
                results.append(res)
            except Exception as e:
                logger.error(f"Error in extraction task: {e}")

        # Consolidate entities and relationships
        all_entities_list = []
        all_relationships = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Extraction failed for chunk {i}: {result}")
                continue

            if isinstance(result, tuple) and len(result) == 2:
                entities, relationships = result
            else:
                logger.error(f"Unexpected result format for chunk {i}: {result}")
                continue

            # Collect all entities for global deduplication
            all_entities_list.extend(entities)
            all_relationships.extend(relationships)

        # Apply global deduplication across all chunks
        deduplicated_entities = self._deduplicate_entities(all_entities_list)

        # Convert to dictionary for compatibility
        all_entities = {
            entity.name.upper().strip(): entity for entity in deduplicated_entities
        }

        # Group relationships by entity pair
        relationships_by_pair = {}
        for rel in all_relationships:
            source_key = rel.source_entity.upper().strip()
            target_key = rel.target_entity.upper().strip()

            # Only keep relationships where both entities exist
            if source_key in all_entities and target_key in all_entities:
                # Create consistent key regardless of direction
                pair_key = tuple(sorted([source_key, target_key]))
                if pair_key not in relationships_by_pair:
                    relationships_by_pair[pair_key] = []
                relationships_by_pair[pair_key].append(rel)

        logger.info(
            f"Consolidated to {len(all_entities)} entities and {len(relationships_by_pair)} relationship pairs"
        )
        return all_entities, relationships_by_pair


# Global instance
entity_extractor = EntityExtractor()
