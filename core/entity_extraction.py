"""
Entity and relationship extraction using LLM for GraphRAG pipeline.
"""

import logging
import re
from typing import Any, Dict, List, Tuple, Optional
import asyncio
from dataclasses import dataclass

from core.llm import llm_manager
from config.settings import settings

logger = logging.getLogger(__name__)


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
        "MONEY"
    ]

    def __init__(self, entity_types: Optional[List[str]] = None):
        """Initialize entity extractor."""
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
    
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

    def _parse_extraction_response(self, response: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
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
                
                entity = Entity(
                    name=name,
                    type=entity_type,
                    description=description,
                    importance_score=min(max(importance, 0.0), 1.0),
                    source_chunks=[chunk_id]
                )
                entities.append(entity)
            
            # Parse relationships
            relationship_pattern = r"- Source: ([^|]+) \| Target: ([^|]+) \| Description: ([^|]+) \| Strength: ([\d.]+)"
            for match in re.finditer(relationship_pattern, relationships_section):
                source = match.group(1).strip()
                target = match.group(2).strip()
                description = match.group(3).strip()
                strength = float(match.group(4))
                
                relationship = Relationship(
                    source_entity=source,
                    target_entity=target,
                    description=description,
                    strength=min(max(strength, 0.0), 1.0),
                    source_chunks=[chunk_id]
                )
                relationships.append(relationship)
                
        except Exception as e:
            logger.error(f"Error parsing extraction response for chunk {chunk_id}: {e}")
            logger.debug(f"Response was: {response}")
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}")
        return entities, relationships
    
    async def extract_from_chunk(self, text: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single text chunk."""
        try:
            prompt = self._get_extraction_prompt(text)

            # Offload synchronous/blocking LLM call to a thread executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm_manager.generate_response(
                    prompt=prompt,
                    max_tokens=4000,
                    temperature=0.1
                ),
            )

            return self._parse_extraction_response(response, chunk_id)

        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk_id}: {e}")
            return [], []
    
    async def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[Dict[str, Entity], Dict[str, List[Relationship]]]:
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
                    return await self.extract_from_chunk(chunk["content"], chunk["chunk_id"])
                except Exception as e:
                    logger.error(f"Extraction failed for chunk {chunk.get('chunk_id')}: {e}")
                    return [], []

        # Schedule tasks with semaphore control
        extraction_tasks = [asyncio.create_task(_sem_extract(chunk)) for chunk in chunks]

        results = []
        for coro in asyncio.as_completed(extraction_tasks):
            try:
                res = await coro
                results.append(res)
            except Exception as e:
                logger.error(f"Error in extraction task: {e}")
        
        # Consolidate entities and relationships
        all_entities = {}
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
            
            # Consolidate entities (merge duplicates by name)
            for entity in entities:
                entity_key = entity.name.upper().strip()
                if entity_key in all_entities:
                    # Merge with existing entity
                    existing = all_entities[entity_key]
                    existing.source_chunks.extend(entity.source_chunks)
                    # Use more detailed description if available
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description
                    # Average importance scores
                    existing.importance_score = (existing.importance_score + entity.importance_score) / 2
                else:
                    all_entities[entity_key] = entity
            
            all_relationships.extend(relationships)
        
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
        
        logger.info(f"Consolidated to {len(all_entities)} entities and {len(relationships_by_pair)} relationship pairs")
        return all_entities, relationships_by_pair


# Global instance
entity_extractor = EntityExtractor()
