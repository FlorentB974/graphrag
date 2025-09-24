# Hybrid Entity-Chunk GraphRAG Approach

## Overview

This implementation provides a **hybrid approach** that combines both chunk-based and entity-based retrieval methods, inspired by Microsoft's GraphRAG but maintaining compatibility with existing chunk-based workflows.

## Architecture Components

### 1. Entity Extraction (`core/entity_extraction.py`)
- **LLM-powered extraction**: Uses configurable models (GPT-4o-mini, etc.) to extract entities and relationships
- **Structured output**: Returns `Entity` and `Relationship` dataclasses with importance scores
- **Concurrent processing**: Configurable concurrency to balance speed vs cost
- **Filtering**: Importance and strength thresholds to reduce noise

### 2. Enhanced Graph Storage (`core/graph_db.py`)
- **Dual node types**: Supports both `Chunk` and `Entity` nodes
- **Rich relationships**: Entity relationships with type, strength, and source chunk tracking  
- **Vector search**: Both chunk embeddings and entity embeddings for similarity search
- **Graph traversal**: Neighborhood exploration for expanded context

### 3. Multiple Retrieval Modes (`rag/enhanced_retriever.py`)
- **Chunk-only**: Traditional chunk-based similarity search (backward compatible)
- **Entity-only**: Entity-centric search like original GraphRAG
- **Hybrid**: Weighted combination of chunk and entity results
- **Graph expansion**: Uses entity relationships for context enrichment

### 4. Configuration Management (`config/settings.py`)
- **Feature toggles**: Enable/disable entity extraction per use case
- **Cost control**: Concurrency limits and model selection for budget management
- **Retrieval tuning**: Configurable weights and thresholds

## Key Benefits

### 1. **Backward Compatibility**
```python
# Existing code continues to work unchanged
retriever.retrieve(query="What is AI?", mode=RetrievalMode.CHUNK_ONLY)
```

### 2. **Progressive Enhancement**
```python
# Enable entity extraction when ready
ENABLE_ENTITY_EXTRACTION=true
retriever.retrieve(query="What is AI?", mode=RetrievalMode.HYBRID)
```

### 3. **Cost-Aware Processing**
- Entity extraction is optional and configurable
- Concurrent request limits prevent API rate limiting
- Model selection allows cost vs quality tradeoffs

### 4. **Rich Graph Visualization**
- Both chunks and entities visible in Neo4j browser
- Relationship types and strengths provide insights
- Source chunk tracking for provenance

## Configuration Examples

### Basic Setup (Chunk-only)
```bash
# Existing configuration - no entities
ENABLE_ENTITY_EXTRACTION=false
DEFAULT_RETRIEVAL_MODE=chunk_only
```

### Hybrid Setup (Recommended)
```bash
# Enable entity extraction with cost controls
ENABLE_ENTITY_EXTRACTION=true
ENTITY_EXTRACTION_MODEL=gpt-4o-mini
ENTITY_EXTRACTION_CONCURRENCY=2
DEFAULT_RETRIEVAL_MODE=hybrid
HYBRID_CHUNK_WEIGHT=0.6
```

### Entity-focused Setup
```bash
# GraphRAG-style entity-centric approach
ENABLE_ENTITY_EXTRACTION=true
ENTITY_EXTRACTION_MODEL=gpt-4o
DEFAULT_RETRIEVAL_MODE=entity_only
ENABLE_GRAPH_EXPANSION=true
```

## Usage Patterns

### 1. Document Ingestion with Entities
```python
# New documents automatically get entity extraction if enabled
from ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor()
await processor.process_document("research_paper.pdf")
# Creates both chunks AND entities based on configuration
```

### 2. Flexible Retrieval
```python
from rag.enhanced_retriever import EnhancedDocumentRetriever, RetrievalMode

retriever = EnhancedDocumentRetriever()

# Try different approaches for the same query
chunk_results = await retriever.retrieve(query, mode=RetrievalMode.CHUNK_ONLY)
entity_results = await retriever.retrieve(query, mode=RetrievalMode.ENTITY_ONLY)  
hybrid_results = await retriever.retrieve(query, mode=RetrievalMode.HYBRID)
```

### 3. Graph-Enhanced Context
```python
# Expand results using entity relationships
expanded = await retriever.retrieve_with_graph_expansion(
    query="AI investments",
    top_k=5
)
# Returns both direct matches and related entities
```

## Performance Considerations

### Entity Extraction Costs
- **GPT-4o-mini**: ~$0.15 per 1M input tokens
- **Typical document**: 5-10 entities per 1000-token chunk
- **Recommendation**: Start with importance threshold > 0.3

### Storage Requirements
- **Neo4j**: Additional entity nodes (~2-5x chunk count)
- **Relationships**: Entity-to-entity connections
- **Embeddings**: Optional entity embeddings for similarity search

### Query Performance
- **Chunk-only**: Fastest (existing performance)
- **Hybrid**: 20-50% slower but better results
- **Entity-only**: Variable based on graph density

## Migration Strategy

### Phase 1: Enable Entity Extraction
1. Set `ENABLE_ENTITY_EXTRACTION=true`
2. Re-ingest key documents to build entity graph
3. Test hybrid retrieval on sample queries

### Phase 2: Optimize Configuration
1. Tune importance/strength thresholds
2. Adjust hybrid weights based on results
3. Monitor extraction costs and quality

### Phase 3: Full Entity-Centric (Optional)
1. Switch to `DEFAULT_RETRIEVAL_MODE=entity_only`
2. Enable graph expansion features
3. Build community detection for global queries

## Example Results

**Query**: "What companies invested in OpenAI?"

**Chunk-only**: Returns chunks mentioning investments
```
1. "Microsoft invested $10 billion in OpenAI..." (score: 0.85)
2. "The partnership between Microsoft and OpenAI..." (score: 0.78)
```

**Entity-only**: Returns related entities
```  
1. Entity: Microsoft (COMPANY) - "Tech giant, OpenAI partner" (score: 0.92)
2. Entity: OpenAI (COMPANY) - "AI research company" (score: 0.88)
3. Entity: Sam Altman (PERSON) - "CEO of OpenAI" (score: 0.73)
```

**Hybrid**: Combines both approaches
```
1. Entity: Microsoft + supporting chunk (score: 0.90)
2. "Microsoft invested $10 billion..." chunk (score: 0.85)
3. Entity: OpenAI + relationship context (score: 0.83)
```

## Next Steps

1. **Entity-aware chunking**: Split documents along entity boundaries
2. **Community detection**: Group related entities for global queries
3. **Temporal relationships**: Track entity changes over time
4. **Multi-hop reasoning**: Chain entity relationships for complex queries

This hybrid approach provides the best of both worlds - the reliability of chunk-based retrieval with the semantic richness of entity-based graphs.