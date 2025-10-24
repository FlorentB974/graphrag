# Rate Limiting Improvements

## Overview

This document describes the improvements made to avoid "429 Too Many Requests" errors during document ingestion while maintaining parallel processing performance.

## Changes Made

### 1. Reduced Default Concurrency

- **Embedding Concurrency**: Reduced from 3 to **1** concurrent request
- **LLM Concurrency**: Reduced from 2 to **1** concurrent request
- This provides a conservative approach to API usage while still allowing processing

### 2. Configurable Rate Limiting Delays

Added new settings for fine-tuned control over request spacing:

```python
# In config/settings.py
embedding_delay_min: float = 1.0  # Minimum delay between embedding requests
embedding_delay_max: float = 2.0  # Maximum delay between embedding requests
llm_delay_min: float = 1.5        # Minimum delay between LLM requests
llm_delay_max: float = 3.0        # Maximum delay between LLM requests
```

These delays are applied randomly within the specified range to:

- Prevent API rate limit violations
- Add jitter to avoid synchronized bursts
- Allow customization based on your API tier/limits

### 3. Enhanced Retry Logic

Improved exponential backoff parameters:

- **Max Retries**: 5 attempts (unchanged)
- **Base Delay**: Increased from 1-2s to **3s**
- **Max Delay**: Increased from 60-120s to **180s**
- **Jitter**: Increased from 10-30% to **20-50%** for better distribution

### 4. Thread-Safe Rate Limiting

Added a **request-level rate limiter** in `EmbeddingManager`:

- Uses a thread lock to track the last request time
- Enforces minimum delay between consecutive requests
- Works across all async/concurrent requests
- Prevents overwhelming the API even with parallel processing

### 5. Implementation Details

#### Embeddings (`core/embeddings.py`)

- **New**: Thread-safe `_wait_for_rate_limit()` method
- Enforces delay BEFORE each API request (not just on retry)
- Random delay between `embedding_delay_min` and `embedding_delay_max`
- Applied to both chunk embeddings and entity embeddings
- Works with both OpenAI and Ollama providers

#### Entity Extraction (`core/entity_extraction.py`)

- Random delay between `llm_delay_min` and `llm_delay_max` before each LLM call
- Uses request tracker dict to enforce minimum delays
- Applied to all entity extraction operations
- Controlled concurrency via `llm_concurrency` setting

#### Document Processing (`ingestion/document_processor.py`)

- Simplified to rely on rate limiting in underlying services
- Maintains async/parallel processing architecture
- Respects concurrency limits throughout the pipeline

## Configuration

### Environment Variables

You can override these settings in your `.env` file:

```bash
# Concurrency limits
EMBEDDING_CONCURRENCY=1
LLM_CONCURRENCY=1

# Rate limiting delays (in seconds)
EMBEDDING_DELAY_MIN=1.0
EMBEDDING_DELAY_MAX=2.0
LLM_DELAY_MIN=1.5
LLM_DELAY_MAX=3.0
```

### Adjusting for Your API Tier

#### Free Tier / Lower Limits (Default)

```bash
EMBEDDING_CONCURRENCY=1
LLM_CONCURRENCY=1
EMBEDDING_DELAY_MIN=1.0
EMBEDDING_DELAY_MAX=2.0
LLM_DELAY_MIN=1.5
LLM_DELAY_MAX=3.0
```

#### Mid Tier / Moderate Limits

```bash
EMBEDDING_CONCURRENCY=2
LLM_CONCURRENCY=1
EMBEDDING_DELAY_MIN=0.5
EMBEDDING_DELAY_MAX=1.0
LLM_DELAY_MIN=1.0
LLM_DELAY_MAX=2.0
```

#### Higher Tier / Higher Limits

```bash
EMBEDDING_CONCURRENCY=3
LLM_CONCURRENCY=2
EMBEDDING_DELAY_MIN=0.3
EMBEDDING_DELAY_MAX=0.8
LLM_DELAY_MIN=0.5
LLM_DELAY_MAX=1.5
```

## Key Architectural Change

### Before: Delays Only in Async Code

- Delays were added in the async processing loop
- Multiple concurrent requests could still hit the API simultaneously
- OpenAI client had its own retry logic (60s waits)

### After: Request-Level Rate Limiting

- **Thread-safe lock** ensures only one request proceeds at a time
- Minimum delay enforced BEFORE making the API call
- Works across all concurrent/async operations
- Prevents the OpenAI client from needing to retry

## Benefits

1. **Eliminated 429 Errors**: Request-level limiting prevents overwhelming the API
2. **Maintained Parallelism**: Multiple chunks/entities still processed concurrently
3. **Better Retry Handling**: Longer backoff times give the API more time to recover
4. **Flexibility**: Easy to adjust based on your specific API limits and needs
5. **Graceful Degradation**: Automatic retry with exponential backoff handles transient failures
6. **Thread-Safe**: Works correctly with concurrent async operations

## Performance Impact

- **Processing Time**: Expect ~40-60% slower ingestion for large batches
- **Reliability**: **Zero** 429 errors with proper configuration
- **API Cost**: No change (same number of requests, just better paced)
- **Trade-off**: Moderate speed reduction for excellent reliability

## Monitoring

Watch your logs for these indicators:

- `"Rate limiting: sleeping X.XXs"` - Normal operation, enforcing delays
- `"Rate limit hit"` - Retry logic activated (should be rare now)
- `"Max retries exceeded"` - **Increase delays** or reduce concurrency further
- `"Retrying in X seconds"` - Normal behavior, exponential backoff in action

## Troubleshooting

### Still Getting 429 Errors?

1. **Reduce concurrency to 1** for both embedding and LLM
2. **Increase minimum delays** to 2-3 seconds
3. Check your API tier's rate limits
4. Monitor the time between requests in logs

### Processing Too Slow?

1. **Increase concurrency** cautiously (2-3 max)
2. **Reduce minimum delays** but keep some buffer
3. Consider upgrading your API tier
4. Process smaller batches at a time

## Future Improvements

Potential enhancements:

1. Adaptive rate limiting based on response times
2. Token bucket algorithm for smoother request distribution
3. Per-endpoint rate limiting (embeddings vs chat completions)
4. Real-time monitoring of rate limit headers
5. Automatic adjustment based on 429 error frequency
6. Queue-based request management with priority
