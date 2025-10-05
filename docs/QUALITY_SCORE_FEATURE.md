# Quality Score Feature

## Overview

The Quality Score feature provides automatic evaluation of LLM-generated answers, displaying a comprehensive quality assessment (0-100%) in the Streamlit UI's Sources tab.

## Quick Start

### Using the Feature

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Ask a question** in the chat interface

3. **View the quality score:**
   - Navigate to the **Sources** tab (first tab in right sidebar)
   - The quality score appears at the top with color coding
   - Expand "📊 View Score Breakdown" for detailed metrics

### Interpreting Scores

- **🟢 Green (≥80%):** High quality - well-grounded, complete, coherent
- **🟡 Yellow (60-79%):** Medium quality - acceptable with minor gaps
- **🔴 Red (<60%):** Needs improvement - significant issues detected

## Score Components

Each answer is evaluated on 5 weighted metrics:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Context Relevance** | 30% | How well the answer uses the provided context |
| **Answer Completeness** | 25% | Whether all parts of the query are addressed |
| **Factual Grounding** | 25% | Degree to which claims are supported by sources |
| **Coherence** | 10% | Logical flow and readability of the answer |
| **Citation Quality** | 10% | Proper use and attribution of sources |

**Total Score** = Weighted sum of all components

## Configuration

### Enable/Disable

Add to your `.env` file:
```bash
# Enable quality scoring (default: True)
ENABLE_QUALITY_SCORING=True

# Enable score caching for performance (default: True)
QUALITY_SCORE_CACHE_ENABLED=True
```

### Customize Weights

Modify in `config/settings.py`:
```python
quality_score_weights: dict = Field(
    default={
        "context_relevance": 0.30,
        "answer_completeness": 0.25,
        "factual_grounding": 0.25,
        "coherence": 0.10,
        "citation_quality": 0.10
    }
)
```

The weights must sum to 1.0.

## Technical Details

### Scoring Method

- **Primary:** LLM-based evaluation (temperature=0.0 for consistency)
- **Fallback:** Heuristic-based scoring if LLM calls fail
- **Scale:** Each component scored 0-10, converted to 0-100%

### Performance

- **Latency:** ~1-3 seconds per score (uses caching to minimize)
- **Caching:** MD5-based cache with 100-entry limit (FIFO eviction)
- **Non-blocking:** Scoring failures never prevent answer generation

### Architecture

```
Query → RAG Pipeline → Generate Answer → Calculate Score → Display in UI
                                            ↓
                                      5 Components
                                            ↓
                                      Weighted Total
```

## API Reference

### QualityScorer Class

Located in `core/quality_scorer.py`:

```python
from core.quality_scorer import quality_scorer

# Calculate quality score
score = quality_scorer.calculate_quality_score(
    answer="The generated answer text",
    query="User's original question",
    context_chunks=[...],  # List of context chunks used
    sources=[...]          # List of source information
)

# Returns:
# {
#     "total": 85.5,
#     "breakdown": {
#         "context_relevance": 90.0,
#         "answer_completeness": 85.0,
#         "factual_grounding": 88.0,
#         "coherence": 82.0,
#         "citation_quality": 80.0
#     },
#     "confidence": "high"  # high/medium/low
# }
```

## Testing

Run the test suite:
```bash
pytest tests/test_quality_scorer.py -v
```

All 8 tests should pass:
- Initialization test
- High quality answer test
- Low quality answer test
- Empty context test
- Disabled scoring test
- Score components test
- Cache functionality test
- Score range validation test

## Troubleshooting

### Score Always Shows 50%

**Issue:** All component scores are around 50%  
**Cause:** LLM calls failing, using neutral heuristic scores  
**Solution:** Check LLM configuration and API connectivity

### No Score Displayed

**Issue:** Quality score section doesn't appear  
**Causes:**
1. Feature disabled in settings
2. Scoring failed during generation

**Solutions:**
1. Check `ENABLE_QUALITY_SCORING` in settings
2. Check logs for scoring errors

### Slow Response Time

**Issue:** Answers take much longer to generate  
**Cause:** Quality scoring adds ~1-3 seconds  
**Solutions:**
1. Caching reduces repeat scores
2. Disable scoring for faster responses: `ENABLE_QUALITY_SCORING=False`

### Inconsistent Scores

**Issue:** Same query gets different scores  
**Cause:** LLM evaluation has some variance  
**Solution:** Enable caching - identical query+answer pairs return cached scores

## Best Practices

### For Users

1. **Use color coding** for quick quality assessment
2. **Expand breakdown** when you need to understand score details
3. **Consider confidence level** - low confidence means inconsistent components
4. **Compare scores** across different queries to gauge system performance

### For Developers

1. **Monitor scores** to identify systematic quality issues
2. **Adjust weights** based on your use case priorities
3. **Enable caching** for production deployments
4. **Log scoring failures** for debugging
5. **Test with diverse queries** to validate scoring accuracy

## Examples

### Example 1: High Quality Answer

**Query:** "What is photosynthesis?"

**Answer:** "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water using sunlight."

**Score:** 🟢 88%
- Context Relevance: 95%
- Answer Completeness: 90%
- Factual Grounding: 92%
- Coherence: 85%
- Citation Quality: 80%

### Example 2: Medium Quality Answer

**Query:** "Explain machine learning in detail"

**Answer:** "Machine learning is a subset of AI that allows computers to learn."

**Score:** 🟡 65%
- Context Relevance: 75%
- Answer Completeness: 55% (incomplete - too brief)
- Factual Grounding: 70%
- Coherence: 65%
- Citation Quality: 70%

### Example 3: Low Quality Answer

**Query:** "What is the capital of France?"

**Answer:** "The capital might be somewhere in Europe, possibly Paris or another major city."

**Score:** 🔴 45%
- Context Relevance: 40%
- Answer Completeness: 50%
- Factual Grounding: 35% (uncertain language)
- Coherence: 60%
- Citation Quality: 40%

## Future Enhancements

Potential improvements for future versions:

1. **Historical Tracking:** Store scores in database for analytics
2. **User Feedback:** Allow thumbs up/down to improve scoring
3. **Specialized Models:** Use smaller, fine-tuned model for scoring
4. **Async Scoring:** Run scoring in background without blocking UI
5. **Score Trends:** Show quality trends over conversation
6. **Auto-Regeneration:** Automatically regenerate low-scoring answers
7. **Export Metrics:** Export quality scores for analysis
8. **A/B Testing:** Test different scoring weight configurations

## Support

For issues or questions:

1. Check logs in the application console
2. Review test suite: `pytest tests/test_quality_scorer.py -v`
3. Refer to comprehensive documentation:
   - `IMPLEMENTATION_COMPLETE.md` - Implementation details
   - `TODO_QUALITY_SCORE.md` - Original planning
   - `docs/quality_score_architecture.md` - Architecture diagrams
   - `docs/quality_score_examples.md` - Code examples

## License

Same as the main GraphRAG project.

---

**Version:** 1.0.0  
**Last Updated:** 2024-01-05  
**Status:** Production Ready ✅
