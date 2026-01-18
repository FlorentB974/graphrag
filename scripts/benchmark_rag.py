#!/usr/bin/env python3
"""
RAG Pipeline Benchmark Script
Times all stages of the RAG pipeline to identify performance bottlenecks.
"""

import asyncio
import functools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors to reduce noise
    format="%(message)s"
)

# Reduce noise from other loggers
for logger_name in ["httpx", "openai", "urllib3", "neo4j"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


def async_timer(func: Callable) -> Callable:
    """Decorator to time async function execution."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


class RAGBenchmark:
    """Benchmark class for the RAG pipeline."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.llm_calls: List[Dict[str, Any]] = []
        self._original_generate_response = None
        self._original_analyze_query = None
        
    def _record_timing(self, stage: str, duration: float):
        """Record timing for a stage."""
        if stage not in self.timings:
            self.timings[stage] = []
        self.timings[stage].append(duration)
        
    def _wrap_llm_calls(self):
        """Wrap LLM calls to track them."""
        from core.llm import llm_manager
        
        self._original_generate_response = llm_manager.generate_response
        self._original_analyze_query = llm_manager.analyze_query
        
        def tracked_generate_response(*args, **kwargs):
            start = time.perf_counter()
            result = self._original_generate_response(*args, **kwargs)
            duration = time.perf_counter() - start
            
            # Extract prompt info
            prompt = args[0] if args else kwargs.get("prompt", "")
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            self.llm_calls.append({
                "type": "generate_response",
                "prompt_preview": prompt_preview,
                "prompt_length": len(prompt),
                "duration": duration,
                "max_tokens": kwargs.get("max_tokens", 1000),
            })
            return result
        
        def tracked_analyze_query(*args, **kwargs):
            start = time.perf_counter()
            result = self._original_analyze_query(*args, **kwargs)
            duration = time.perf_counter() - start
            
            query = args[0] if args else kwargs.get("query", "")
            
            self.llm_calls.append({
                "type": "analyze_query",
                "query": query,
                "duration": duration,
            })
            return result
        
        llm_manager.generate_response = tracked_generate_response
        llm_manager.analyze_query = tracked_analyze_query
        
    def _unwrap_llm_calls(self):
        """Restore original LLM methods."""
        from core.llm import llm_manager
        
        if self._original_generate_response:
            llm_manager.generate_response = self._original_generate_response
        if self._original_analyze_query:
            llm_manager.analyze_query = self._original_analyze_query
    
    def run_benchmark(self, query: str, runs: int = 1) -> Dict[str, Any]:
        """Run benchmark on the RAG pipeline."""
        
        print(f"\n{'='*70}")
        print(f"RAG PIPELINE BENCHMARK")
        print(f"{'='*70}")
        print(f"\nQuery: \"{query}\"")
        print(f"Runs: {runs}")
        print(f"\n{'-'*70}")
        
        # Initialize components
        print("\n[1] Initializing components...")
        
        init_start = time.perf_counter()
        from core.embeddings import embedding_manager
        from core.graph_db import graph_db
        from core.llm import llm_manager
        from rag.graph_rag import graph_rag
        from rag.nodes.query_analysis import analyze_query, _detect_follow_up_question, _create_contextualized_query
        from rag.nodes.retrieval import retrieve_documents_async, document_retriever
        from rag.nodes.graph_reasoning import reason_with_graph
        from rag.nodes.generation import generate_response
        init_duration = time.perf_counter() - init_start
        print(f"   Initialization: {init_duration:.3f}s")
        
        # Wrap LLM calls for tracking
        self._wrap_llm_calls()
        
        try:
            for run in range(runs):
                print(f"\n{'='*70}")
                print(f"RUN {run + 1}/{runs}")
                print(f"{'='*70}")
                
                self.llm_calls = []  # Reset for each run
                
                # ===== STAGE 1: QUERY ANALYSIS =====
                print("\n[STAGE 1] Query Analysis")
                print("-" * 40)
                
                # 1a. Follow-up detection (if chat history exists)
                chat_history = []  # No history for this benchmark
                follow_up_start = time.perf_counter()
                is_follow_up = False
                needs_context = False
                context_query = query
                
                if chat_history and len(chat_history) >= 2:
                    follow_up_result = _detect_follow_up_question(query, chat_history)
                    is_follow_up = follow_up_result.get("is_follow_up", False)
                    needs_context = follow_up_result.get("needs_context", False)
                follow_up_duration = time.perf_counter() - follow_up_start
                print(f"   1a. Follow-up detection: {follow_up_duration:.3f}s")
                self._record_timing("follow_up_detection", follow_up_duration)
                
                # 1b. REMOVED: Direct LLM query analysis test (was causing double-counting)
                # The full pipeline test below will measure the actual analyze_query() function
                
                # 1c. Full analyze_query (now using optimized heuristics-only approach)
                full_analysis_start = time.perf_counter()
                query_analysis = analyze_query(query, chat_history)
                full_analysis_duration = time.perf_counter() - full_analysis_start
                print(f"   1b. Full analyze_query (heuristics): {full_analysis_duration:.3f}s")
                self._record_timing("full_query_analysis", full_analysis_duration)
                
                print(f"   Query type: {query_analysis.get('query_type', 'unknown')}")
                print(f"   Complexity: {query_analysis.get('complexity', 'unknown')}")
                print(f"   Multi-hop recommended: {query_analysis.get('multi_hop_recommended', False)}")
                
                # ===== STAGE 2: EMBEDDING GENERATION =====
                print("\n[STAGE 2] Embedding Generation")
                print("-" * 40)
                
                embed_start = time.perf_counter()
                query_embedding = embedding_manager.get_embedding(query)
                embed_duration = time.perf_counter() - embed_start
                print(f"   Query embedding: {embed_duration:.3f}s")
                print(f"   Embedding dimension: {len(query_embedding)}")
                self._record_timing("embedding_generation", embed_duration)
                
                # ===== STAGE 3: DOCUMENT RETRIEVAL =====
                print("\n[STAGE 3] Document Retrieval")
                print("-" * 40)
                
                # 3a. Vector similarity search
                vector_start = time.perf_counter()
                similar_chunks = graph_db.vector_similarity_search(query_embedding, 5)
                vector_duration = time.perf_counter() - vector_start
                print(f"   3a. Vector similarity search: {vector_duration:.3f}s ({len(similar_chunks)} chunks)")
                self._record_timing("vector_search", vector_duration)
                
                # 3b. Entity similarity search
                entity_start = time.perf_counter()
                relevant_entities = graph_db.entity_similarity_search(query, 5)
                entity_duration = time.perf_counter() - entity_start
                print(f"   3b. Entity similarity search: {entity_duration:.3f}s ({len(relevant_entities)} entities)")
                self._record_timing("entity_search", entity_duration)
                
                # 3c. Full retrieval (hybrid mode)
                retrieval_start = time.perf_counter()
                retrieved_chunks = asyncio.run(
                    retrieve_documents_async(
                        query=query,
                        query_analysis=query_analysis,
                        retrieval_mode="hybrid",
                        top_k=5,
                        graph_expansion=True,
                        use_multi_hop=False,
                    )
                )
                retrieval_duration = time.perf_counter() - retrieval_start
                print(f"   3c. Full hybrid retrieval: {retrieval_duration:.3f}s ({len(retrieved_chunks)} chunks)")
                self._record_timing("full_retrieval", retrieval_duration)
                
                # ===== STAGE 4: GRAPH REASONING =====
                print("\n[STAGE 4] Graph Reasoning")
                print("-" * 40)
                
                graph_start = time.perf_counter()
                graph_context = reason_with_graph(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    query_analysis=query_analysis,
                    retrieval_mode="hybrid",
                )
                graph_duration = time.perf_counter() - graph_start
                print(f"   Graph reasoning: {graph_duration:.3f}s ({len(graph_context)} final chunks)")
                self._record_timing("graph_reasoning", graph_duration)
                
                # ===== STAGE 5: RESPONSE GENERATION =====
                print("\n[STAGE 5] Response Generation")
                print("-" * 40)
                
                generation_start = time.perf_counter()
                response_data = generate_response(
                    query=query,
                    context_chunks=graph_context,
                    query_analysis=query_analysis,
                    temperature=0.7,
                    chat_history=None,
                )
                generation_duration = time.perf_counter() - generation_start
                print(f"   Response generation: {generation_duration:.3f}s [LLM CALL]")
                self._record_timing("response_generation", generation_duration)
                
                response_preview = response_data.get("response", "")[:200]
                print(f"   Response preview: {response_preview}...")
                
                # ===== FULL PIPELINE TEST =====
                # OPTIMIZATION (Solution 6): Clear LLM call tracking before full pipeline test
                # to avoid double-counting individual stage tests with the full pipeline
                print("\n[FULL PIPELINE] Using graph_rag.query() - Starting Fresh")
                print("-" * 40)
                print("   Note: LLM call counter reset to measure only the full pipeline")
                
                # Reset LLM call tracking for accurate full pipeline measurement
                self.llm_calls = []
                
                full_start = time.perf_counter()
                full_result = graph_rag.query(
                    user_query=query,
                    retrieval_mode="hybrid",
                    top_k=5,
                    temperature=0.7,
                    use_multi_hop=False,
                )
                full_duration = time.perf_counter() - full_start
                print(f"   Full pipeline: {full_duration:.3f}s")
                print(f"   LLM calls in full pipeline: {len(self.llm_calls)}")
                self._record_timing("full_pipeline", full_duration)
                
            # ===== SUMMARY =====
            print(f"\n{'='*70}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*70}")
            
            print("\n📊 Stage Timings (average over runs):")
            print("-" * 50)
            
            stage_order = [
                ("follow_up_detection", "Follow-up Detection"),
                ("full_query_analysis", "Query Analysis (Heuristics)"),
                ("embedding_generation", "Embedding Generation"),
                ("vector_search", "Vector Similarity Search"),
                ("entity_search", "Entity Similarity Search"),
                ("full_retrieval", "Full Hybrid Retrieval"),
                ("graph_reasoning", "Graph Reasoning"),
                ("response_generation", "Response Generation (Stage Test)"),
                ("full_pipeline", "FULL PIPELINE (Actual)"),
            ]
            
            total_individual = 0
            for key, label in stage_order:
                if key in self.timings:
                    avg = sum(self.timings[key]) / len(self.timings[key])
                    if key != "full_pipeline":
                        total_individual += avg
                    marker = " 🔴" if avg > 1.0 else " 🟡" if avg > 0.5 else " 🟢"
                    print(f"   {label:30s}: {avg:6.3f}s{marker}")
            
            print(f"\n{'='*70}")
            print("🔍 LLM CALLS ANALYSIS")
            print(f"{'='*70}")
            
            llm_call_count = len(self.llm_calls)
            total_llm_time = sum(call["duration"] for call in self.llm_calls)
            
            print(f"\n   Total LLM calls: {llm_call_count}")
            print(f"   Total LLM time: {total_llm_time:.3f}s")
            
            if self.llm_calls:
                print("\n   Individual LLM calls:")
                for i, call in enumerate(self.llm_calls, 1):
                    call_type = call["type"]
                    duration = call["duration"]
                    if call_type == "generate_response":
                        preview = call.get("prompt_preview", "")[:50]
                        print(f"   {i}. {call_type}: {duration:.3f}s - \"{preview}...\"")
                    else:
                        query_preview = call.get("query", "")[:50]
                        print(f"   {i}. {call_type}: {duration:.3f}s - \"{query_preview}\"")
            
            print(f"\n{'='*70}")
            print("⚠️  POTENTIAL BOTTLENECKS")
            print(f"{'='*70}")
            
            bottlenecks = []
            for key, label in stage_order:
                if key in self.timings:
                    avg = sum(self.timings[key]) / len(self.timings[key])
                    if avg > 1.0:
                        bottlenecks.append((label, avg, "🔴 CRITICAL"))
                    elif avg > 0.5:
                        bottlenecks.append((label, avg, "🟡 WARNING"))
            
            if bottlenecks:
                for label, duration, severity in sorted(bottlenecks, key=lambda x: -x[1]):
                    print(f"   {severity} {label}: {duration:.3f}s")
            else:
                print("   No significant bottlenecks detected.")
                
            print(f"\n{'='*70}")
            print("📈 RECOMMENDATIONS")
            print(f"{'='*70}")
            
            # Count LLM calls per stage
            analysis_calls = sum(1 for c in self.llm_calls if "analyze" in c.get("type", "").lower() or "analyze" in c.get("query", "").lower())
            
            if llm_call_count > 3:
                print(f"   • Too many LLM calls ({llm_call_count}). Consider caching or reducing.")
                
            if total_llm_time > 5:
                print(f"   • LLM calls taking {total_llm_time:.1f}s. Consider parallel execution or caching.")
                
            if "embedding_generation" in self.timings:
                embed_avg = sum(self.timings["embedding_generation"]) / len(self.timings["embedding_generation"])
                if embed_avg > 0.5:
                    print(f"   • Embedding generation slow ({embed_avg:.3f}s). Consider local embedding model.")
                    
            return {
                "timings": {k: sum(v)/len(v) for k, v in self.timings.items()},
                "llm_calls": self.llm_calls,
                "total_llm_calls": llm_call_count,
                "total_llm_time": total_llm_time,
            }
            
        finally:
            self._unwrap_llm_calls()


def main():
    """Run benchmark."""
    query = "tell me about the heat pump quote"
    
    print("\n" + "="*70)
    print("TESTING CACHE - RUNNING QUERY TWICE")
    print("="*70)
    print(f"Query: '{query}'")
    print("First run should generate embeddings and LLM responses")
    print("Second run should use cached values and be much faster")
    print("="*70 + "\n")
    
    benchmark = RAGBenchmark()
    
    # First run - no cache
    print("\n" + "🔵 " + "="*65)
    print("FIRST RUN (Building Cache)")
    print("="*70)
    results1 = benchmark.run_benchmark(query, runs=1)
    
    # Second run - should use cache
    print("\n\n" + "🟢 " + "="*65)
    print("SECOND RUN (Using Cache)")
    print("="*70)
    benchmark2 = RAGBenchmark()
    results2 = benchmark2.run_benchmark(query, runs=1)
    
    # Compare results
    print("\n" + "="*70)
    print("CACHE PERFORMANCE COMPARISON")
    print("="*70)
    
    if results1 and results2:
        for key in ["full_pipeline", "embedding_generation"]:
            if key in results1["timings"] and key in results2["timings"]:
                first = results1["timings"][key]
                second = results2["timings"][key]
                speedup = ((first - second) / first * 100) if first > 0 else 0
                print(f"   {key:30s}: {first:.3f}s -> {second:.3f}s ({speedup:+.1f}% improvement)")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
