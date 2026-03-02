"""
Query cache performance benchmarking.

Measures cache hit/miss latency, hit ratio, and performance improvement.
"""

import time
import json
from typing import List, Dict
from pathlib import Path
from app.services.query_cache_service import QueryCache
from app.services.rag_service import rag_retrieve, get_embed_model


def benchmark_query_cache(
    queries: List[Dict[str, str]],
    num_repeats: int = 3,
    cache_similarity_threshold: float = 0.80
) -> Dict:
    """
    Benchmark query cache performance: hit ratio, latency, memory usage.
    
    Steps:
    1. Run queries WITHOUT cache (baseline latency)
    2. Run queries WITH cache DISABLED (cache miss latency)
    3. Run queries again (cache hit latency)
    4. Calculate improvement metrics
    
    Args:
        queries: List of {"text": str, "lang": str} dicts
        num_repeats: Number of times to run queries after filling cache (for hit ratio)
        cache_similarity_threshold: Threshold for cache hit detection
    
    Returns:
        {
            "summary": {
                "total_queries": int,
                "total_runs": int,
                "avg_hit_ratio": float (0-1),
                "avg_latency_no_cache_ms": float,
                "avg_latency_cache_miss_ms": float,
                "avg_latency_cache_hit_ms": float,
                "speedup_hit_vs_miss": float,
                "speedup_hit_vs_no_cache": float,
                "total_time_saved_ms": float,
            },
            "per_query": [
                {
                    "text": str,
                    "lang": str,
                    "latency_no_cache_ms": float,
                    "latency_cache_miss_ms": float,
                    "latency_cache_hit_ms": [float, ...],  // per repeat
                    "avg_latency_cache_hit_ms": float,
                    "hit_ratio": float,
                    "speedup_hit_vs_miss": float,
                    "speedup_hit_vs_no_cache": float,
                }
            ],
            "cache_state": {
                "final_cache_entries": int,
                "cache_utilization_percent": float,
            }
        }
    """
    embed_model = get_embed_model()
    
    # Create temporary cache file
    cache_file = Path("models") / "cache_benchmark_temp.json"
    
    results = {
        "summary": {},
        "per_query": [],
        "cache_state": {},
    }
    
    # Stage 1: Baseline (no cache at all - just RAG retrieval)
    print("[Cache Benchmark] Stage 1: Baseline (no cache, RAG only)...")
    baseline_times = []
    
    for query_obj in queries:
        text = query_obj.get("text", "")
        
        t0 = time.perf_counter()
        rag_docs = rag_retrieve(text, top_k=3)
        t1 = time.perf_counter()
        
        baseline_time_ms = (t1 - t0) * 1000
        baseline_times.append(baseline_time_ms)
        print(f"  Query: {text[:50]}... → {baseline_time_ms:.2f}ms")
    
    avg_baseline_ms = sum(baseline_times) / len(baseline_times) if baseline_times else 0
    print(f"[Cache Benchmark] Baseline avg: {avg_baseline_ms:.2f}ms\n")
    
    # Stage 2: Cache miss (first run with cache enabled)
    print("[Cache Benchmark] Stage 2: First cache run (misses)...")
    
    # Create fresh cache
    cache = QueryCache(
        cache_file=cache_file,
        similarity_threshold=cache_similarity_threshold,
        max_entries=1000
    )
    cache.clear()  # Start fresh
    
    cache_miss_times = []
    cache_embedding_times = []
    
    for query_obj in queries:
        text = query_obj.get("text", "")
        
        # Time the embedding + cache lookup + miss + RAG retrieval
        t0 = time.perf_counter()
        
        # Encode query
        query_embedding = embed_model.encode([text])[0].tolist()
        
        # Check cache (will miss)
        cached_result = cache.find_similar_query(query_embedding)
        if cached_result is None:
            # Miss - do RAG retrieval
            rag_docs = rag_retrieve(text, top_k=3)
            
            # Add to cache
            cache.add_query(text, query_embedding, rag_docs)
        
        t1 = time.perf_counter()
        
        time_ms = (t1 - t0) * 1000
        cache_miss_times.append(time_ms)
        print(f"  Query: {text[:50]}... → {time_ms:.2f}ms (miss)")
    
    avg_cache_miss_ms = sum(cache_miss_times) / len(cache_miss_times) if cache_miss_times else 0
    print(f"[Cache Benchmark] Cache miss avg: {avg_cache_miss_ms:.2f}ms\n")
    
    # Stage 3: Cache hits (repeated runs)
    print(f"[Cache Benchmark] Stage 3: Cache hits ({num_repeats} repeats)...")
    
    cache_hit_times = {i: [] for i in range(len(queries))}
    cache_hits_total = 0
    
    for repeat in range(num_repeats):
        for idx, query_obj in enumerate(queries):
            text = query_obj.get("text", "")
            
            # Time embedding + cache lookup (should hit now)
            t0 = time.perf_counter()
            
            query_embedding = embed_model.encode([text])[0].tolist()
            cached_result = cache.find_similar_query(query_embedding)
            
            hit = cached_result is not None
            if hit:
                cache_hits_total += 1
            
            t1 = time.perf_counter()
            
            time_ms = (t1 - t0) * 1000
            cache_hit_times[idx].append(time_ms)
            
            if repeat == 0:  # Only print first repeat to avoid spam
                print(f"  Query {idx+1}: {text[:50]}... → {time_ms:.2f}ms ({'hit' if hit else 'miss'})")
    
    print(f"[Cache Benchmark] Total cache hits: {cache_hits_total}/{len(queries) * num_repeats}\n")
    
    # Calculate per-query stats
    per_query_stats = []
    total_speedup_ms = 0
    
    for idx, query_obj in enumerate(queries):
        text = query_obj.get("text", "")
        lang = query_obj.get("lang", "en")
        
        baseline = baseline_times[idx]
        miss = cache_miss_times[idx]
        hits = cache_hit_times[idx]
        avg_hit = sum(hits) / len(hits) if hits else 0
        
        hit_ratio = len([1 for h in hits if h < miss * 1.2]) / len(hits) if hits else 0  # Within 20% is a "hit"
        speedup_hit_vs_miss = miss / avg_hit if avg_hit > 0 else 0
        speedup_hit_vs_baseline = baseline / avg_hit if avg_hit > 0 else 0
        
        per_query_stats.append({
            "text": text,
            "lang": lang,
            "latency_no_cache_ms": round(baseline, 2),
            "latency_cache_miss_ms": round(miss, 2),
            "latency_cache_hit_ms": [round(h, 2) for h in hits],
            "avg_latency_cache_hit_ms": round(avg_hit, 2),
            "hit_ratio": round(hit_ratio, 3),
            "speedup_hit_vs_miss": round(speedup_hit_vs_miss, 2),
            "speedup_hit_vs_no_cache": round(speedup_hit_vs_baseline, 2),
        })
        
        # Rough estimate of time saved per query if always cached
        total_speedup_ms += (baseline - avg_hit)
    
    results["per_query"] = per_query_stats
    
    # Summary statistics
    avg_hit_ratio = sum(q["hit_ratio"] for q in per_query_stats) / len(per_query_stats) if per_query_stats else 0
    avg_baseline = sum(baseline_times) / len(baseline_times) if baseline_times else 0
    avg_hit = sum(q["avg_latency_cache_hit_ms"] for q in per_query_stats) / len(per_query_stats) if per_query_stats else 0
    
    speedup_hit_vs_miss_overall = avg_cache_miss_ms / avg_hit if avg_hit > 0 else 0
    speedup_hit_vs_baseline_overall = avg_baseline / avg_hit if avg_hit > 0 else 0
    
    results["summary"] = {
        "total_queries": len(queries),
        "total_runs": len(queries) * (1 + num_repeats),
        "num_repeats": num_repeats,
        "avg_hit_ratio": round(avg_hit_ratio, 3),
        "avg_latency_no_cache_ms": round(avg_baseline, 2),
        "avg_latency_cache_miss_ms": round(avg_cache_miss_ms, 2),
        "avg_latency_cache_hit_ms": round(avg_hit, 2),
        "speedup_hit_vs_miss": round(speedup_hit_vs_miss_overall, 2),
        "speedup_hit_vs_no_cache": round(speedup_hit_vs_baseline_overall, 2),
        "total_time_saved_ms": round(total_speedup_ms, 2),
        "cache_efficiency_percent": round((1 - avg_hit / avg_baseline) * 100, 1) if avg_baseline > 0 else 0,
    }
    
    # Cache state
    cache_stats = cache.stats()
    results["cache_state"] = {
        "final_cache_entries": cache_stats["num_cached_queries"],
        "max_entries": cache_stats["max_entries"],
        "cache_utilization_percent": round((cache_stats["num_cached_queries"] / cache_stats["max_entries"]) * 100, 1),
    }
    
    print("[Cache Benchmark] Complete!")
    print(f"Summary: Avg {avg_hit_ratio*100:.1f}% hit ratio, {speedup_hit_vs_miss_overall:.1f}x faster on hit vs miss")
    print(f"Time saved: {total_speedup_ms:.0f}ms total across all queries")
    
    return results
