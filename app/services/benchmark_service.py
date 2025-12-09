import time
import psutil
import json
from app.services.translator_service import translate
from app.services.llm_service import llm_generate
from app.services.rag_service import rag_retrieve

def measure_time(fn, *args, **kwargs):
    """Utility to time any function call."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def memory_snapshot():
    """Capture current process memory usage."""
    p = psutil.Process()
    mem = p.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
    }

def benchmark_pipeline(
    test_text="കേരളത്തിൽ മഴ കനത്തിരിക്കുന്നു.",
    src_lang="mal_Mlym",
    tgt_lang="eng_Latn",
    max_tokens=64,
):
    """
    Runs a synthetic benchmark:
    - Translate → English
    - RAG retrieval
    - LLM inference
    - Back-translate
    Logs intermediate latencies and memory usage to console.
    """

    print("\n===== Running Benchmark =====")
    print(f"Input text: {test_text}")

    results = {}

    # Snapshot 1
    results["memory_before"] = memory_snapshot()

    # 1. Translate → English
    english_text, t_trans_in = measure_time(
        translate, test_text, src_lang, tgt_lang
    )
    print(f"[Benchmark] Translation (input → EN): {t_trans_in:.4f}s")
    results["t_translate_in"] = t_trans_in

    # 2. RAG retrieve
    rag_docs, t_rag = measure_time(rag_retrieve, english_text, 3)
    print(f"[Benchmark] RAG Retrieval: {t_rag:.4f}s (docs={len(rag_docs)})")
    results["t_rag"] = t_rag
    results["rag_docs"] = rag_docs

    # Build prompt
    context = "\n".join(f"Doc {i+1}: {d}" for i, d in enumerate(rag_docs)) if rag_docs else ""
    context_block = f"Relevant context:\n{context}\n" if context else ""
    final_prompt = (
        f"User question:\n{english_text}\n\n"
        f"{context_block}"
        "Answer in simple English."
    )

    # 3. LLM inference
    llm_out_en, t_llm = measure_time(
        llm_generate, final_prompt, max_new_tokens=max_tokens
    )
    print(f"[Benchmark] LLM Inference: {t_llm:.4f}s")
    print(f"[Benchmark] LLM Output: {llm_out_en[:80]}...")
    results["t_llm"] = t_llm

    # 4. Translate back
    final_output, t_trans_out = measure_time(
        translate, llm_out_en, tgt_lang, src_lang
    )
    print(f"[Benchmark] Translation (EN → output): {t_trans_out:.4f}s")
    results["t_translate_out"] = t_trans_out
    results["final_output"] = final_output

    # Total time
    results["total_latency"] = t_trans_in + t_rag + t_llm + t_trans_out
    print(f"[Benchmark] TOTAL latency: {results['total_latency']:.4f}s")

    # Snapshot 2
    results["memory_after"] = memory_snapshot()

    # SIMPLE round-trip accuracy proxy (string overlap)
    overlap = round((len(set(test_text) & set(final_output)) /
                    max(len(set(test_text)), 1)) * 100, 2)
    results["roundtrip_similarity"] = overlap
    print(f"[Benchmark] Round-trip similarity: {overlap}%")

    print("===== Benchmark Complete =====\n")

    return results
