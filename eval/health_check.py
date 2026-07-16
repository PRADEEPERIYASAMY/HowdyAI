import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    AppConfig,
    ResponseCache,
    ConversationMemory,
    QueryGuardrail,
    QueryRewriter,
    HybridRetriever,
    OpenAILanguageModel,
    run_pipeline,
    pipeline_graph
)

def run_test(name, func):
    print(f"\n[{name}] Running...")
    try:
        success, msg = func()
        if success:
            print(f"[{name}] PASS: {msg}")
        else:
            print(f"[{name}] FAIL: {msg}")
    except Exception as e:
        print(f"[{name}] ERROR: {e}")

def get_components():
    config = AppConfig()
    cache = ResponseCache(config.CACHE_DB_PATH)
    memory = ConversationMemory(max_turns=config.MEMORY_MAX_TURNS)
    guardrail = QueryGuardrail(config)
    rewriter = QueryRewriter(config)
    retriever = HybridRetriever(config)
    generator = OpenAILanguageModel(template_path=config.TEMPLATE_PATH, model_name=config.FAST_MODEL)
    return config, cache, memory, guardrail, rewriter, retriever, generator

def test_1_sanity_check():
    config, cache, memory, guardrail, rewriter, retriever, generator = get_components()
    q = "What is the course description for CSCE 221?"
    result = run_pipeline(q, config, cache, memory, guardrail, rewriter, retriever, generator, use_cache=False)
    ans = result.get("answer", "")
    if "[" in ans and "]" in ans and len(ans) > 20:
        return True, f"Generated grounded answer with citation: {ans[:60]}... (Lat: {result.get('latency_s'):.2f}s)"
    return False, "Answer lacked citations or was empty."

def test_2_cache_hit():
    config, cache, memory, guardrail, rewriter, retriever, generator = get_components()
    q = "Who is the dean of the College of Engineering?"
    
    # Clear cache for this query to guarantee Run 1 is a true MISS
    with cache._conn() as conn:
        conn.execute("DELETE FROM cache WHERE query=?", (q,))
        conn.commit()
    
    # Run 1
    t0 = time.time()
    r1 = run_pipeline(q, config, cache, memory, guardrail, rewriter, retriever, generator, use_cache=True)
    t1 = time.time()
    
    # Run 2
    t2 = time.time()
    r2 = run_pipeline(q, config, cache, memory, guardrail, rewriter, retriever, generator, use_cache=True)
    t3 = time.time()
    
    lat1, lat2 = t1-t0, t3-t2
    if r2.get("cache_hit") == True and lat2 < 0.5:
        return True, f"Run 1: {lat1:.2f}s, Run 2: {lat2:.2f}s (Cache Hit Confirmed)"
    return False, f"Run 2 cache_hit was {r2.get('cache_hit')}, latency: {lat2:.2f}s"

def test_3_guardrail_ordering():
    config, cache, memory, guardrail, rewriter, retriever, generator = get_components()
    q = "Who won the Stanley Cup in 2024?"
    
    # We mock retrieve to ensure it doesn't get called
    called_retrieve = [False]
    original_retrieve = retriever.retrieve
    def mock_retrieve(*args, **kwargs):
        called_retrieve[0] = True
        return original_retrieve(*args, **kwargs)
    retriever.retrieve = mock_retrieve
    
    result = run_pipeline(q, config, cache, memory, guardrail, rewriter, retriever, generator, use_cache=False)
    
    if result.get("guardrail_hit") and not called_retrieve[0]:
        return True, "Query correctly rejected by guardrail, retrieval was entirely bypassed."
    return False, f"Guardrail Hit: {result.get('guardrail_hit')}, Retrieve Called: {called_retrieve[0]}"

def test_5_6_retry_and_cache():
    config, cache, memory, guardrail, rewriter, retriever, generator = get_components()
    q = "How many total credit hours are required for the CSCE degree?"
    
    config.USE_REFLECTION = True
    # Force hallucination by giving it garbage context
    original_retrieve = retriever.retrieve
    def mock_retrieve(*args, **kwargs):
        return [{"url": "http://garbage.com", "title": "Garbage", "description": "Absolutely nothing relevant here. The TAMU CSCE degree takes 9000 hours.", "metadata": {"content": "Absolutely nothing relevant here. The TAMU CSCE degree takes 9000 hours."}}]
    retriever.retrieve = mock_retrieve
    
    # Clear cache for this query
    with cache._conn() as conn:
        conn.execute("DELETE FROM cache WHERE query=?", (q,))
        conn.commit()
    
    result = run_pipeline(q, config, cache, memory, guardrail, rewriter, retriever, generator, use_cache=True)
    
    # Check cache and answer
    cached_val = cache.get(q)
    ans = result.get("answer", "")
    
    if "I apologize, but I could not verify" in ans and cached_val is None:
        return True, "Hallucination caught, safely rejected, and NOT written to cache."
    return False, f"Answer: {ans[:30]}, Cache entry exists: {cached_val is not None}"

def test_7_langgraph_structure():
    graph = pipeline_graph.get_graph()
    try:
        mermaid = graph.draw_mermaid()
        if "node_guardrail" in mermaid and "node_evaluate" in mermaid:
            return True, "Graph compiles and nodes are correctly registered."
    except Exception as e:
        return False, f"Failed to compile graph: {e}"
    return False, "Graph compiled but nodes missing."

if __name__ == "__main__":
    print("=======================================")
    print("  HowdyAI System Health Check Runbook")
    print("=======================================")
    run_test("1. Sanity Check", test_1_sanity_check)
    run_test("2. Cache Miss -> Hit", test_2_cache_hit)
    run_test("3. Guardrail Ordering", test_3_guardrail_ordering)
    run_test("5/6. Retry Loop & Cache Safety", test_5_6_retry_and_cache)
    run_test("7. LangGraph Compilation", test_7_langgraph_structure)
    print("\nHealth check complete.")
