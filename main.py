"""
main.py  –  HowdyAI enhanced pipeline orchestrator.

Replaces the base single-turn argparse CLI with a full multi-stage pipeline:
  1. Out-of-scope guardrail
  2. SQLite cache lookup
  3. LLM query rewriting
  4. Hybrid retrieval  (Google Search + Chroma, RRF fusion)
  5. LLM-based summarization  (parallel, per document)
  6. LLM-based re-ranking
  7. Answer generation with inline citations
  8. Conversation memory update
  9. Cache write

Can be run as a CLI for testing:
    python main.py "What are the CSCE 670 prerequisites?"

The Streamlit UI (app.py) imports run_pipeline() directly.
"""

import argparse
import logging
import logging.config
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import AppConfig
from src.cache import ResponseCache
from src.guardrail import QueryGuardrail
from src.memory import ConversationMemory
from src.search.query_rewriter import QueryRewriter
from src.search.hybrid_retriever import HybridRetriever
from src.search.summarize_with_llm import summarize_search_results_with_llm
from src.search.rank_with_llm import rank_results_with_llm
from src.language_models.openai_language_model import OpenAILanguageModel

logger = logging.getLogger(__name__)


# ── Citation injection ────────────────────────────────────────────────────────

def build_cited_context(ranked_results: list) -> tuple[str, list]:
    """
    Build a numbered source block from ranked results and return
    (context_string, sources_list) where sources_list has title + url.
    """
    lines = []
    sources = []
    words_used = 0
    MAX_CONTEXT_WORDS = 3000

    grouped = {}
    for item in ranked_results:
        url = item.get("url", "")
        if url not in grouped:
            grouped[url] = {"title": item.get("title", url), "summaries": []}
        
        summary = item.get("llm_summary", item.get("truncated_html", ""))[:1200]
        grouped[url]["summaries"].append(summary)

    for i, (url, data) in enumerate(grouped.items(), start=1):
        title = data["title"]
        combined_summary = " ... ".join(data["summaries"])
        word_count = len(combined_summary.split())
        
        if words_used + word_count > MAX_CONTEXT_WORDS:
            combined_summary = " ".join(combined_summary.split()[:MAX_CONTEXT_WORDS - words_used])
            
        lines.append(f"[{i}] Source: {title}\nURL: {url}\n{combined_summary}")
        sources.append({"index": i, "title": title, "url": url})
        words_used += word_count
        
        if words_used >= MAX_CONTEXT_WORDS:
            break

    return "\n\n".join(lines), sources


# ── Core pipeline function ────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    config: AppConfig,
    cache: ResponseCache,
    memory: ConversationMemory,
    guardrail: QueryGuardrail,
    rewriter: QueryRewriter,
    retriever: HybridRetriever,
    generator: OpenAILanguageModel,
    use_cache: bool = True,
) -> dict:
    """
    Run the full HowdyAI pipeline for a single query.

    Returns dict:
        answer          : str   — markdown answer with inline [n] citations
        sources         : list  — [{"index", "title", "url"}, ...]
        rewritten_query : str
        cache_hit       : bool
        guardrail_hit   : bool
        latency_s       : float
    """
    t0 = time.perf_counter()

    # ── 1. Out-of-scope guardrail ─────────────────────────────────────────
    in_scope, rejection_msg = guardrail.check(query)
    if not in_scope:
        return {
            "answer": rejection_msg,
            "sources": [],
            "rewritten_query": query,
            "cache_hit": False,
            "guardrail_hit": True,
            "latency_s": round(time.perf_counter() - t0, 2),
        }

    # ── 2. Cache lookup ───────────────────────────────────────────────────
    if use_cache:
        cached = cache.get(query)
        if cached:
            cached["cache_hit"] = True
            cached["guardrail_hit"] = False
            cached["latency_s"] = round(time.perf_counter() - t0, 2)
            return cached

    # ── 3. Query rewriting ────────────────────────────────────────────────
    history_str = memory.as_string()
    rewritten = rewriter.rewrite(query, history=history_str)

    # ── 4. Hybrid retrieval ───────────────────────────────────────────────
    logger.info(f"Retrieving documents for: {rewritten!r}")
    search_results = retriever.retrieve(rewritten)
    if not search_results:
        answer = (
            "I couldn't find relevant TAMU information for your question. "
            "Please try rephrasing or visit [tamu.edu](https://www.tamu.edu)."
        )
        return {
            "answer": answer, "sources": [], "rewritten_query": rewritten,
            "cache_hit": False, "guardrail_hit": False,
            "latency_s": round(time.perf_counter() - t0, 2),
        }

    # ── 5. LLM summarization (parallel) ──────────────────────────────────
    logger.info(f"Summarizing {len(search_results)} retrieved documents…")
    filtered, _ = summarize_search_results_with_llm(config, query, search_results)

    # ── 6. LLM re-ranking ────────────────────────────────────────────────
    logger.info("Re-ranking summaries…")
    ranked, _ = rank_results_with_llm(config, query, filtered)
    ranked = ranked[:config.RANKER_TOP_K]

    # ── 7. Answer generation with inline citations ────────────────────────
    context, sources = build_cited_context(ranked)
    prompt = generator.generate_prompt(
        context=context,
        question=query,
        history=history_str,
    )
    logger.info("Generating final answer…")
    response = generator.invoke(prompt)
    answer = response.content.strip()

    # ── 8. Memory update ──────────────────────────────────────────────────
    memory.add_turn(query, answer)

    # ── 9. Cache write ────────────────────────────────────────────────────
    result = {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten,
        "cache_hit": False,
        "guardrail_hit": False,
        "latency_s": round(time.perf_counter() - t0, 2),
    }
    if use_cache:
        cache.set(query, result)

    return result


# ── Component factory ─────────────────────────────────────────────────────────

def build_pipeline_components(config: AppConfig):
    """Instantiate and return all pipeline components."""
    cache = ResponseCache(config.CACHE_DB_PATH)
    memory = ConversationMemory(max_turns=config.MEMORY_MAX_TURNS)
    guardrail = QueryGuardrail(config)
    rewriter = QueryRewriter(config)
    retriever = HybridRetriever(config)
    generator = OpenAILanguageModel(config.TEMPLATE_PATH)
    return cache, memory, guardrail, rewriter, retriever, generator


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    config = AppConfig(log_level="INFO")
    logging.config.dictConfig(config.logging_config)

    parser = argparse.ArgumentParser(description="HowdyAI – TAMU RAG Chatbot")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    cache, memory, guardrail, rewriter, retriever, generator = build_pipeline_components(config)

    result = run_pipeline(
        query=args.query_text,
        config=config,
        cache=cache,
        memory=memory,
        guardrail=guardrail,
        rewriter=rewriter,
        retriever=retriever,
        generator=generator,
    )

    print("\n" + "="*60)
    print(f"Query    : {args.query_text}")
    print(f"Rewritten: {result['rewritten_query']}")
    print(f"Cache hit: {result['cache_hit']}  |  Latency: {result['latency_s']}s")
    print("-"*60)
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['title']}\n      {s['url']}")
    print("="*60)


if __name__ == "__main__":
    main()
