"""
main.py  –  HowdyAI enhanced pipeline orchestrator using LangGraph.

Replaces the base single-turn linear logic with a robust graph-based state machine:
  1. Cache lookup
  2. Query rewriting
  3. Out-of-scope guardrail
  4. Hybrid retrieval  (Brave Search + Chroma, RRF fusion)
  5. Local TF-IDF context extraction
  6. Answer generation with inline citations
  7. Conversation memory update & Cache write
"""

import argparse
import logging
import logging.config
import time
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END

from config import AppConfig
from src.cache import ResponseCache
from src.guardrail import QueryGuardrail
from src.memory import ConversationMemory
from src.search.query_rewriter import QueryRewriter
from src.search.hybrid_retriever import HybridRetriever
from src.search.data_processor import clean_html, extract_relevant_context
from src.language_models.openai_language_model import OpenAILanguageModel
from src.search.cross_encoder_ranker import CrossEncoderRanker

logger = logging.getLogger(__name__)

# Global singleton for the cross-encoder to avoid reloading weights
_cross_encoder_instance = None
def get_cross_encoder():
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        _cross_encoder_instance = CrossEncoderRanker()
    return _cross_encoder_instance

# ── Citation injection ────────────────────────────────────────────────────────

def build_cited_context(ranked_results: list) -> tuple[str, list]:
    lines = []
    sources = []
    words_used = 0
    MAX_CONTEXT_WORDS = 8000

    grouped = {}
    for item in ranked_results:
        url = item.get("url", "")
        title = item.get("title") or url or "TAMU Document"
        if url not in grouped:
            grouped[url] = {"title": title, "summaries": []}
        
        summary = item.get("llm_summary", item.get("truncated_html", ""))
        grouped[url]["summaries"].append(summary)

    for i, (url, data) in enumerate(grouped.items(), start=1):
        title = data["title"]
        combined_summary = " ... ".join(data["summaries"])
        word_count = len(combined_summary.split())
        
        if words_used + word_count > MAX_CONTEXT_WORDS:
            combined_summary = " ".join(combined_summary.split()[:MAX_CONTEXT_WORDS - words_used])
        
        words_used += len(combined_summary.split())
            
        lines.append(f"[{i}] Source: {title}\nURL: {url}\n{combined_summary}")
        sources.append({"index": i, "title": title, "url": url})
        words_used += word_count
        
        if words_used >= MAX_CONTEXT_WORDS:
            break

    return "\n\n".join(lines), sources


# ── LangGraph State & Nodes ───────────────────────────────────────────────────

class PipelineState(TypedDict):
    query: str
    config: AppConfig
    cache: ResponseCache
    memory: ConversationMemory
    guardrail: QueryGuardrail
    rewriter: QueryRewriter
    retriever: HybridRetriever
    generator: OpenAILanguageModel
    use_cache: bool
    stream: bool
    
    # Internal routing state
    start_time: float
    history_str: str
    rewritten_query: Optional[str]
    search_results: Optional[List[Dict]]
    context: Optional[str]
    sources: Optional[List[Dict]]
    retry_count: int
    broad_search_used: bool
    hallucination_detected: bool
    
    # Final outputs
    answer: Optional[str]
    answer_stream: Any
    cache_hit: bool
    guardrail_hit: bool
    latency_s: float


def node_check_cache(state: PipelineState):
    if state.get("use_cache"):
        cached = state["cache"].get(state["query"])
        if cached:
            return {
                "cache_hit": True,
                "guardrail_hit": False,
                "latency_s": round(time.perf_counter() - state["start_time"], 2),
                "answer": cached.get("answer"),
                "sources": cached.get("sources", []),
                "rewritten_query": cached.get("rewritten_query")
            }
    return {"cache_hit": False}


def route_after_cache(state: PipelineState):
    return END if state.get("cache_hit") else "node_guardrail"


def node_guardrail(state: PipelineState):
    history_str = state["memory"].as_string()
    is_safe, blocked_msg = state["guardrail"].check(state["query"], history=history_str)
    if not is_safe:
        return {
            "history_str": history_str,
            "guardrail_hit": True,
            "answer": blocked_msg,
            "sources": [],
            "latency_s": round(time.perf_counter() - state["start_time"], 2)
        }
    return {"history_str": history_str, "guardrail_hit": False}


def route_after_guardrail(state: PipelineState):
    return END if state.get("guardrail_hit") else "node_rewrite_query"


def node_rewrite_query(state: PipelineState):
    # history_str is already populated by node_guardrail
    history_str = state.get("history_str", "")
    rewritten = state["rewriter"].rewrite(state["query"], history=history_str)
    return {"rewritten_query": rewritten}


def node_retrieve(state: PipelineState):
    logger.info(f"Retrieving documents for: {state['rewritten_query']!r}")
    search_results = state["retriever"].retrieve(state["rewritten_query"])
    if not search_results:
        answer = (
            "I couldn't find relevant TAMU information for your question. "
            "Please try rephrasing or visit [tamu.edu](https://www.tamu.edu)."
        )
        return {
            "search_results": [],
            "answer": answer,
            "sources": [],
            "latency_s": round(time.perf_counter() - state["start_time"], 2)
        }
    return {"search_results": search_results}


def route_after_retrieve(state: PipelineState):
    return "node_broad_search" if not state.get("search_results") else "node_extract_context"


def node_broad_search(state: PipelineState):
    logger.info(f"Retrieving broader web documents for: {state['rewritten_query']!r}")
    search_results = state["retriever"].retrieve(state["rewritten_query"], broad=True)
    if not search_results:
        answer = (
            "I couldn't find relevant information for your question even with a broad search. "
            "Please try rephrasing."
        )
        return {
            "search_results": [],
            "answer": answer,
            "sources": [],
            "broad_search_used": True,
            "latency_s": round(time.perf_counter() - state["start_time"], 2)
        }
    return {"search_results": search_results, "broad_search_used": True}


def route_after_broad_search(state: PipelineState):
    return END if not state.get("search_results") else "node_extract_context"


def node_extract_context(state: PipelineState):
    logger.info(f"Extracting context from {len(state['search_results'])} documents locally…")
    filtered = []
    for item in state["search_results"][:15]:
        html_content = item.get("metadata", {}).get("content", "")
        cleaned_html = clean_html(html_content)
        item["truncated_html"] = extract_relevant_context(cleaned_html, state["rewritten_query"], window_size=3000)
        filtered.append(item)
    
    # Rerank extracted snippets using Cross-Encoder
    ranked = get_cross_encoder().rank(
        query=state["rewritten_query"],
        documents=filtered,
        text_key="truncated_html",
        top_k=state["config"].RANKER_TOP_K
    )
    context, sources = build_cited_context(ranked)
    return {"context": context, "sources": sources}


def node_generate(state: PipelineState):
    prompt = state["generator"].generate_prompt(
        context=state["context"],
        question=state["query"],
        history=state["history_str"],
    )
    logger.info("Generating final answer…")
    
    updates = {
        "latency_s": round(time.perf_counter() - state["start_time"], 2)
    }
    
    if state["stream"]:
        updates["answer_stream"] = state["generator"].stream(prompt)
    else:
        response = state["generator"].invoke(prompt)
        answer = response.content.strip()
        updates["answer"] = answer
        
        # We do NOT write to cache or memory here anymore.
        # We wait until node_evaluate confirms it is faithful to prevent caching hallucinations.
    
    return updates


def node_evaluate(state: PipelineState):
    def commit_success():
        # Helper to commit to cache and memory when answer is faithful
        if not state["stream"]:
            state["memory"].add_turn(state["query"], state["answer"])
            if state["use_cache"]:
                result_to_cache = {
                    "answer": state["answer"],
                    "sources": state.get("sources", []),
                    "rewritten_query": state.get("rewritten_query"),
                    "cache_hit": True, 
                    "guardrail_hit": False,
                    "latency_s": round(time.perf_counter() - state["start_time"], 2)
                }
                state["cache"].set(state["query"], result_to_cache)
        return {"retry_count": state.get("retry_count", 0), "hallucination_detected": False}

    if not state["config"].USE_REFLECTION:
        return commit_success()
        
    logger.info("Evaluating answer for hallucination...")
    prompt = f"""Context:
{state['context']}

Answer:
{state['answer']}

For EACH factual claim in the answer, find the exact sentence in the Context that supports it.
If you cannot find a supporting sentence for a claim, quote nothing for it and mark it UNSUPPORTED.

Reply in this exact format:
CLAIM: <claim text>
EVIDENCE: <exact quoted sentence from context, or "NONE FOUND">
---
(repeat for each claim)

FINAL: HALLUCINATION or FAITHFUL"""
    
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    judge_model = ChatOpenAI(model=state["config"].FAST_MODEL, temperature=0)
    response = judge_model.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    
    if "FINAL: FAITHFUL" not in decision:
        retry_count = state.get('retry_count', 0) + 1
        logger.warning(f"Hallucination detected (retry {retry_count})")
        if retry_count >= 2:
            # Force exit with safe refusal instead of serving hallucination
            return {
                "retry_count": retry_count,
                "hallucination_detected": True,
                "answer": "I apologize, but I could not verify the accuracy of the information found in the sources. Please try rephrasing your question or check tamu.edu directly."
            }
        return {"retry_count": retry_count, "hallucination_detected": True}
        
    return commit_success()


def route_after_evaluate(state: PipelineState):
    if not state["config"].USE_REFLECTION:
        return END
        
    if not state.get("hallucination_detected"):
        return END
        
    retry_count = state.get("retry_count", 0)
    if retry_count < 2:
        return "node_rewrite_query"
    return END


# Compile Graph
def build_pipeline_graph():
    builder = StateGraph(PipelineState)
    
    builder.add_node("node_check_cache", node_check_cache)
    builder.add_node("node_rewrite_query", node_rewrite_query)
    builder.add_node("node_guardrail", node_guardrail)
    builder.add_node("node_retrieve", node_retrieve)
    builder.add_node("node_broad_search", node_broad_search)
    builder.add_node("node_extract_context", node_extract_context)
    builder.add_node("node_generate", node_generate)
    builder.add_node("node_evaluate", node_evaluate)
    
    builder.set_entry_point("node_check_cache")
    
    builder.add_conditional_edges("node_check_cache", route_after_cache, {END: END, "node_guardrail": "node_guardrail"})
    builder.add_conditional_edges("node_guardrail", route_after_guardrail, {END: END, "node_rewrite_query": "node_rewrite_query"})
    builder.add_edge("node_rewrite_query", "node_retrieve")
    builder.add_conditional_edges("node_retrieve", route_after_retrieve, {"node_broad_search": "node_broad_search", "node_extract_context": "node_extract_context"})
    builder.add_conditional_edges("node_broad_search", route_after_broad_search, {END: END, "node_extract_context": "node_extract_context"})
    builder.add_edge("node_extract_context", "node_generate")
    builder.add_edge("node_generate", "node_evaluate")
    builder.add_conditional_edges("node_evaluate", route_after_evaluate, {END: END, "node_rewrite_query": "node_rewrite_query"})
    
    return builder.compile()

# Global graph instance
pipeline_graph = build_pipeline_graph()


# ── Core pipeline wrapper ─────────────────────────────────────────────────────

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
    stream: bool = False,
) -> dict:
    """
    Wrapper function to maintain compatibility with app.py.
    Executes the underlying LangGraph state machine.
    """
    initial_state = {
        "query": query,
        "config": config,
        "cache": cache,
        "memory": memory,
        "guardrail": guardrail,
        "rewriter": rewriter,
        "retriever": retriever,
        "generator": generator,
        "use_cache": use_cache,
        "stream": stream,
        "start_time": time.perf_counter(),
        "history_str": "",
        "rewritten_query": None,
        "search_results": None,
        "context": None,
        "sources": [],
        "answer": None,
        "answer_stream": None,
        "cache_hit": False,
        "guardrail_hit": False,
        "latency_s": 0.0,
        "retry_count": 0,
        "broad_search_used": False,
    }
    
    final_state = pipeline_graph.invoke(initial_state)
    
    result = {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "context": final_state.get("context", ""),
        "rewritten_query": final_state.get("rewritten_query") or query,
        "cache_hit": final_state.get("cache_hit", False),
        "guardrail_hit": final_state.get("guardrail_hit", False),
        "latency_s": final_state.get("latency_s", 0.0)
    }
    
    if stream and "answer_stream" in final_state and final_state["answer_stream"]:
        result["answer_stream"] = final_state["answer_stream"]
        
    return result


# ── Component factory ─────────────────────────────────────────────────────────

def build_pipeline_components(config: AppConfig):
    cache = ResponseCache(config.CACHE_DB_PATH)
    memory = ConversationMemory(max_turns=config.MEMORY_MAX_TURNS)
    guardrail = QueryGuardrail(config)
    rewriter = QueryRewriter(config)
    retriever = HybridRetriever(config)
    generator = OpenAILanguageModel(config.TEMPLATE_PATH, model_name=config.STRONG_MODEL)
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
    print(f"Rewritten: {result.get('rewritten_query')}")
    print(f"Cache hit: {result.get('cache_hit')}  |  Latency: {result.get('latency_s')}s")
    print("-"*60)
    print(result.get("answer"))
    print("\nSources:")
    for s in result.get("sources", []):
        print(f"  [{s['index']}] {s['title']}\n      {s['url']}")
    print("="*60)


if __name__ == "__main__":
    main()
