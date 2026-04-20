# HowdyAI — Enhanced RAG Chatbot for Texas A&M University

**CSCE 670 · Information Retrieval · Texas A&M University**  
*By Pradeep Periyasamy (437005610)*

---

## Overview

HowdyAI is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Texas A&M University, grounded in verifiable sources with clickable inline citations.

### Enhancements

| # | Enhancement | File(s) |
|---|------------|---------|
| 1 | **Hybrid retrieval** — Activates the existing unused Chroma vector store and fuses it with Google Search via Reciprocal Rank Fusion (RRF) | `src/search/hybrid_retriever.py`, `src/database.py` |
| 2 | **LLM query rewriting** — Reformulates user queries with TAMU-specific terminology before retrieval | `src/search/query_rewriter.py`, `src/templates/rewrite_template.txt` |
| 3 | **Semantic chunking** — Replaces the fixed `RecursiveCharacterTextSplitter(chunk_size=300)` with embedding-similarity-based chunking | `create_database.py` |
| 4 | **Conversation memory** — Prepends prior turns to the context for coherent multi-turn follow-ups | `src/memory.py` |
| 5 | **Inline citations** — Every answer includes `[1]`, `[2]` references linked to source URLs | `main.py` (`build_cited_context`), `src/templates/chat_template.txt` |
| 6 | **Out-of-scope guardrail** — Lightweight LLM classifier blocks non-TAMU queries before the full pipeline runs | `src/guardrail.py`, `src/templates/guardrail_template.txt` |
| 7 | **SQLite response cache** — SHA-256-keyed cache returns stored answers instantly for repeated queries | `src/cache.py` |
| 8 | **Streamlit UI** — Replaces the argparse CLI with a full web interface showing citations, badges, and sidebar controls | `app.py` |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         HowdyAI Pipeline                        │
│                                                                 │
│  ① Guardrail ──► OUT_OF_SCOPE? ──► Rejection message           │
│       │                                                         │
│       ▼                                                         │
│  ② Cache lookup ──► HIT? ──► Return cached answer              │
│       │                                                         │
│       ▼                                                         │
│  ③ Query Rewriter  (gpt-4o-mini)                               │
│       │  rewritten query                                        │
│       ▼                                                         │
│  ④ Hybrid Retriever                                             │
│    ┌──────────────────┐    ┌──────────────────┐                │
│    │  Google Custom   │    │  Chroma Dense    │                │
│    │  Search (top-5)  │    │  Search  (top-5) │                │
│    └────────┬─────────┘    └────────┬─────────┘                │
│             └──────── RRF ──────────┘                          │
│                  fused top-8                                    │
│       │                                                         │
│       ▼                                                         │
│  ⑤ Summarizer  (gpt-4o-mini, parallel)                        │
│       │  per-doc summaries                                      │
│       ▼                                                         │
│  ⑥ Re-ranker   (gpt-4o-mini)                                  │
│       │  top-5 ranked docs                                      │
│       ▼                                                         │
│  ⑦ Generator   (gpt-4o)  + conversation history               │
│       │  answer with [1][2]… inline citations                  │
│       ▼                                                         │
│  ⑧ Memory update + Cache write                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Answer + Clickable Sources
```

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd HowdyAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Fill in:
#   OPENAI_API_KEY    — from platform.openai.com
#   GOOGLE_CSE_ID     — from cse.google.com
#   GOOGLE_CSE_API_KEY — from console.cloud.google.com
```

### 3. Crawl TAMU content (optional but recommended)

```bash
cd crawler
scrapy crawl tamu -s CLOSESPIDER_PAGECOUNT=500
cd ..
```

Crawled HTML files will be saved under `data/documents/`.

### 4. Build the Chroma vector store

```bash
python create_database.py           # incremental build
python create_database.py --reset   # wipe and rebuild from scratch
```

### 5. Launch the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 6. Or run the CLI

```bash
python main.py "What are the CSCE 670 prerequisites?"
```

---

## Project Structure

```
HowdyAI/
├── app.py                          # Streamlit UI (NEW)
├── main.py                         # Enhanced pipeline orchestrator
├── create_database.py              # Semantic chunking + Chroma indexing (ENHANCED)
├── config.py                       # App config (EXTENDED)
├── requirements.txt
├── .env.example
│
├── src/
│   ├── cache.py                    # SQLite response cache (NEW)
│   ├── memory.py                   # Conversation memory (NEW)
│   ├── guardrail.py                # Out-of-scope guard (NEW)
│   ├── database.py                 # Chroma DB wrapper (UPDATED — now active)
│   ├── embeddings.py               # OpenAI text-embedding-3-large (base)
│   │
│   ├── search/
│   │   ├── hybrid_retriever.py     # Google + Chroma + RRF fusion (NEW)
│   │   ├── query_rewriter.py       # LLM query rewriting (NEW)
│   │   ├── google_search.py        # Google Custom Search (base)
│   │   ├── summarize_with_llm.py   # Per-doc summarization (base)
│   │   ├── rank_with_llm.py        # LLM re-ranking (base)
│   │   └── data_processor.py       # HTML cleaning + NLTK (base)
│   │
│   ├── language_models/
│   │   ├── language_model.py       # Abstract base (base)
│   │   └── openai_language_model.py # ChatOpenAI wrapper (base)
│   │
│   └── templates/
│       ├── chat_template.txt       # Answer generation prompt (ENHANCED — citations + history)
│       ├── summary_template.txt    # Summarization prompt (base)
│       ├── rank_template.txt       # Ranking prompt (base)
│       ├── rewrite_template.txt    # Query rewriting prompt (NEW)
│       └── guardrail_template.txt  # Out-of-scope classifier prompt (NEW)
│
├── crawler/
│   └── crawler/spiders/tamu.py    # Scrapy spider (base)
│
└── data/
    ├── documents/                  # HTML files for Chroma indexing
    ├── database/                   # Chroma vector store (persisted)
    └── samples/                    # Sample search results
```

---

## Evaluation Plan

| Metric | Description |
|--------|-------------|
| Answer Accuracy | Correct / Partially correct / Incorrect on 50–100 curated TAMU queries |
| Retrieval Recall@K | Fraction of relevant docs retrieved in top-K |
| nDCG@10 | Normalized discounted cumulative gain |
| Faithfulness | % of claims grounded in retrieved context |
| Citation Accuracy | % of inline citations pointing to the correct source |
| Latency p50/p95 | With and without cache |
| Guardrail Precision | % of out-of-scope queries correctly blocked |


---

## References

- LangChain: [python.langchain.com](https://python.langchain.com)
- ChromaDB: [trychroma.com](https://www.trychroma.com)
- OpenAI Embeddings (`text-embedding-3-large`): [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
- Reciprocal Rank Fusion: Cormack et al., SIGIR 2009
- Scrapy: [scrapy.org](https://scrapy.org)
- Streamlit: [streamlit.io](https://streamlit.io)
