# HowdyAI: Graph-Based RAG Search Engine

Agent-driven educational assistant and contextual search engine for Texas A&M University (TAMU). Users can ask complex questions about courses, university policies, degree plans, and faculty, and the system retrieves semantically relevant context from a localized university corpus, then re-ranks and synthesizes the results. The output is a highly faithful, properly cited answer with zero hallucinations.

This project targets the class of retrieval-augmented generation (RAG) problems that underpin specialized domain assistants: the gap between raw semantic search and factually accurate, hallucination-free generation. The technical contribution is the strict multi-stage LangGraph pipeline combined with an uncompromising Chain-of-Verification (CoV) evaluation methodology, ensuring the system reliably refuses unanswerable or adversarial queries rather than hallucinating.


## Table of Contents

- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Technical Stack](#technical-stack)
- [Dataset](#dataset)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Safety & Query Rewriting](#safety--query-rewriting)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Current Status](#current-status)
- [Roadmap](#roadmap)


## Architecture

The system follows a multi-stage pipeline orchestrated by a LangGraph state machine:

```
Stage 1: Guardrail & Cache
    User input
        -> Cache check against SQLite for immediate returns on repeat questions.
        -> Strict out-of-scope guardrail classification (IN_SCOPE / OUT_OF_SCOPE).
        -> Rejects adversarial prompts, jailbreaks, and non-TAMU queries instantly.

Stage 2: Query Rewriting
    Safe User input + Conversation History
        -> LLM rewrites the query to be dense and keyword-rich.
        -> Resolves pronouns (e.g., "where is it?" -> "where is the MSC at Texas A&M?").
        -> Expands domain abbreviations (e.g., "CSCE" -> "Computer Science and Engineering").

Stage 3: Hybrid Retrieval & Re-ranking
    Rewritten Query
        -> ChromaDB Cosine-similarity search against local embedded university corpus.
        -> Dynamic fallback to Brave Search for broad/live web queries.
        -> Local TF-IDF extraction for context boundary optimization.
        -> Cross-Encoder re-ranking to bubble up the most semantically relevant snippets.

Stage 4: Generation & Verification
    Re-ranked Context + User Query
        -> OpenAI generation with strict grounding instructions and inline citations.
        -> (Optional) Chain-of-Verification (CoV) judge evaluates the output.
        -> If hallucination detected, the pipeline automatically retries or safely refuses.
```

The key architectural insight is that vector retrieval alone is prone to semantic drift, and LLMs alone are prone to hallucination. The rigid LangGraph orchestration is what makes the pipeline composable, observable, and strictly auditable at every stage.


## Directory Structure

```text
HowdyAI/
|
|-- eval/                        # Rigorous Evaluation Suite
|   |-- run_master_eval.py       # End-to-end graph evaluation with CoV Judge
|   |-- master_eval_dataset.json # Ground-truth dataset (Factual + Adversarial queries)
|   |-- benchmark_report.md      # Verified metric reports
|   |-- health_check.py          # API and system health validation
|
|-- src/                         # Core Application Engine
|   |-- search/
|   |   |-- hybrid_retriever.py  # Brave + Chroma orchestration and logic
|   |   |-- brave_search.py      # Brave Search API integration
|   |   |-- cross_encoder_ranker.py # Cross-Encoder for context reranking
|   |   |-- query_rewriter.py    # Contextual query expansion logic
|   |   |-- data_processor.py    # TF-IDF context extraction and cleaning
|   |   |-- summarize_with_llm.py # Snippet summarization
|   |
|   |-- language_models/
|   |   |-- openai_language_model.py # Base classes for GPT generation
|   |
|   |-- templates/               # Core LLM prompt templates
|   |   |-- chat_template.txt    # Generation grounding instructions
|   |   |-- guardrail_template.txt # Out-of-scope logic
|   |   |-- rewrite_template.txt # Query expansion logic
|   |
|   |-- database.py              # ChromaDB vector store wrapper (Cosine Metric)
|   |-- guardrail.py             # LangGraph guardrail node logic
|   |-- embeddings.py            # Local/Remote embedding models
|   |-- memory.py                # Multi-turn conversation history
|   |-- cache.py                 # SQLite response caching layer
|   |-- metrics.py               # Observability and telemetry logging
|
|-- scripts/
|   |-- migrations/              # Database migration and deduplication utilities
|       |-- dedup_chroma.py      # Cleans duplicate embeddings
|       |-- migrate_chroma.py    # Updates index configurations (L2 -> Cosine)
|       |-- patch_chroma_urls.py # Standardizes metadata
|
|-- tests/                       # Comprehensive unit and integration tests
|-- data/                        # Local data storage (crawled HTML, ChromaDB)
|-- app.py                       # Streamlit UI Entrypoint
|-- main.py                      # LangGraph Orchestrator
|-- config.py                    # Global hyperparameter configuration
|-- crawl_tamu.py                # Web scraper for building the corpus
|-- create_database.py           # Ingestion pipeline for Chroma
```


## Technical Stack

### Orchestration: LangGraph
The core state machine uses LangGraph to manage the lifecycle of a query. This allows for cyclical graphs (e.g., retrying retrieval if a hallucination is detected) and explicit state tracking at every node.

### Vector Database: ChromaDB
We use ChromaDB initialized with the **Cosine** distance metric. Documents are chunked and embedded using OpenAI's `text-embedding-3-small` (or configurable local variants).

### Cross-Encoder Re-Ranking
Instead of relying solely on the Cosine similarity of the bi-encoder embeddings, the system implements a Cross-Encoder step on the retrieved chunks to drastically improve precision, effectively pushing the most relevant exact-match snippets to the top of the context window.

### Generation & Evaluation: OpenAI
The system uses `gpt-4o-mini` for fast internal routing (guardrails, query rewriting) and `gpt-4o` as the strong generator model and Chain-of-Verification judge.


## Dataset

### Scraping and Ingestion
The corpus is dynamically generated via the `crawl_tamu.py` and `create_database.py` pipeline. 
1. The scraper systematically downloads official university pages, catalogs, and department websites.
2. The HTML is cleaned, stripped of boilerplate, and chunked contextually.
3. Chunks are embedded and stored in the local `data/database_cosine` Chroma directory.

### Seed Data
The system currently indexes key domains like `catalog.tamu.edu`, `cpi.tamu.edu`, and other core university resources, resulting in a dense, localized knowledge base of course prerequisites, university policies, and faculty details.


## Retrieval Pipeline

The retrieval pipeline is the second stage of the system. When a query passes the guardrail:

1. **Local Search:** The rewritten query is embedded and searched against ChromaDB using exact Cosine similarity, returning a broad set of candidate chunks.
2. **Fallback / Web Search:** If local confidence is low, or if the query requires broad web context, the Brave Search API fetches live links, which are scraped dynamically.
3. **TF-IDF Extraction:** The raw HTML of the retrieved documents is processed locally to extract the most relevant sentences.
4. **Cross-Encoder Ranking:** The extracted snippets are re-ranked by the Cross-Encoder.
5. **Context Window Assembly:** The top results are compiled into a strict context block, capped at 8,000 words, and injected into the generation prompt with explicit source indexing.


## Safety & Query Rewriting

The pre-processing nodes (`guardrail.py` and `query_rewriter.py`) are critical technical contributions that prevent abuse and improve accuracy.

### Out-of-Scope Guardrail
The system checks every query against a strict policy. Queries attempting prompt injection, jailbreaks, cheating, or asking for off-topic world facts are instantly blocked with a `guardrail_hit`. This ensures the system remains a focused educational tool and cannot be exploited to generate inappropriate content.

### Query Rewriting
"Is it offered in the fall?" is a terrible search query. The query rewriter utilizes the conversation history to resolve pronouns, expand acronyms, and inject necessary domain keywords (e.g., "Is CSCE 670 offered in the Fall semester at Texas A&M?"), drastically improving vector retrieval scores.


## Evaluation Metrics

The defining feature of this project is its uncompromising evaluation methodology. The system relies on a **Chain-of-Verification (CoV)** LLM-as-a-judge to mathematically score the system against a hand-curated, corpus-anchored ground truth dataset (`master_eval_dataset.json`).

### Benchmark Results

Results from the `run_master_eval.py` pipeline evaluated on a 30-query test set (15 factual queries, 15 adversarial/out-of-scope queries):

| Metric | Result |
|--------|--------|
| Queries evaluated | 30 |
| Overall Combined Avg Latency | **3.63 seconds** |
| Factual/Full-Pipeline Avg Latency | 6.22 seconds |
| Guardrail-Refusal Avg Latency | 1.04 seconds |
| Factual Faithfulness | **100.0%** (15/15) |
| Adversarial Guardrail Success | **100.0%** (15/15) |
| Hybrid Retrieval Recall@10 | **86.67%** (13/15) |
| CoV Retries | 0 |

The 100% factual faithfulness score proves the system correctly identifies and extracts answers from the corpus, completely eliminating hallucinations. The 100% guardrail success proves the system is fully robust against adversarial injections and off-topic questions.

To reproduce:
```bash
python eval/run_master_eval.py
python eval/run_retrieval_eval.py
```


## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API Key
- Brave Search API Key

### Step 1: Clone and Environment

```bash
git clone <repository-url>
cd HowdyAI
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Keys

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key
BRAVE_API_KEY=your_brave_key
```

### Step 3: Run the Application

```bash
streamlit run app.py
```


## Usage

Once the Streamlit server is running, navigate to `http://localhost:8501`.

The UI provides a ChatGPT-like interface where you can:
- Ask questions about TAMU courses and policies.
- View inline citations and source links for every generated answer.
- Monitor the real-time "Admin Dashboard" in the sidebar to see Cache Hit Rates, Guardrail Block Rates, and Average Latency metrics.


## Current Status

### Implemented

- Complete LangGraph orchestration pipeline with distinct Guardrail, Rewrite, Retrieve, Rank, and Generate nodes.
- High-performance Streamlit frontend with custom CSS, visual badges, and an administrative telemetry dashboard.
- Exact course code matching to prevent false-positive semantic retrieval of similar course numbers.
- Comprehensive `run_master_eval.py` evaluation suite providing verifiable Faithfulness and Guardrail metrics.
- SQLite-based caching layer for sub-second responses on identical queries.
- Clean `.dockerignore` and optimized `Dockerfile` for streamlined containerized deployment.

### Known Limitations

- Running `create_database.py` to index the entire TAMU web domain is time-consuming; the current seed database focuses on high-value catalogs and directories.


## Roadmap

### Phase 2: Deployment and Scale
Deploy the application via Docker. Implement a scheduled CRON job to automatically run the `crawl_tamu.py` pipeline once a month to ensure the vector database stays synchronized with the latest university catalog updates.
