# HowdyAI: Graph-Based RAG Search Engine

<p align="left">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/praddep/HowdyAI/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat" alt="PRs Welcome"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/hhatto/autopep8"><img src="https://img.shields.io/badge/code%20style-autopep8-lightgrey.svg" alt="Code Style: autopep8"></a>
  <br>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat" alt="ChromaDB"></a>
  <a href="https://brave.com/search/api/"><img src="https://img.shields.io/badge/Brave_Search-FB542B?style=flat&logo=brave&logoColor=white" alt="Brave Search"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-000000?style=flat" alt="LangGraph"></a>
</p>

Agent-driven educational assistant and contextual search engine for Texas A&M University (TAMU). Users can ask complex questions about courses, university policies, degree plans, and faculty, and the system retrieves semantically relevant context from a localized university corpus, then re-ranks and synthesizes the results. The output is a highly faithful, properly cited answer with zero hallucinations.

This project targets the class of retrieval-augmented generation (RAG) problems that underpin specialized domain assistants: the gap between raw semantic search and factually accurate, hallucination-free generation. The technical contribution is the strict multi-stage LangGraph pipeline combined with an uncompromising Chain-of-Verification (CoV) evaluation methodology, ensuring the system reliably refuses unanswerable or adversarial queries rather than hallucinating.

## Table of Contents

- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Technical Stack](#technical-stack)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Retrieval Pipeline](#retrieval-pipeline)
- [LLM Prompt Templates](#llm-prompt-templates)
- [Safety & Query Rewriting](#safety--query-rewriting)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Docker Deployment](#docker-deployment)
- [CI/CD & Code Quality](#cicd--code-quality)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Current Status](#current-status)
- [Roadmap](#roadmap)
- [Community & Policies](#community--policies)
- [Citation](#citation)


## Architecture

The system follows a multi-stage pipeline orchestrated by a LangGraph state machine. Each stage is a discrete, observable node — making the pipeline composable, auditable, and safe.

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

### End-to-End Pipeline Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'primaryBorderColor': '#7a0000', 'lineColor': '#c09000', 'secondaryColor': '#1a1a2e', 'tertiaryColor': '#2c2c4a', 'background': '#0f0f1e', 'mainBkg': '#1a1a2e', 'nodeBorder': '#c09000', 'clusterBkg': '#2c2c4a', 'titleColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart TD
    USER(["👤 User Query"]):::user
    CACHE{"🗄️ Cache Hit?"}:::decision
    GUARD{"🛡️ Guardrail\nIN_SCOPE?"}:::decision
    REWRITE["✍️ Query Rewriter\ngpt-4o-mini"]:::process
    RETRIEVE["🔍 Hybrid Retriever\nChroma + Brave Search"]:::process
    BROAD{"📡 Results\nFound?"}:::decision
    BROADSRCH["🌐 Broad Web\nSearch Fallback"]:::process
    EXTRACT["⚙️ Context Extraction\nTF-IDF + Clean HTML"]:::process
    RERANK["📊 Cross-Encoder\nRe-ranking"]:::process
    GENERATE["🤖 GPT-4o\nGeneration + Citations"]:::process
    EVALUATE{"🔬 CoV Judge\nFaithful?"}:::decision
    RETRY{"> 2 retries?"}:::decision
    SAFE["⚠️ Safe Refusal"]:::danger
    BLOCKED["🚫 Blocked\nResponse"]:::danger
    CACHE_RESP["⚡ Cached\nResponse"]:::success
    ANSWER["✅ Final Answer\n+ Sources"]:::success
    MEMORY["💾 Update Memory\n& Write Cache"]:::storage

    USER --> CACHE
    CACHE -->|"Hit"| CACHE_RESP
    CACHE -->|"Miss"| GUARD
    GUARD -->|"OUT_OF_SCOPE"| BLOCKED
    GUARD -->|"IN_SCOPE"| REWRITE
    REWRITE --> RETRIEVE
    RETRIEVE --> BROAD
    BROAD -->|"No results"| BROADSRCH
    BROAD -->|"Results"| EXTRACT
    BROADSRCH -->|"No results"| SAFE
    BROADSRCH -->|"Results"| EXTRACT
    EXTRACT --> RERANK
    RERANK --> GENERATE
    GENERATE --> EVALUATE
    EVALUATE -->|"FAITHFUL"| MEMORY
    EVALUATE -->|"HALLUCINATION"| RETRY
    RETRY -->|"< 2"| REWRITE
    RETRY -->|">= 2"| SAFE
    MEMORY --> ANSWER

    classDef user fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef process fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef decision fill:#2c2c4a,stroke:#c09000,color:#fff,font-weight:bold
    classDef success fill:#2d6a4f,stroke:#40916c,color:#fff,font-weight:bold
    classDef danger fill:#7f1d1d,stroke:#ef4444,color:#fff
    classDef storage fill:#374151,stroke:#9ca3af,color:#e2e8f0
```

### LangGraph State Machine

Exact node graph as compiled by `build_pipeline_graph()` in `main.py`.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
stateDiagram-v2
    direction LR
    [*] --> node_check_cache

    node_check_cache --> END : cache_hit = True
    node_check_cache --> node_guardrail : cache_hit = False

    node_guardrail --> END : guardrail_hit = True
    node_guardrail --> node_rewrite_query : guardrail_hit = False

    node_rewrite_query --> node_retrieve

    node_retrieve --> node_broad_search : no results
    node_retrieve --> node_extract_context : results found

    node_broad_search --> END : no results
    node_broad_search --> node_extract_context : results found

    node_extract_context --> node_generate
    node_generate --> node_evaluate

    node_evaluate --> END : faithful / reflection OFF
    node_evaluate --> node_rewrite_query : hallucination & retry < 2
    node_evaluate --> END : hallucination & retry >= 2 → safe refusal

    END --> [*]
```

### C4 Context — System Ecosystem

High-level view of HowdyAI in its broader ecosystem of users, internal components, and external services.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart TB
    USER(["👤 TAMU Student\nor Researcher"]):::person

    subgraph SYS ["  HowdyAI System  "]
        UI["Streamlit Web UI\nport 8501"]:::internal
        PIPELINE["LangGraph Pipeline\nmain.py"]:::internal
        CACHE_COMP["SQLite Cache\nhowdyai_cache.db"]:::internal
        VECTOR["ChromaDB\ndata/database_cosine"]:::internal
    end

    subgraph EXTERNAL ["  External Services  "]
        OPENAI["☁️ OpenAI API\ngpt-4o / gpt-4o-mini\ntext-embedding-3-small"]:::external
        BRAVE_EXT["🌐 Brave Search API\nLive web results"]:::external
        TAMU_WEB["🏫 TAMU Websites\ncatalog.tamu.edu\ncpi.tamu.edu"]:::external
    end

    USER -->|"Natural language\nquestion"| UI
    UI -->|"run_pipeline()"| PIPELINE
    PIPELINE -->|"Cache read/write"| CACHE_COMP
    PIPELINE -->|"Similarity search"| VECTOR
    PIPELINE -->|"Guardrail, Rewrite,\nGenerate, CoV Judge"| OPENAI
    PIPELINE -->|"Web fallback\nqueries"| BRAVE_EXT
    TAMU_WEB -->|"crawl_tamu.py\n(offline ingestion)"| VECTOR
    UI -->|"Cited answer\n+ latency metrics"| USER

    classDef person fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef internal fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef external fill:#1e3a5f,stroke:#3b82f6,color:#bfdbfe
```


## Directory Structure

```text
HowdyAI/
|
|-- .github/                         # GitHub Configuration
|   |-- workflows/
|   |   |-- ci.yml                   # GitHub Actions CI pipeline (lint + test)
|   |-- ISSUE_TEMPLATE/
|   |   |-- bug_report.md            # Bug report template
|   |   |-- feature_request.md       # Feature request template
|   |-- PULL_REQUEST_TEMPLATE.md     # PR description template
|
|-- eval/                            # Rigorous Evaluation Suite
|   |-- run_master_eval.py           # End-to-end graph evaluation with CoV Judge
|   |-- run_retrieval_eval.py        # Standalone hybrid retrieval recall evaluation
|   |-- master_eval_dataset.json     # Ground-truth dataset (Factual + Adversarial queries)
|   |-- retrieval_qa_pairs.json      # QA pairs for retrieval-only evaluation
|   |-- benchmark_report.md          # Verified metric reports
|   |-- health_check.py              # API and system health validation
|
|-- src/                             # Core Application Engine
|   |-- search/
|   |   |-- hybrid_retriever.py      # Brave + Chroma orchestration and logic
|   |   |-- brave_search.py          # Brave Search API integration
|   |   |-- cross_encoder_ranker.py  # Cross-Encoder for context reranking
|   |   |-- query_rewriter.py        # Contextual query expansion logic
|   |   |-- data_processor.py        # TF-IDF context extraction and cleaning
|   |   |-- summarize_with_llm.py    # Snippet summarization
|   |
|   |-- language_models/
|   |   |-- language_model.py        # Abstract base class for LLMs
|   |   |-- openai_language_model.py # GPT-4o / GPT-4o-mini implementation
|   |
|   |-- templates/                   # LLM prompt templates
|   |   |-- chat_template.txt        # Main generation grounding instructions
|   |   |-- guardrail_template.txt   # Out-of-scope classification prompt
|   |   |-- rewrite_template.txt     # Query expansion prompt
|   |   |-- summary_template.txt     # Snippet summarization prompt
|   |   |-- rank_template.txt        # Cross-encoder ranking prompt
|   |
|   |-- database.py                  # ChromaDB vector store wrapper (Cosine Metric)
|   |-- guardrail.py                 # LangGraph guardrail node logic
|   |-- embeddings.py                # Local/Remote embedding models
|   |-- memory.py                    # Multi-turn conversation history (6-turn window)
|   |-- cache.py                     # SQLite response caching layer
|   |-- metrics.py                   # Observability and telemetry logging
|
|-- scripts/
|   |-- migrations/                  # Database migration and deduplication utilities
|       |-- dedup_chroma.py          # Cleans duplicate embeddings from the vector store
|       |-- migrate_chroma.py        # Updates index configurations (L2 -> Cosine)
|       |-- patch_chroma_urls.py     # Standardizes source URL metadata
|
|-- tests/                           # Comprehensive unit and integration tests
|   |-- test_cache.py                # SQLite cache layer tests
|   |-- test_chunking.py             # Document chunking tests
|   |-- test_database.py             # ChromaDB wrapper tests
|   |-- test_embeddings.py           # Embedding model tests
|   |-- test_guardrail.py            # Guardrail classification tests
|   |-- test_hybrid_retriever.py     # Hybrid retriever unit tests
|   |-- test_hybrid_retriever_coverage.py # Extended retriever coverage tests
|   |-- test_integration.py          # End-to-end pipeline integration tests
|   |-- test_memory.py               # Conversation memory tests
|   |-- test_metrics.py              # Telemetry and observability tests
|   |-- test_openai_language_model.py # OpenAI LLM wrapper tests
|   |-- test_query_rewriter.py       # Query rewriting tests
|   |-- test_summarize_with_llm.py   # Summarization tests
|
|-- data/                            # Local data storage (crawled HTML, ChromaDB index)
|-- logs/                            # Runtime logs (debug.log, info.log, error.log)
|-- app.py                           # Streamlit UI Entrypoint
|-- main.py                          # LangGraph Orchestrator (pipeline state machine)
|-- config.py                        # Global hyperparameter configuration (AppConfig)
|-- crawl_tamu.py                    # Web scraper for building the university corpus
|-- create_database.py               # Ingestion pipeline for ChromaDB
|-- Dockerfile                       # Docker container definition
|-- .dockerignore                    # Files excluded from Docker build context
|-- .env.example                     # Example environment variable template
|-- .pre-commit-config.yaml          # Pre-commit hooks (autopep8, ruff, trailing whitespace)
|-- requirements.txt                 # Python package dependencies
|-- CHANGELOG.md                     # Version history and notable changes
|-- CITATION.cff                     # Machine-readable citation metadata
|-- LICENSE                          # MIT License
```

### Project Mind Map

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
mindmap
  root((🤠 HowdyAI))
    📄 Entrypoints
      app.py
      main.py
      config.py
    🔧 src
      search
        hybrid_retriever.py
        brave_search.py
        cross_encoder_ranker.py
        query_rewriter.py
        data_processor.py
        summarize_with_llm.py
      language_models
        language_model.py
        openai_language_model.py
      templates
        chat_template.txt
        guardrail_template.txt
        rewrite_template.txt
        summary_template.txt
        rank_template.txt
      cache.py
      database.py
      embeddings.py
      guardrail.py
      memory.py
      metrics.py
    🔬 eval
      run_master_eval.py
      run_retrieval_eval.py
      health_check.py
      master_eval_dataset.json
      retrieval_qa_pairs.json
      benchmark_report.md
    🧪 tests
      test_cache.py
      test_chunking.py
      test_database.py
      test_embeddings.py
      test_guardrail.py
      test_hybrid_retriever.py
      test_integration.py
      test_memory.py
      test_metrics.py
      test_openai_language_model.py
      test_query_rewriter.py
      test_summarize_with_llm.py
    🛠️ scripts
      migrations
        dedup_chroma.py
        migrate_chroma.py
        patch_chroma_urls.py
    🐙 .github
      workflows
        ci.yml
      ISSUE_TEMPLATE
        bug_report.md
        feature_request.md
      PULL_REQUEST_TEMPLATE.md
    🐳 Docker
      Dockerfile
      .dockerignore
    📚 Docs
      README.md
      CHANGELOG.md
      CONTRIBUTING.md
      CODE_OF_CONDUCT.md
      SECURITY.md
      SUPPORT.md
      LICENSE
      CITATION.cff
    📦 Data
      crawl_tamu.py
      create_database.py
      data
      logs
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

### Web Search Fallback: Brave Search API
When local vector retrieval confidence is low, or a query requires live web context, the Brave Search API is invoked to fetch and scrape fresh web pages dynamically.

### Frontend: Streamlit
A ChatGPT-like interface with custom CSS, visual status badges, inline citations, and a real-time Admin Dashboard for monitoring cache hit rates, guardrail block rates, and average latency.

### Module Dependency Graph

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart LR
    subgraph ENTRY ["  Entrypoints  "]
        APP["app.py\nStreamlit UI"]:::entry
        MAIN["main.py\nLangGraph Orchestrator"]:::entry
    end

    subgraph CFG ["  Config  "]
        CONFIG["config.py\nAppConfig"]:::config
    end

    subgraph CORE ["  src/ — Core Modules  "]
        CACHE["cache.py\nResponseCache"]:::module
        GUARD["guardrail.py\nQueryGuardrail"]:::module
        MEM["memory.py\nConversationMemory"]:::module
        DB["database.py\nChromaDB Wrapper"]:::module
        EMB["embeddings.py\nEmbedding Models"]:::module
        MET["metrics.py\nTelemetry"]:::module
    end

    subgraph SEARCH ["  src/search/  "]
        HR["hybrid_retriever.py"]:::module
        QR["query_rewriter.py"]:::module
        BS["brave_search.py"]:::module
        CE["cross_encoder_ranker.py"]:::module
        DP["data_processor.py"]:::module
        SL["summarize_with_llm.py"]:::module
    end

    subgraph LM ["  src/language_models/  "]
        BASE["language_model.py\n(Abstract)"]:::abstract
        OAI["openai_language_model.py\nGPT-4o wrapper"]:::module
    end

    subgraph TEMPLATES ["  src/templates/  "]
        CT["chat_template.txt"]:::template
        GT["guardrail_template.txt"]:::template
        RT["rewrite_template.txt"]:::template
        ST["summary_template.txt"]:::template
        RKT["rank_template.txt"]:::template
    end

    APP --> MAIN
    APP --> CONFIG
    MAIN --> CONFIG
    MAIN --> CACHE
    MAIN --> GUARD
    MAIN --> MEM
    MAIN --> QR
    MAIN --> HR
    MAIN --> OAI
    MAIN --> CE

    HR --> DB
    HR --> BS
    HR --> EMB

    OAI --> BASE
    OAI --> CT

    GUARD --> GT
    QR --> RT
    SL --> ST
    CE --> RKT

    DB --> EMB
    CONFIG --> CACHE
    CONFIG --> GUARD
    CONFIG --> QR
    CONFIG --> HR

    classDef entry fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef config fill:#78350f,stroke:#f59e0b,color:#fff,font-weight:bold
    classDef module fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef abstract fill:#2c2c4a,stroke:#9ca3af,color:#d1d5db,font-style:italic
    classDef template fill:#1e3a5f,stroke:#3b82f6,color:#bfdbfe
```


## Configuration

All system-wide hyperparameters are managed through the `AppConfig` class in `config.py`. The key parameters are:

| Parameter | Default | Description |
|---|---|---|
| `FAST_MODEL` | `gpt-4o-mini` | Model for guardrails and query rewriting |
| `STRONG_MODEL` | `gpt-4o` | Model for generation and CoV judge |
| `CHROMA_NUM_RESULTS` | `10` | Number of candidate chunks from ChromaDB |
| `NUM_SEARCH_RESULTS` | `5` | Number of results from Brave Search |
| `FUSION_TOP_N` | `12` | Top candidates passed to the Cross-Encoder |
| `RANKER_TOP_K` | `8` | Final top-K snippets after re-ranking |
| `MEMORY_MAX_TURNS` | `6` | Maximum conversation turns stored in memory |
| `USE_REFLECTION` | `False` | Enable/disable CoV retry loop |
| `USE_HYDE` | `False` | Enable/disable HyDE query expansion |

Environment variables are loaded from a `.env` file. See `.env.example` for the required keys.


## Dataset

### Scraping and Ingestion
The corpus is dynamically generated via the `crawl_tamu.py` and `create_database.py` pipeline.
1. The scraper systematically downloads official university pages, catalogs, and department websites.
2. The HTML is cleaned, stripped of boilerplate, and chunked contextually.
3. Chunks are embedded and stored in the local `data/database_cosine` Chroma directory.

### Seed Data
The system currently indexes key domains like `catalog.tamu.edu`, `cpi.tamu.edu`, and other core university resources, resulting in a dense, localized knowledge base of course prerequisites, university policies, and faculty details.

### Data Flow & Storage Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart TD
    subgraph INGEST ["  📥  Corpus Ingestion (Offline)  "]
        CRAWL["crawl_tamu.py\ncatalog.tamu.edu\ncpi.tamu.edu"]:::process
        CREATE["create_database.py\nChunk → Embed → Ingest"]:::process
        CHROMA_STORE[("data/database_cosine\nChromaDB\nCosine Metric")]:::store
    end

    subgraph RUNTIME ["  ⚙️  Runtime Pipeline  "]
        QUERY["User Query"]:::input
        SQLITE[("howdyai_cache.db\nSQLite Cache")]:::store
        MEMORY_STORE["In-Memory\n6-turn History"]:::store
        BRAVE_LIVE["Brave Search API\nLive Web Pages"]:::external
        CHROMA_READ[("ChromaDB\nRead-only")]:::store
        LOGS[("logs/\ndebug.log\ninfo.log\nerror.log")]:::store
    end

    subgraph EVAL ["  🔬  Evaluation (Offline)  "]
        DATASET["master_eval_dataset.json\n30 ground-truth queries"]:::data
        RET_PAIRS["retrieval_qa_pairs.json\nRetrieval QA pairs"]:::data
        REPORT["benchmark_report.md\nVerified Metrics"]:::data
    end

    CRAWL -->|"HTML pages"| CREATE
    CREATE -->|"Embeddings + Metadata"| CHROMA_STORE

    QUERY -->|"Cache lookup"| SQLITE
    QUERY -->|"Read history"| MEMORY_STORE
    QUERY -->|"Dense search"| CHROMA_READ
    QUERY -->|"Web fallback"| BRAVE_LIVE

    SQLITE -->|"Cache hit → response"| QUERY
    CHROMA_READ -->|"Chunks + scores"| QUERY
    BRAVE_LIVE -->|"Scraped content"| QUERY

    QUERY -->|"Write answer"| SQLITE
    QUERY -->|"Write turn"| MEMORY_STORE
    QUERY -->|"Structured logs"| LOGS

    DATASET -->|"run_master_eval.py"| REPORT
    RET_PAIRS -->|"run_retrieval_eval.py"| REPORT

    classDef process fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef store fill:#2c2c4a,stroke:#9ca3af,color:#e2e8f0
    classDef input fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef external fill:#1e3a5f,stroke:#3b82f6,color:#bfdbfe
    classDef data fill:#374151,stroke:#9ca3af,color:#e2e8f0
```

### Entity Relationship Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
erDiagram
    QUERY_CACHE {
        TEXT   query_hash  PK   "SHA-256 of query"
        TEXT   query            "Original user query"
        TEXT   answer           "Generated answer text"
        TEXT   sources_json     "JSON array of source objects"
        TEXT   rewritten_query  "LLM-rewritten query"
        REAL   latency_s        "Response latency in seconds"
        TEXT   created_at       "ISO-8601 timestamp"
    }

    CHROMA_COLLECTION {
        TEXT   id          PK   "UUID document ID"
        TEXT   content          "Raw chunk text (page_content)"
        REAL   embedding[]      "Float32 vector (1536-dim)"
        TEXT   source           "Source file path"
        TEXT   url              "Canonical web URL"
        TEXT   title            "Page title"
        TEXT   chunk_index      "Position of chunk in document"
    }

    CONVERSATION_TURN {
        INT    turn_index  PK   "Turn number (in-memory only)"
        TEXT   user_msg         "User query text"
        TEXT   assistant_msg    "Assistant answer text"
    }

    PIPELINE_LOG {
        TEXT   timestamp        "ISO-8601 log time"
        TEXT   level            "DEBUG / INFO / WARNING / ERROR"
        TEXT   module           "Python module name"
        TEXT   message          "Log message content"
    }

    QUERY_CACHE ||--o{ CHROMA_COLLECTION : "retrieved from"
    QUERY_CACHE ||--o| CONVERSATION_TURN : "updates"
    PIPELINE_LOG }o--|| QUERY_CACHE : "traces"
```


## Retrieval Pipeline

The retrieval pipeline is the second stage of the system. When a query passes the guardrail:

1. **Local Search:** The rewritten query is embedded and searched against ChromaDB using exact Cosine similarity, returning a broad set of candidate chunks.
2. **Fallback / Web Search:** If local confidence is low, or if the query requires broad web context, the Brave Search API fetches live links, which are scraped dynamically.
3. **TF-IDF Extraction:** The raw HTML of the retrieved documents is processed locally to extract the most relevant sentences.
4. **Cross-Encoder Ranking:** The extracted snippets are re-ranked by the Cross-Encoder.
5. **Context Window Assembly:** The top results are compiled into a strict context block, capped at 8,000 words, and injected into the generation prompt with explicit source indexing.

### Hybrid Retrieval Sub-Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#2c2c4a', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart TD
    Q(["🔤 Rewritten Query"]):::input

    subgraph CHROMA ["  🗃️  ChromaDB Dense Retrieval  "]
        HYDE{"USE_HYDE\nenabled?"}:::decision
        GEN_HYDE["Generate HyDE\nHypothetical Document"]:::process
        SEARCH_CHROMA["Cosine Similarity Search\nk = CHROMA_NUM_RESULTS"]:::process
        SCORE{"Top Score\n≥ 0.60?"}:::decision
        CODE{"Course Code\nMatch?"}:::decision
        HIGH_CONF["✅ High Confidence\n(skip web search)"]:::success
        LOW_CONF["📉 Low Confidence\n(augment with web)"]:::warn
    end

    subgraph WEB ["  🌐  Brave Web Search  "]
        BRAVE["Brave Search API\nn = NUM_SEARCH_RESULTS"]:::process
        SCRAPE["Fetch & Scrape\nLive Web Pages"]:::process
    end

    subgraph FUSION ["  ⚡  Reciprocal Rank Fusion  "]
        RRF["RRF Merge\nScore = Σ 1/(k + rank)"]:::process
        TOP_N["Top FUSION_TOP_N\nCandidates"]:::process
    end

    Q --> HYDE
    HYDE -->|"Yes"| GEN_HYDE --> SEARCH_CHROMA
    HYDE -->|"No"| SEARCH_CHROMA
    SEARCH_CHROMA --> SCORE
    SCORE -->|"≥ 0.60"| CODE
    CODE -->|"Match"| HIGH_CONF
    CODE -->|"Mismatch"| LOW_CONF
    SCORE -->|"< 0.60"| LOW_CONF

    HIGH_CONF -->|"Chroma only"| TOP_N
    LOW_CONF --> BRAVE
    BRAVE --> SCRAPE
    SCRAPE -->|"+ Chroma results"| RRF --> TOP_N

    TOP_N -->|"→ TF-IDF Extraction"| OUT(["📄 Fused Result List"]):::output

    classDef input fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef output fill:#2d6a4f,stroke:#40916c,color:#fff,font-weight:bold
    classDef process fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef decision fill:#2c2c4a,stroke:#c09000,color:#fff
    classDef success fill:#2d6a4f,stroke:#40916c,color:#fff
    classDef warn fill:#78350f,stroke:#f59e0b,color:#fff
```

### User Interaction Sequence

End-to-end sequence for a typical factual query, showing component interactions.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'actorBkg': '#1a1a2e', 'actorBorder': '#c09000', 'actorTextColor': '#fff', 'noteBkgColor': '#2c2c4a', 'noteTextColor': '#e2e8f0', 'fontFamily': 'Inter, sans-serif'}}}%%
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Graph as LangGraph Pipeline
    participant Cache as SQLite Cache
    participant Guard as Guardrail (gpt-4o-mini)
    participant Rewriter as Query Rewriter (gpt-4o-mini)
    participant Retriever as Hybrid Retriever
    participant Chroma as ChromaDB
    participant Brave as Brave Search API
    participant Ranker as Cross-Encoder
    participant LLM as Generator (gpt-4o)

    User->>UI: "What are the prereqs for CSCE 670?"
    UI->>Graph: run_pipeline(query)

    Graph->>Cache: get(query)
    Cache-->>Graph: MISS

    Graph->>Guard: check(query, history)
    Guard-->>Graph: IN_SCOPE ✅

    Graph->>Rewriter: rewrite(query, history)
    Rewriter-->>Graph: "CSCE 670 prerequisites Texas A&M Computer Science"

    Graph->>Retriever: retrieve(rewritten_query)
    Retriever->>Chroma: cosine_similarity(embedding, k=10)
    Chroma-->>Retriever: 10 chunks (top_score=0.82)
    Note over Retriever: Score ≥ 0.60 → High confidence, skip web
    Retriever-->>Graph: 3 high-confidence chunks

    Graph->>Ranker: rank(query, chunks, top_k=8)
    Ranker-->>Graph: Re-ranked snippets

    Graph->>LLM: generate(context, query, history)
    LLM-->>Graph: "CSCE 670 requires... [1][2]"

    Graph->>Cache: set(query, answer)
    Graph->>UI: {answer, sources, latency_s}
    UI-->>User: Streamed answer + citations
```


## LLM Prompt Templates

All prompts are managed as plain-text files in `src/templates/`:

| Template | Purpose |
|---|---|
| `chat_template.txt` | Main generation grounding instructions; enforces citation and hallucination refusal |
| `guardrail_template.txt` | Classifies query as `IN_SCOPE` or `OUT_OF_SCOPE` |
| `rewrite_template.txt` | Expands abbreviated or ambiguous queries using conversation history |
| `summary_template.txt` | Summarizes raw web-scraped snippets for context injection |
| `rank_template.txt` | Prompt for LLM-assisted re-ranking step |


## Safety & Query Rewriting

The pre-processing nodes (`guardrail.py` and `query_rewriter.py`) are critical technical contributions that prevent abuse and improve accuracy.

### Out-of-Scope Guardrail
The system checks every query against a strict policy. Queries attempting prompt injection, jailbreaks, cheating, or asking for off-topic world facts are instantly blocked with a `guardrail_hit`. This ensures the system remains a focused educational tool and cannot be exploited to generate inappropriate content.

### Query Rewriting
"Is it offered in the fall?" is a terrible search query. The query rewriter utilizes the conversation history to resolve pronouns, expand acronyms, and inject necessary domain keywords (e.g., "Is CSCE 670 offered in the Fall semester at Texas A&M?"), drastically improving vector retrieval scores.


## Evaluation

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

See the full breakdown in [`eval/benchmark_report.md`](eval/benchmark_report.md).

To reproduce:
```bash
python eval/run_master_eval.py
python eval/run_retrieval_eval.py
```

### Evaluation Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#2c2c4a', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart TD
    DS(["📋 master_eval_dataset.json\n30 queries"]):::input

    subgraph FACTUAL ["  ✅  Factual Queries (15)  "]
        F_PIPE["Full Pipeline\nGuardrail → Rewrite → Retrieve\n→ Rank → Generate"]:::process
        COV["CoV Judge (gpt-4o)\nClaim-by-claim verification"]:::process
        F_SCORE{"FAITHFUL?"}:::decision
        F_PASS["✅ Pass"]:::success
        F_FAIL["❌ Fail"]:::danger
    end

    subgraph ADVERSARIAL ["  🛡️  Adversarial Queries (15)  "]
        A_GUARD["Guardrail Check Only\ngpt-4o-mini classifier"]:::process
        A_SCORE{"OUT_OF_SCOPE?"}:::decision
        A_PASS["✅ Pass (Refused)"]:::success
        A_FAIL["❌ Fail (Leaked)"]:::danger
    end

    subgraph METRICS ["  📊  Aggregate Metrics  "]
        M1["Factual Faithfulness\n100% (15/15)"]:::metric
        M2["Adversarial Success\n100% (15/15)"]:::metric
        M3["Retrieval Recall@10\n86.67% (13/15)"]:::metric
        M4["Avg Full-Pipeline Latency\n6.22s"]:::metric
        M5["Avg Guardrail Latency\n1.04s"]:::metric
        REPORT["📄 benchmark_report.md"]:::output
    end

    DS -->|"Type: factual"| F_PIPE
    DS -->|"Type: adversarial"| A_GUARD
    F_PIPE --> COV
    COV --> F_SCORE
    F_SCORE -->|"Yes"| F_PASS
    F_SCORE -->|"No"| F_FAIL
    A_GUARD --> A_SCORE
    A_SCORE -->|"Yes"| A_PASS
    A_SCORE -->|"No"| A_FAIL

    F_PASS & F_FAIL --> M1
    A_PASS & A_FAIL --> M2
    F_PIPE --> M3
    F_PIPE --> M4
    A_GUARD --> M5
    M1 & M2 & M3 & M4 & M5 --> REPORT

    classDef input fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef process fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef decision fill:#2c2c4a,stroke:#c09000,color:#fff
    classDef success fill:#2d6a4f,stroke:#40916c,color:#fff,font-weight:bold
    classDef danger fill:#7f1d1d,stroke:#ef4444,color:#fff
    classDef metric fill:#1e3a5f,stroke:#3b82f6,color:#bfdbfe,font-weight:bold
    classDef output fill:#78350f,stroke:#f59e0b,color:#fff,font-weight:bold
```

### Health Check

Before running evaluations, validate that all APIs and system components are functional:
```bash
python eval/health_check.py
```


## Tests

The `tests/` directory contains 13 test modules covering unit and integration testing across all major system components. Tests are run with `pytest` and coverage is enforced at a minimum of 70%.

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov=main --cov-report=term-missing

# Run a specific module
pytest tests/test_guardrail.py -v
```

Test coverage includes: cache, chunking, ChromaDB, embeddings, guardrail, hybrid retriever, integration pipeline, conversation memory, telemetry metrics, OpenAI LLM wrapper, query rewriter, and summarization.


## Docker Deployment

The application is fully containerized. The `Dockerfile` uses a `python:3.10-slim` base image and exposes port `8501` for the Streamlit server.

```bash
# Build the image
docker build -t howdyai .

# Run the container
docker run -p 8501:8501 --env-file .env howdyai
```

The `.dockerignore` file excludes `.venv`, `data/`, `logs/`, and other non-essential files from the build context.


## CI/CD & Code Quality

### GitHub Actions CI
The `.github/workflows/ci.yml` pipeline automatically runs on every push and pull request to `main`. It:
1. Sets up Python 3.10
2. Installs all dependencies from `requirements.txt`
3. Lints the codebase with `ruff`
4. Runs all `pytest` tests with coverage enforcement (≥70%)

### Pre-commit Hooks
The `.pre-commit-config.yaml` enforces code quality on every local commit:
- **autopep8**: Auto-formats Python code
- **ruff**: Fast linting
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml**: Validates YAML syntax
- **check-merge-conflict**: Prevents committing merge conflict markers

Install the hooks with:
```bash
pip install pre-commit
pre-commit install
```

### CI/CD Pipeline Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#500000', 'primaryTextColor': '#fff', 'lineColor': '#c09000', 'edgeLabelBackground': '#1a1a2e', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart LR
    PUSH(["git push / PR\nto main"]):::trigger

    subgraph GHA ["  GitHub Actions (ubuntu-latest)  "]
        CHECKOUT["actions/checkout@v3"]:::step
        PYTHON["Setup Python 3.10\nactions/setup-python@v4"]:::step
        INSTALL["pip install -r requirements.txt\npip install pytest pytest-cov ruff"]:::step
        LINT["ruff check ."]:::step
        TEST["pytest tests/\n--cov=src --cov=main\n--cov-fail-under=70"]:::step
        LINT_OK{"Lint\nPassed?"}:::decision
        TEST_OK{"Coverage\n≥ 70%?"}:::decision
        SUCCESS["✅ CI Green"]:::success
        FAIL["❌ CI Failed"]:::danger
    end

    subgraph LOCAL ["  Local Pre-commit Hooks  "]
        PC1["autopep8\nauto-format"]:::hook
        PC2["ruff\nlinting"]:::hook
        PC3["trailing-whitespace\nend-of-file-fixer"]:::hook
        PC4["check-yaml\ncheck-merge-conflict"]:::hook
    end

    PUSH --> CHECKOUT --> PYTHON --> INSTALL --> LINT --> LINT_OK
    LINT_OK -->|"Pass"| TEST --> TEST_OK
    LINT_OK -->|"Fail"| FAIL
    TEST_OK -->|"Pass"| SUCCESS
    TEST_OK -->|"Fail"| FAIL

    LOCAL -->|"Blocks commit"| PUSH

    classDef trigger fill:#500000,stroke:#c09000,color:#fff,font-weight:bold
    classDef step fill:#1a1a2e,stroke:#c09000,color:#e2e8f0
    classDef decision fill:#2c2c4a,stroke:#c09000,color:#fff
    classDef success fill:#2d6a4f,stroke:#40916c,color:#fff,font-weight:bold
    classDef danger fill:#7f1d1d,stroke:#ef4444,color:#fff
    classDef hook fill:#78350f,stroke:#f59e0b,color:#fff
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

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_key
BRAVE_API_KEY=your_brave_key
```

### Step 3: Build the Corpus (Optional)

If you want to build a fresh local vector database:
```bash
python crawl_tamu.py        # Crawl TAMU web pages
python create_database.py   # Embed and ingest into ChromaDB
```

### Step 4: Run the Application

```bash
streamlit run app.py
```


## Usage

Once the Streamlit server is running, navigate to `http://localhost:8501`.

The UI provides a ChatGPT-like interface where you can:
- Ask questions about TAMU courses and policies.
- View inline citations and source links for every generated answer.
- Monitor the real-time **Admin Dashboard** in the sidebar to see Cache Hit Rates, Guardrail Block Rates, and Average Latency metrics.


## Current Status

### Implemented

- Complete LangGraph orchestration pipeline with distinct Guardrail, Rewrite, Retrieve, Rank, and Generate nodes.
- High-performance Streamlit frontend with custom CSS, visual badges, and an administrative telemetry dashboard.
- Exact course code matching to prevent false-positive semantic retrieval of similar course numbers.
- Comprehensive `run_master_eval.py` evaluation suite providing verifiable Faithfulness and Guardrail metrics.
- Standalone `run_retrieval_eval.py` for isolated hybrid retrieval recall measurement.
- SQLite-based caching layer for sub-second responses on identical queries.
- 13-module test suite with ≥70% code coverage enforcement via CI.
- Clean `.dockerignore` and optimized `Dockerfile` for streamlined containerized deployment.
- GitHub Actions CI pipeline for automated linting and testing.
- Pre-commit hook configuration for consistent local code quality.

### Known Limitations

- Running `create_database.py` to index the entire TAMU web domain is time-consuming; the current seed database focuses on high-value catalogs and directories.


## Roadmap

### Phase 2: Deployment and Scale
Deploy the application via Docker to a cloud provider. Implement a scheduled CRON job to automatically run the `crawl_tamu.py` pipeline once a month to ensure the vector database stays synchronized with the latest university catalog updates.

### Phase 3: Expanded Coverage
Extend the scraper to cover additional TAMU domains including student organization directories, housing resources, and financial aid portals.


## Community & Policies

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Support](SUPPORT.md)
- [Security](SECURITY.md)
- [License](LICENSE)


## Citation

If you use this software in your research or project, please cite it:

```bibtex
@software{howdyai,
  author = {Praddep},
  title = {HowdyAI: Graph-Based RAG Search Engine},
  year = {2026},
  url = {https://github.com/praddep/HowdyAI}
}
```

---
<p align="center">Made with ❤️ by Pradeep Periyasamy</p>
