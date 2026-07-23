# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-07-22

### Added

**Core Pipeline**
- LangGraph multi-stage orchestration pipeline with Guardrail, Rewrite, Retrieve, Rank, and Generate nodes
- Out-of-scope guardrail classifier using `gpt-4o-mini` and a strict policy prompt (`guardrail_template.txt`)
- Contextual query rewriter that resolves pronouns and expands TAMU-specific abbreviations
- Hybrid retrieval combining ChromaDB (Cosine similarity) and Brave Search API live web fallback
- Cross-Encoder re-ranking for precision improvement over bi-encoder retrieval alone
- TF-IDF context extraction for noise reduction in scraped web documents
- LLM-based snippet summarization for cleaner context injection
- OpenAI GPT-4o generation with strict grounding instructions and inline source citations
- Optional Chain-of-Verification (CoV) judge loop with automatic retry on hallucination detection
- Multi-turn conversation memory with a 6-turn sliding window (`memory.py`)
- SQLite-based response caching layer for sub-second repeat query responses (`cache.py`)

**Data & Infrastructure**
- `crawl_tamu.py` web scraper for building the TAMU university corpus
- `create_database.py` ingestion pipeline for embedding and indexing into ChromaDB
- ChromaDB vector store wrapper with Cosine metric (`database.py`)
- Database migration utilities: `dedup_chroma.py`, `migrate_chroma.py`, `patch_chroma_urls.py`

**Evaluation**
- End-to-end master evaluation pipeline (`run_master_eval.py`) with CoV judge scoring
- Standalone hybrid retrieval recall evaluation (`run_retrieval_eval.py`)
- 30-query hand-curated ground-truth dataset (`master_eval_dataset.json`)
- Retrieval QA pairs dataset (`retrieval_qa_pairs.json`)
- System health check script (`eval/health_check.py`)
- Verified benchmark report achieving 100% Factual Faithfulness and 100% Adversarial Guardrail Success

**Frontend**
- Streamlit UI (`app.py`) with ChatGPT-like interface, custom CSS, and visual status badges
- Real-time Admin Dashboard with Cache Hit Rate, Guardrail Block Rate, and Average Latency telemetry

**Testing & CI**
- 13-module pytest test suite covering all major system components
- ≥70% code coverage enforcement
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`) for automated lint and test on push/PR
- Pre-commit hooks (`.pre-commit-config.yaml`) for autopep8, ruff, whitespace, and YAML validation

**Observability**
- Structured logging to `logs/debug.log`, `logs/info.log`, and `logs/error.log` via `metrics.py`

**Containerization**
- `Dockerfile` using `python:3.10-slim` base image
- `.dockerignore` for optimized build context
