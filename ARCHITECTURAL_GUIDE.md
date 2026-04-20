# HowdyAI: Comprehensive Project Architecture & Deep-Dive Guide

This document provides an exhaustive technical analysis of the HowdyAI ecosystem. Each section below offers a high-density explanation of the module's implementation, logical dependencies, and the architectural justifications that differentiate HowdyAI from standard large language model wrappers.

---

### 🖥️ The Interactive Layer: `app.py` (Frontend & UI Logic)

`app.py` serves as the primary gateway for user interaction, utilizing the Streamlit framework to deliver a real-time, stateful web experience. Unlike traditional static frontends, this module manages the entire lifecycle of a conversation session through `st.session_state`. It handles the initialization of the conversation history, the management of UI widgets (such as the sidebar metrics and the chat input), and the rendering of complex HTML/CSS blocks used for high-fidelity styling. To achieve a "premium" aesthetic, the file injects custom CSS to implement glassmorphism effects, a sleek Apple-style dark mode, and responsive layout adjustments that override Streamlit’s default grid system. When a user submits a query, `app.py` triggers the `main.py` pipeline, captures the streaming or batch response, and updates the UI dynamically. It also handles the logic for "Citations" by parsing the response metadata and rendering interactive chips that link directly to the source PDFs or webpages used by the AI. This file is the "face" of the project, ensuring that the complex RAG processes happening in the background are translated into a user-friendly, visually stunning interface.

---

### 🎼 The Pipeline Orchestrator: `main.py` (Execution Logic)

`main.py` is the central "brain" or orchestrator of the HowdyAI 9-stage pipeline. Its role is to ensure that data flows seamlessly from the user’s input through various refinement, retrieval, and generation stages without data loss or logic errors. The code is structured as a series of gates: it first calls the `QueryGuardrail` to classify the intent, then checks the `ResponseCache` to see if a pre-computed answer exists. If a search is required, it initiates a complex "Hybrid Retrieval" flow, merging data from both the live web and the local vector store. The file is responsible for error handling—if the web search fails, it ensures the local database fallback is activated; if the AI fails to summarize a specific result, it ensures the pipeline continues regardless. By centralizing the orchestration here, we separate the "What to do" (the pipeline) from the "How to do it" (the individual modules in `src/`). This allows for modular testing and easy upgrades to specific stages (like switching the ranking algorithm) without redesigning the entire interaction loop.

---

### 🦆 The Real-Time Web Engine: `google_search.py` (Web Search & Scraping)

Despite its legacy name, `google_search.py` is the module responsible for live information gathering using the DuckDuckGo Search (DDGS) API. This provides the RAG system with "Temporal Freshness"—the ability to answer questions about campus news or events that happened after the LLM’s training cutoff. The logic in this file is designed for "Hyper-Relevance": it first attempts a domain-scoped search (`site:tamu.edu`) to ensure academic authority, and only falls back to a broader search if results are scarce. Beyond simple searching, this file includes a robust "Scraping Layer" using `requests` and `PyPDF2`. When a search result is found, the code visits the URL, extracts the raw HTML or PDF content, and handles various edge cases like timeout errors or content-blocking by switching to the search provider’s body snippet. This dual-path extraction ensures that the AI always has some context to work with, even if a firewall blocks the live scraping of a specific TAMU department page.

---

### 🌊 The Data Fusion Layer: `hybrid_retriever.py` (RRF Retrieval)

The `HybridRetriever` in `src/search/hybrid_retriever.py` represents the technical pinnacle of the retrieval stage. It implements **Reciprocal Rank Fusion (RRF)**, a consensus-based ranking algorithm that allows HowdyAI to merge results from the local ChromaDB (semantic search) and DuckDuckGo (keyword/web search) into a single, optimized list. The RRF formula—`1 / (k + rank)`—ensures that if a document appears high in *both* search lists, it is strongly prioritized, while documents that only appear in one list are given lower weights. This module acts as the bridge between the "Static Knowledge" (the vector database we crawled) and "Dynamic Knowledge" (the live internet). It also handles the conversion of diverse data formats into a standardized dictionary that the downstream `summarize_with_llm` stage can process. By using RRF, we solve the problem of "Weighting Bias," where a vector search might be too vague or a web search might be too literal; the fusion layer finds the semantic "Golden Mean" between them.

---

### 🏗️ The Knowledge Ingestion Engine: `create_database.py` (Semantic Chunking)

`create_database.py` is the offline script responsible for building the "Long-Term Memory" of the chatbot. Its implementation is significantly more advanced than standard loaders because it replaces fixed-size character splitting with **Semantic Chunking**. Instead of cutting text every 500 characters—which often results in "chopped-up" sentences—it uses the `text-embedding-3-large` model to calculate the cosine similarity between every sentence in a document. If the similarity between consecutive sentences drops below a threshold (0.80), the code identifies a "Topic Shift" and creates a new chunk. This ensures that every entry in the **ChromaDB** is a semantically coherent "idea." The file also manages the persistence of the vector store, ensuring that the 1,700+ TAMU pages we crawled are indexed efficiently with metadata (like source URLs and page titles) intact, allowing the final chatbot to provide the "Clickable Citations" that prove its answers are grounded in fact.

---

### 🛡️ The Semantic Gatekeeper: `guardrail.py` (Intent & Scope Control)

`guardrail.py` serves as the project’s security and cost-optimization layer. It uses a specialized prompt for `gpt-4o-mini` to classify the user’s query as either "Inside TAMU Scope" or "Outside Scope" (e.g., general world news, medical advice, or malicious jailbreak attempts). This is executed at "Stage 0," meaning that if a query is deemed out-of-scope, the pipeline never initiates the expensive search, summarization, or ranking stages. This architectural choice protects the API budget and ensures that HowdyAI maintains its "TAMU Brand Voice," refusing to answer off-topic questions that might lead to hallucinations. The logic also includes checks for "empty" or "non-sensical" queries, providing a graceful rejection message instead of a generic AI error. By implementing guardrails as an LLM-classifier, we gain much higher precision than simple keyword-based blacklists, as the AI can understand the *context* of why a question might be inappropriate for a university assistant.

---

### 📉 The Context Compression Layer: `summarize_with_llm.py` (Parallel Processing)

The `summarize_with_llm.py` module solves the "Information Overload" problem inherent in web-based RAG systems. When a search returns 5–10 long webpages, providing the raw text to the AI would exceed its context window and lead to confusion. This file uses a **Parallel Processing architecture** (via `ThreadPoolExecutor`) to send each search result to a "Summarizer" LLM simultaneously. This reduces total latency from 30+ seconds to under 5 seconds by summarizing multiple pages in parallel. The logic is tuned to extract "Answer Signals"—if a page contains a date, a location, or a requirement, the summarizer highlights it while discarding the "noise" (like navigation menus or footer links). The result is a refined set of "Snippets" that are perfectly sized for the final generation stage. This "Map-Reduce" style approach is why HowdyAI can process vast amounts of TAMU data without becoming slow or inaccurate.

---

### 🏛️ The Instructional Framework: `src/templates/*.txt` (Prompt Engineering)

The `.txt` files in the `src/templates/` directory represent the "Programming" of the AI’s personality and logic. We purposefully decoupled these from the Python code to allow for rapid iteration on the bot’s "Instructional Integrity." 
*   **`chat_template.txt`**: Defines the "Aggressive Grounding" rules, strictly forbidding the AI from answering if it cannot find a source and shaping the identity of "HowdyAI."
*   **`rewrite_template.txt`**: Teaches the AI how to expand vague queries into TAMU-specific search terms.
*   **`rank_template.txt`**: Gives the AI the criteria for determining which search result is a "Primary Source" vs. "Secondary Source."
*   **`summary_template.txt`**: instructs the model to prioritize tabular data and key facts.
By using these templates, we ensure the system is "Deterministic"—it follows a strict script that prioritizes academic integrity and source attribution above all else.

---

### 💾 The Persistence & Vector Layers: `.db` Files (Storage Internals)

HowdyAI uses a "Dual-Database" strategy to balance speed and intelligence. 
1.  **`howdyai_cache.db` (SQLite)**: This is a relational database that stores a SHA-256 hash of every successful user query. If a user (or another user) asks the same question later, the system retrieves the answer instantly from this local file, bypassing all AI stages and costing $0.00.
2.  **`data/database/` (ChromaDB)**: This is a **Vector Store** that stores the multi-dimensional embeddings of the TAMU knowledge base. Unlike a normal database, you cannot "read" it like a table; instead, it provides a "Similarity Search" interface. When you ask a question, ChromaDB does a "Nearest Neighbor" calculation to find the most similar pieces of TAMU information.
This combination—**Exact-Match Cache** for speed and **Semantic Vector Store** for intelligence—is what makes HowdyAI feel both extremely fast for common questions and incredibly smart for new ones.

---

## 📂 Exhaustive Technical Directory

This appendix provides a file-by-file audit of the HowdyAI repository, documenting the specific code, functions, and internal logic contained within each path.

### 📍 Root Directory: System Ingress & Ingestion

#### `app.py`
The primary Streamlit application file. It contains the logic for UI orchestration, including:
- **`main()`**: The main entry point that initializes the sidebar and chat interface.
- **`render_sidebar_metrics()`**: Logic for displaying "Knowledge Source" counts and "Live Search" status.
- **`inject_custom_css()`**: Injects the premium glassmorphism and Apple-style dark mode styling.
- **Interaction Loop**: Code that manages `st.session_state` to prevent page resets and connects user input to the `HowdyAI.run_pipeline()` method.

#### `main.py`
The central orchestrator of the RAG pipeline. It contains the **`HowdyAI`** class and its core method:
- **`run_pipeline(query)`**: Implementation of the 9-stage sequence. It initializes every tool in the `src/` directory and manages the state transfer from the Guardrail through to the final Response Generation.

#### `config.py`
The central configuration management file. It contains the **`AppConfig`** class:
- **`__init__()`**: Loads environment variables (API keys) via `python-dotenv`.
- **Constants**: Defines paths for the database, templates, and logs. It also sets critical hyper-parameters like `SEMANTIC_THRESHOLD = 0.80` and `NUM_SEARCH_RESULTS = 5`.

#### `crawl_tamu.py`
The data collection engine. It contains the **`TamuCrawler`** class:
- **`crawl(start_url)`**: A BFS-style recursive web crawler.
- **`scrape_page()`**: Uses `BeautifulSoup4` to extract clean text while filtering out navigation bars and footers.
- **Logic**: Handles URL normalization and respects `robots.txt` directives to ensure ethical scraping.

#### `create_database.py`
the knowledge ingestion script. It contains the **`DatabaseGenerator`** class and custom chunking logic:
- **`semantic_chunk()`**: An advanced algorithm that uses `text-embedding-3-large` to calculate cosine similarity between sentences, splitting text only when a topic shift is detected.
- **`generate_database()`**: Orchestrates the loading of HTML files via LangChain's `BSHTMLLoader` and the persistence of embeddings into the Chroma vector store.

---

### 📍 Core Source (`src/`): Fundamental Abstractions

#### `src/guardrail.py`
The security layer. Contains the **`QueryGuardrail`** class:
- **`is_in_scope(query)`**: Sends the query to `gpt-4o-mini` to classify if the intent is related to Texas A&M University, preventing irrelevant or malicious queries from entering the pipeline.

#### `src/memory.py`
The context management layer. Contains the **`ConversationMemory`** class:
- **`get_history()`**: Retrieves the sliding window of the last 6 conversation turns.
- **`add_message()`**: Updates the history with new user/bot interactions, ensuring the LLM can resolve pronouns like "it" or "he" in follow-up questions.

#### `src/cache.py`
The performance layer. Contains the **`ResponseCache`** class:
- **`get(query)`**: Hashes the query using SHA-256 and checks a local SQLite database for a match.
- **`set(query, response)`**: Stores new, verified answers to eliminate duplicate API costs for common questions.

#### `src/database.py`
The vector interface layer. Contains the **`Database`** class:
- **`search(query, k)`**: A wrapper for the Chroma vector store that returns the top `k` documents along with their relevance scores for filtering.

#### `src/embeddings.py`
The vectorization engine. Contains the **`Embeddings`** class:
- **`embedding_function`**: A singleton implementation that provides the `text-embedding-3-large` model interface to both the database and the search engines.

---

### 📍 Search Specialists (`src/search/`): Retrieval & Refinement

#### `src/search/hybrid_retriever.py`
The data fusion engine. Contains the **`HybridRetriever`** class:
- **`reciprocal_rank_fusion()`**: The mathematical implementation of RRF used to merge web and local results.
- **`retrieve(query)`**: Orchestrates the dual-retrieval process and applies a `0.40` relevance threshold to filtered results.

#### `src/search/google_search.py`
The real-time research engine. Contains the **`duckduckgo_search_engine`** function:
- **`scrape_content()`**: Uses `requests` with custom headers to extract text from live webpages.
- **`scrape_pdf_content()`**: Uses `PyPDF2` to read and index PDF documents found during web searches.

#### `src/search/query_rewriter.py`
The query expansion engine. Contains the **`QueryRewriter`** class:
- **`rewrite(query, history)`**: Uses an LLM to transform short questions into detailed, search-ready prompts optimized for the TAMU domain.

#### `src/search/summarize_with_llm.py`
The context compression engine. Contains the **`summarize_results()`** function:
- **Multi-threading**: Uses `concurrent.futures.ThreadPoolExecutor` to process multiple sources in parallel, drastically reducing the time needed to digest search results.

#### `src/search/rank_with_llm.py`
The quality control engine. Contains the **`rank_results()`** function:
- **Semantic Sorting**: Asks an LLM to evaluate summaries against the original user query, discarding low-signal or irrelevant information before final generation.

#### `src/search/data_processor.py`
The text utility library. Contains:
- **`clean_text()`**: Regex-based logic to remove double spaces, unwanted symbols, and HTML debris.
- **`truncate_text()`**: Ensures all text passed to the final generative stage fits within the token limit of the model.

---

### 📍 AI & Instructions: Models & Templates

#### `src/language_models/openai_language_model.py`
The model abstraction layer. Contains the **`OpenAILanguageModel`** class:
- **`generate()`**: Manages the prompt-to-model mapping, handling communication with OpenAI's Chat Completion API.

#### `src/templates/*.txt`
The logical foundation of the bot's behavior. Content includes:
- **`chat_template.txt`**: The high-level persona and grounding rules for the final response.
- **`guardrail_template.txt`**: The classification criteria for OOS query detection.
- **`rank_template.txt`**: The objective evaluation criteria for sorting search results.
- **`rewrite_template.txt`**: Rules for resolving coreference and expanding queries.
- **`summary_template.txt`**: Formatting instructions for compressing raw data into key facts.
