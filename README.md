# 🧠 Chat Memo Tool

> *"You've had hundreds of AI conversations. What have you actually built?"*

Most developers use AI assistants daily — but the insights, decisions, and progress buried across those conversations are invisible. **Chat Memo Tool** solves that by turning your raw ChatGPT history into a structured, searchable knowledge base with automated strategic reporting — running **completely offline on your own machine**.

---

## The Problem

When working on a long-running project, you accumulate dozens (or hundreds) of AI conversations: architecture decisions, bug investigations, feature explorations, research notes. They're scattered, unsearchable, and forgotten the moment the tab closes.

There's no memory. No continuity. No overview.

---

## The Solution

Chat Memo Tool is an **automated knowledge extraction pipeline** that:

1. **Parses** your exported ChatGPT conversation history
2. **Extracts** user prompt + AI response pairs, filtered by project tag
3. **Classifies** each pair using a **local Ollama model** → summary, category, priority
4. **Stores** everything in a local SQLite database
5. **Clusters** similar conversations semantically (sentence-transformers + KMeans)
6. **Generates** a strategic project status report: what you've built, key decisions, blockers, and suggested next steps

The result: a living, queryable record of everything you've explored — and an AI-written status report you can read in 60 seconds. **No API keys. No cloud. No cost.**

---

## Architecture

```
conversations.json  (ChatGPT Export)
        │
        ▼
  extractor.py       ←  filters by project prefix, pairs user+AI turns
        │
        ▼
  classifier.py      ←  Ollama (local) → {summary, category, priority}
        │
        ▼
  storage.py         ←  SQLite (memo.db) — persistent, queryable
        │
        ├──▶  clustering.py   ←  semantic KMeans grouping
        │
        └──▶  reporter.py     ←  AI strategic report / offline summary
```

Each module has a single responsibility. The pipeline is crash-safe: progress is saved to the database after every message, so interrupted runs resume exactly where they left off.

---

## Key Technical Decisions

### Why local Ollama instead of a cloud API?
The original version used Google's Gemini API — which meant API keys, rate limits, 10-second waits between messages, and your private project data leaving your machine. Switching to Ollama means classification runs on your own hardware: zero cost, no quotas, 2-second intervals, and complete data privacy.

### Why SQLite instead of flat JSON?
The original approach used two JSON files (`config.json` + `classified_results.json`) which were fully rewritten on every save. At scale this becomes slow, fragile, and error-prone. SQLite gives us proper indexing, atomic writes, upserts, and the ability to query by category, date range, or priority without loading everything into memory.

### Why tenacity for retries?
Local models can still fail transiently — timeouts, cold starts, resource spikes. A bare `try/except` that silently swallows errors and marks a message as "processed" means data is permanently lost. `tenacity` provides exponential backoff with configurable attempts, so transient failures are recovered automatically.

### Why walk the full child tree for AI responses?
ChatGPT's export stores conversations as a node tree, not a flat list. When a user regenerates a response, multiple children exist for a single user turn. Taking `children[0]` blindly picks whichever response happened to be first in the dict — not necessarily the one the user kept. The extractor now walks all children (and one level of grandchildren) to find the canonical assistant response.

### Why embedding cache?
`SentenceTransformer` encodes each message into a 384-dimensional vector. On a corpus of 500 messages this takes ~30 seconds. With a disk cache keyed by MD5 hash, re-runs only encode genuinely new messages — making clustering near-instant on subsequent calls.

---

## Features

| Feature | Detail |
|---|---|
| **Fully offline** | Runs on local Ollama — no API key, no internet required |
| **Zero cost** | No cloud API calls, no usage limits, no billing |
| **Incremental processing** | Tracks processed node IDs in SQLite — never re-classifies the same message |
| **Project filtering** | Only processes conversations matching your configured prefix (e.g. `bjk`) |
| **Retry logic** | 3 attempts with exponential backoff on every model call |
| **Offline report** | Full report from local data — no model calls needed |
| **Semantic clustering** | Groups similar conversations automatically using KMeans |
| **Structured logging** | Console output + persistent `chat_memo.log` file |
| **CLI interface** | `--mode` flag to run any individual pipeline stage |
| **Secret-safe** | No credentials needed — nothing to expose |

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download/windows) installed and running

### Install the model
```bash
ollama pull qwen2.5:0.5b
```

### Install the tool
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/chat-memo-tool.git
cd chat-memo-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set your CONVERSATIONS_PATH to your ChatGPT export file
```

---

## Usage

```bash
# Full pipeline: extract → classify → cluster → report
python main.py

# Only classify new messages
python main.py --mode classify

# Generate strategic report from stored data
python main.py --mode report

# Offline report — zero model calls
python main.py --mode offline

# Re-run semantic clustering
python main.py --mode cluster

# Remove failed classification rows and re-queue them
python main.py --mode cleanup

# Control how many items feed into the report
python main.py --mode report --limit 50
```

---

## Sample Report Output

```
## 1. CURRENT FOCUS
The developer is actively integrating a real-time data pipeline with the
existing REST API layer, with emphasis on error handling and retry logic.

## 2. KEY DECISIONS MADE
- Chose SQLite over PostgreSQL for local-first simplicity
- Adopted event-driven architecture for the notification service
- Deferred authentication to Phase 2 to unblock core feature work

## 3. POTENTIAL BLOCKERS
- Unresolved latency issue in the message queue under high load
- Third-party SDK lacks async support — may require custom wrapper

## 4. SUGGESTED NEXT STEPS
1. Write integration tests for the pipeline's failure recovery path
2. Benchmark the queue under simulated peak load before next demo
3. Open spike ticket for async SDK wrapper — 2-day time-box
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Local AI Engine | [Ollama](https://ollama.com) |
| Classification Model | qwen2.5:0.5b (runs fully offline) |
| Semantic Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Clustering | scikit-learn KMeans |
| Storage | SQLite (via stdlib `sqlite3`) |
| Retry Logic | `tenacity` |
| Configuration | `python-dotenv` |
| Logging | stdlib `logging` |
| CLI | stdlib `argparse` |

---

## Project Structure

```
chat_memo_tool/
├── main.py          # Entry point + CLI argument parsing
├── config.py        # Centralised config, reads from .env
├── logger.py        # Logging to console + file
├── extractor.py     # Parses conversations.json, extracts message pairs
├── classifier.py    # Ollama API calls with retry, returns typed results
├── clustering.py    # Semantic clustering with embedding cache
├── reporter.py      # AI report + offline report generation
├── storage.py       # Full SQLite persistence layer
├── requirements.txt
├── .env.example     # Safe config template
└── .gitignore
```

---

## What I Learned Building This

- Designing a **multi-stage data pipeline** where each stage is independently testable and replaceable
- Working with **tree-structured data** (ChatGPT's conversation mapping format) and handling edge cases like regenerated responses
- Practical patterns for **resilient local AI integrations**: retry strategies, rate limiting, incremental saves
- The difference between **prototype-grade and production-grade** Python: typed outputs, structured logging, separation of concerns, and crash safety
- Building a **local-first AI system** — understanding the tradeoffs between cloud APIs and on-device inference

---

## License

MIT — feel free to adapt for your own workflow.
