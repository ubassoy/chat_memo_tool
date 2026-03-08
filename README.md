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
3. **Classifies** each pair using a local Ollama model → summary, category, priority
4. **Stores** everything in a local SQLite database
5. **Finds the optimal cluster count** automatically using Silhouette Score
6. **Clusters** similar conversations semantically into topic groups
7. **Generates** a focused strategic report **per cluster** + a **master overview** across all clusters

The result: a living, queryable second brain — where every topic area gets its own focused report, and a master summary ties everything together. **No API keys. No cloud. No cost.**

---

## Architecture

```
conversations.json  (ChatGPT Export)
        │
        ▼
  extractor.py       ←  filters by project prefix, pairs user+AI turns
        │
        ▼
  classifier.py      ←  Ollama qwen2.5:0.5b → {summary, category, priority}
        │
        ▼
  storage.py         ←  SQLite (memo.db) — persistent, queryable
        │
        ▼
  clustering.py      ←  Silhouette Score finds best k → KMeans → labels saved to DB
        │
        ├──▶  reporter.py  →  reports/cluster_0_feature.txt
        ├──▶  reporter.py  →  reports/cluster_1_research.txt
        ├──▶  reporter.py  →  reports/cluster_2_planning.txt
        │     ... (one file per cluster)
        │
        └──▶  reporter.py  →  reports/master.txt
```

Each module has a single responsibility. The pipeline is crash-safe: progress is saved to the database after every message, so interrupted runs resume exactly where they left off.

---

## Key Technical Decisions

### Why local Ollama instead of a cloud API?
The original version used Google's Gemini API — which meant API keys, rate limits, 10-second waits between messages, and your private project data leaving your machine. Switching to Ollama means classification runs on your own hardware: zero cost, no quotas, 2-second intervals, and complete data privacy.

### Why Silhouette Score for cluster count?
Choosing the number of clusters manually is guesswork — too few mixes unrelated topics, too many creates meaningless micro-groups. The Silhouette Score mathematically measures how well-separated clusters are (range: -1 to +1). The system tests every value of k from 2 to 7, scores each one, and automatically selects the k with the highest score. You get the optimal grouping without any manual tuning.

### Why per-cluster reports instead of one big report?
A single report across all conversations mixes unrelated topics together, producing incoherent output. Grouping conversations semantically first and then generating one focused report per cluster means each report is coherent, specific, and actually useful. A master report then synthesises across clusters for a high-level overview.

### Why deduplicate before prompting?
When many conversations cover similar ground, their summaries are near-identical. Feeding 17 copies of "build a financial literacy plan" into a small model causes it to loop and repeat. The reporter deduplicates summaries before building the prompt, caps input at 4-8 unique items per cluster, and applies `repeat_penalty` to the model — eliminating the repetition problem at multiple layers.

### Why a two-model split?
Classification runs hundreds of times (once per message) so speed and low RAM usage matter most — `qwen2.5:0.5b` (400MB) is ideal. Report generation runs only a handful of times but needs better reasoning — a slightly larger model produces more coherent output. This split gives the best quality/performance balance on hardware with limited RAM.

### Why SQLite instead of flat JSON?
The original approach used two JSON files which were fully rewritten on every save. At scale this becomes slow and fragile. SQLite gives proper indexing, atomic writes, upserts, and the ability to query by category, cluster, date, or priority without loading everything into memory.

### Why tenacity for retries?
Local models can still fail transiently — timeouts, cold starts, resource spikes. A bare `try/except` that silently swallows errors marks a message as "processed" and loses it forever. `tenacity` provides exponential backoff with configurable attempts so transient failures recover automatically.

### Why walk the full child tree for AI responses?
ChatGPT's export stores conversations as a node tree, not a flat list. When a user regenerates a response, multiple children exist for a single user turn. Taking `children[0]` blindly picks whichever response came first — not necessarily the one the user kept. The extractor walks all children (and one level of grandchildren) to find the canonical assistant response.

### Why embedding cache?
`SentenceTransformer` encodes each message into a 384-dimensional vector. On a corpus of 500 messages this takes ~30 seconds. With a disk cache keyed by MD5 hash, re-runs only encode genuinely new messages — making clustering near-instant on subsequent calls.

---

## Features

| Feature | Detail |
|---|---|
| **Fully offline** | Runs on local Ollama — no API key, no internet required |
| **Zero cost** | No cloud API calls, no usage limits, no billing |
| **Auto optimal clustering** | Silhouette Score tests k=2 to k=7 and picks the best automatically |
| **Per-cluster reports** | Each topic group gets its own focused, coherent report |
| **Master report** | Cross-cluster overview summarising the entire project |
| **Deduplication** | Identical summaries filtered before prompting — no repeated output |
| **Incremental processing** | Tracks processed node IDs — never re-classifies the same message |
| **Project filtering** | Only processes conversations matching your configured prefix |
| **Retry logic** | 3 attempts with exponential backoff on every model call |
| **Semantic clustering** | Auto-labels clusters by dominant category (Feature, Research, etc.) |
| **Structured logging** | Console output + persistent `chat_memo.log` file |
| **CLI interface** | `--mode` flag to run any individual pipeline stage |
| **Secret-safe** | No credentials needed — nothing to expose |

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download/windows) installed and running

### Install models
```bash
# For classification (fast, lightweight)
ollama pull qwen2.5:0.5b

# For report generation (smarter output)
ollama pull qwen2.5:1.5b
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
# Edit .env: set CONVERSATIONS_PATH to your ChatGPT export file
```

---

## Usage

```bash
# Full pipeline: extract → classify → cluster → all reports
python main.py

# Only classify new messages
python main.py --mode classify

# Re-run clustering (auto-finds optimal k)
python main.py --mode cluster

# Generate per-cluster reports + master report
python main.py --mode report

# Generate per-cluster reports only
python main.py --mode report_cluster

# Generate master report only
python main.py --mode report_master

# Offline grouped summary — zero model calls
python main.py --mode offline

# Remove failed classification rows and re-queue them
python main.py --mode cleanup
```

---

## How Optimal Clustering Works

When you run `--mode cluster`, the system automatically tests every cluster count from 2 to 7 and prints a scoring table:

```
Finding optimal cluster count (k=2 to k=7)...
k      Silhouette Score     Inertia
---------------------------------------------
k=2    score=0.3821         inertia=142.30
k=3    score=0.5103         inertia=98.12
k=4    score=0.4890         inertia=76.44
k=5    score=0.3201         inertia=61.22
k=6    score=0.2988         inertia=55.10
k=7    score=0.2541         inertia=49.87
---------------------------------------------
✅ Best k = 3  (Silhouette Score: 0.5103)
Cluster quality: Excellent separation
```

**Reading the Silhouette Score:**

| Score | Meaning |
|---|---|
| > 0.5 | Excellent — clusters are clearly distinct |
| 0.3 – 0.5 | Good — reasonable separation |
| 0.1 – 0.3 | Moderate — some topic overlap |
| < 0.1 | Weak — data may not cluster naturally |

You can still override the auto-selection by setting `NUM_CLUSTERS` in your `.env` file if you want to force a specific number.

---

## Output Structure

After running the full pipeline, a `reports/` folder is created:

```
reports/
├── cluster_0_feature.txt      ←  focused report on Feature conversations
├── cluster_1_research.txt     ←  focused report on Research conversations
├── cluster_2_planning.txt     ←  focused report on Planning conversations
└── master.txt                 ←  strategic overview across ALL clusters
```

---

## Sample Cluster Report

```
## CLUSTER: Research (12 conversations)

### 1. WHAT THIS CLUSTER IS ABOUT
Investigative conversations around data pipeline architecture,
third-party SDK evaluation, and performance benchmarking.

### 2. KEY INSIGHTS (4 unique points)
- Chose event-driven approach after benchmarking three alternatives
- Identified latency bottleneck in the message queue under high load
- Rejected one SDK due to lack of async support
- Load test results inconclusive above 10k req/min

### 3. OPEN QUESTIONS
- Async SDK wrapper still needs a time-boxed spike

### 4. NEXT STEPS
1. Open a 2-day spike ticket for the async wrapper
2. Re-run load test with latest queue configuration
3. Document architecture decision with tradeoffs
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Local AI Engine | [Ollama](https://ollama.com) |
| Classification Model | qwen2.5:0.5b (fast, lightweight) |
| Report Generation Model | qwen2.5:1.5b (smarter reasoning) |
| Cluster Optimisation | Silhouette Score (scikit-learn) |
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
├── classifier.py    # Ollama classification with retry, typed results
├── clustering.py    # Silhouette Score + KMeans + embedding cache
├── reporter.py      # Per-cluster + master report generation
├── storage.py       # Full SQLite persistence layer
├── requirements.txt
├── .env.example     # Safe config template
└── .gitignore
```

---

## What I Learned Building This

- Designing a **multi-stage data pipeline** where each stage is independently testable and replaceable
- Working with **tree-structured data** (ChatGPT's conversation mapping format) and handling edge cases like regenerated responses
- Using **Silhouette Score** to find the mathematically optimal cluster count instead of guessing — and understanding the tradeoff between cluster quality and granularity
- Why **semantic clustering before reporting** matters — mixing unrelated topics into a single prompt produces incoherent output; separating them first produces focused, useful reports
- **Prompt engineering for small models** — deduplication, input capping, and repeat penalty are necessary to prevent looping on limited hardware
- The difference between **prototype-grade and production-grade** Python: typed outputs, structured logging, separation of concerns, and crash safety
- Building a **local-first AI system** — understanding the tradeoffs between cloud APIs and on-device inference, and optimising for constrained hardware

---


