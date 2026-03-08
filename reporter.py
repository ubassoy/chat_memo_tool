"""
reporter.py — Generates per-cluster and master strategic reports.

Modes:
  1. Per-cluster reports  — one focused report per cluster → reports/cluster_N.txt
  2. Master report        — overview across all clusters   → reports/master.txt
  3. Offline report       — local summary, no model calls
"""

import os
import time
import requests
from datetime import datetime
from typing import Optional

from config import cfg
from logger import log
from storage import load_results, load_results_grouped_by_cluster, get_cluster_ids


# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
REPORTS_DIR = "reports"

# Max unique bullet points fed into any single prompt
# Prevents the model from looping on too much input
MAX_ITEMS_PER_CLUSTER = 8
MAX_ITEMS_MASTER = 3  # top N items per cluster for master summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_timestamp(ts) -> str:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    if isinstance(ts, str):
        return ts[:10]
    return "Unknown Date"


def _build_bullet_log(rows: list[dict], max_items: int = MAX_ITEMS_PER_CLUSTER) -> str:
    """
    Builds a deduplicated bullet list from rows.
    - Limits to max_items to prevent model looping
    - Deduplicates by summary text so repeated summaries don't bloat the prompt
    """
    seen = set()
    lines = []

    for item in rows:
        summary = (item.get("summary") or "").strip()
        if not summary or summary in seen:
            continue
        seen.add(summary)

        date_str = _format_timestamp(item.get("timestamp"))
        category = item.get("category") or "General"
        lines.append(f"- [{date_str}] [{category}]: {summary}")

        if len(lines) >= max_items:
            break

    return "\n".join(lines)


def _ensure_reports_dir() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _call_ollama(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """Sends a prompt to local Ollama and returns the response text."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,    # Lower = less hallucination
                "num_predict": max_tokens,
                "repeat_penalty": 1.3, # Penalises repetition heavily
                "top_p": 0.9,
            }
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama. Make sure it is running.")
        return None
    except Exception as exc:
        log.error("Ollama call failed: %s", exc)
        return None


def _save_report(filename: str, content: str) -> None:
    _ensure_reports_dir()
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Per-cluster report prompt
# ---------------------------------------------------------------------------

CLUSTER_REPORT_PROMPT = """You are a project analyst. Respond in English only.
Do NOT repeat the same point more than once.
Be concise — each section should have 2-4 unique bullet points maximum.

Write a short report for the topic cluster below.

## CLUSTER: {label} ({count} conversations)

### 1. WHAT THIS CLUSTER IS ABOUT
One short paragraph describing the theme.

### 2. KEY INSIGHTS (max 4 unique points, no repetition)

### 3. OPEN QUESTIONS

### 4. NEXT STEPS (max 3 actions)

WORK LOG ({shown} of {total} items shown):
{log_data}

Important: Do not repeat any point. Stop after section 4."""


# ---------------------------------------------------------------------------
# Master report prompt
# ---------------------------------------------------------------------------

MASTER_REPORT_PROMPT = """You are a project analyst. Respond in English only.
Do NOT repeat the same point more than once. Be concise.

Write a master overview report based on the cluster summaries below.

## MASTER PROJECT REPORT

### 1. OVERALL STATUS (2-3 sentences)

### 2. TOP 3 PRIORITIES (unique, no repetition)

### 3. CROSS-CLUSTER CONNECTIONS (if any)

### 4. RECOMMENDED NEXT STEPS (3 actions only)

CLUSTER SUMMARIES:
{cluster_summaries}

Important: Do not repeat any point. Stop after section 4."""


# ---------------------------------------------------------------------------
# Per-cluster report generation
# ---------------------------------------------------------------------------

def generate_cluster_reports(db_path: str) -> dict[int, str]:
    """
    Generates one report per cluster, saved to reports/cluster_N.txt
    Returns dict of {cluster_id: report_text}
    """
    grouped = load_results_grouped_by_cluster(db_path)
    clustered_groups = {k: v for k, v in grouped.items() if k >= 0}

    if not clustered_groups:
        log.warning("No cluster assignments found. Run: python main.py --mode cluster")
        return {}

    log.info("Generating reports for %d clusters...", len(clustered_groups))
    reports = {}

    for cluster_id, rows in sorted(clustered_groups.items()):
        cluster_label = rows[0].get("cluster_label") or f"Cluster {cluster_id}"
        total_items = len(rows)
        shown_items = min(total_items, MAX_ITEMS_PER_CLUSTER)

        log.info("=" * 60)
        log.info(
            "Cluster %d: '%s' — %d total, using top %d unique items",
            cluster_id, cluster_label, total_items, shown_items
        )

        bullet_log = _build_bullet_log(rows, max_items=MAX_ITEMS_PER_CLUSTER)

        if not bullet_log.strip():
            log.warning("Cluster %d has no usable summaries — skipping.", cluster_id)
            continue

        prompt = CLUSTER_REPORT_PROMPT.format(
            label=cluster_label,
            count=total_items,
            shown=shown_items,
            total=total_items,
            log_data=bullet_log,
        )

        report_text = _call_ollama(prompt, max_tokens=500)

        if report_text:
            reports[cluster_id] = report_text
            safe_label = cluster_label.lower().replace(" ", "_")[:30]
            filename = f"cluster_{cluster_id}_{safe_label}.txt"
            _save_report(filename, report_text)
            for line in report_text.splitlines():
                log.info(line)
        else:
            log.warning("Failed to generate report for Cluster %d", cluster_id)

        if cluster_id < max(clustered_groups.keys()):
            time.sleep(1)

    log.info("=" * 60)
    log.info("Cluster reports saved to: ./%s/", REPORTS_DIR)
    return reports


# ---------------------------------------------------------------------------
# Master report generation
# ---------------------------------------------------------------------------

def generate_master_report(db_path: str, cluster_reports: Optional[dict] = None) -> Optional[str]:
    """
    Generates a master overview report across all clusters.
    Saved to reports/master.txt
    """
    if cluster_reports is None:
        cluster_reports = generate_cluster_reports(db_path)

    if not cluster_reports:
        log.warning("No cluster reports available for master report.")
        return None

    grouped = load_results_grouped_by_cluster(db_path)
    cluster_summaries = []

    for cluster_id, rows in sorted({k: v for k, v in grouped.items() if k >= 0}.items()):
        cluster_label = rows[0].get("cluster_label") or f"Cluster {cluster_id}"

        # Deduplicate summaries for master prompt too
        seen = set()
        unique_summaries = []
        for r in rows:
            s = (r.get("summary") or "").strip()
            if s and s not in seen:
                seen.add(s)
                unique_summaries.append(s)
            if len(unique_summaries) >= MAX_ITEMS_MASTER:
                break

        cluster_summaries.append(
            f"CLUSTER {cluster_id} — '{cluster_label}' ({len(rows)} items):\n" +
            "\n".join(f"  • {s[:120]}" for s in unique_summaries)
        )

    prompt = MASTER_REPORT_PROMPT.format(
        cluster_summaries="\n\n".join(cluster_summaries)
    )

    log.info("Generating MASTER report...")
    report_text = _call_ollama(prompt, max_tokens=600)

    if report_text:
        _save_report("master.txt", report_text)
        log.info("=" * 60)
        log.info("MASTER REPORT")
        log.info("=" * 60)
        for line in report_text.splitlines():
            log.info(line)
        log.info("=" * 60)
        log.info("Saved to: ./%s/master.txt", REPORTS_DIR)

    return report_text


# ---------------------------------------------------------------------------
# Offline report (no model calls)
# ---------------------------------------------------------------------------

def generate_offline_report(db_path: str, limit: Optional[int] = None) -> None:
    """Prints a grouped summary by cluster — no model calls."""
    grouped = load_results_grouped_by_cluster(db_path)

    if not grouped:
        log.warning("No results found.")
        return

    log.info("=" * 60)
    log.info("OFFLINE REPORT — grouped by cluster")
    log.info("=" * 60)

    for cluster_id, rows in sorted(grouped.items()):
        label = rows[0].get("cluster_label") or (
            f"Cluster {cluster_id}" if cluster_id >= 0 else "Unclustered"
        )
        log.info("")
        log.info("── CLUSTER %s: %s (%d items) ──", cluster_id, label, len(rows))

        display_rows = rows[:limit] if limit else rows
        for idx, item in enumerate(display_rows, start=1):
            date_str = _format_timestamp(item.get("timestamp"))
            category = item.get("category") or "?"
            summary = item.get("summary") or ""
            if not summary or summary.startswith("[Classification failed"):
                summary = "[Raw] " + (item.get("original_text") or "")[:150].replace("\n", " ") + "..."
            log.info("  %2d. [%s] [%s] %s", idx, date_str, category, summary[:120])

    log.info("=" * 60)
