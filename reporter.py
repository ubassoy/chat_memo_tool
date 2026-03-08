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
OLLAMA_MODEL = "qwen2.5:0.5b"
REPORTS_DIR = "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_timestamp(ts) -> str:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    if isinstance(ts, str):
        return ts[:10]
    return "Unknown Date"


def _build_bullet_log(rows: list[dict]) -> str:
    lines = []
    for item in rows:
        date_str = _format_timestamp(item.get("timestamp"))
        category = item.get("category") or "General"
        summary = item.get("summary") or "No summary."
        lines.append(f"- [{date_str}] [{category}]: {summary}")
    return "\n".join(lines)


def _ensure_reports_dir() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _call_ollama(prompt: str, max_tokens: int = 800) -> Optional[str]:
    """Sends a prompt to local Ollama and returns the response text."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens,
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

CLUSTER_REPORT_PROMPT = """You are a senior technical project manager. 
Always respond in English only. Do not use Chinese or any other language.

Below is a log of work from a specific topic cluster called '{label}'.
Write a focused 'Cluster Status Report' in English with these 4 sections:

## CLUSTER: {label} ({count} conversations)

### 1. WHAT THIS CLUSTER IS ABOUT
(One paragraph describing the theme of this group of conversations.)

### 2. KEY INSIGHTS & DECISIONS
(The most important things learned or decided in this cluster.)

### 3. OPEN QUESTIONS / BLOCKERS
(Anything unresolved or worth revisiting.)

### 4. SUGGESTED NEXT STEPS
(2-3 concrete actions specific to this cluster.)

WORK LOG:
{log_data}"""


# ---------------------------------------------------------------------------
# Master report prompt
# ---------------------------------------------------------------------------

MASTER_REPORT_PROMPT = """You are a senior technical project manager.
Always respond in English only. Do not use Chinese or any other language.

Below is a high-level summary across all project clusters.
Write a 'Master Strategic Report' in English with these sections:

## MASTER PROJECT REPORT

### 1. OVERALL PROJECT STATUS
(What is the developer's main focus across all clusters?)

### 2. TOP 3 PRIORITIES RIGHT NOW
(The most important things to work on based on all clusters combined.)

### 3. CROSS-CLUSTER CONNECTIONS
(Are there any clusters that relate to or depend on each other?)

### 4. RECOMMENDED NEXT STEPS
(3 concrete actions that would move the overall project forward most.)

CLUSTER SUMMARIES:
{cluster_summaries}"""


# ---------------------------------------------------------------------------
# Per-cluster report generation
# ---------------------------------------------------------------------------

def generate_cluster_reports(db_path: str) -> dict[int, str]:
    """
    Generates one report per cluster and saves each to reports/cluster_N.txt
    Returns a dict of {cluster_id: report_text}
    """
    grouped = load_results_grouped_by_cluster(db_path)

    if not grouped:
        log.warning("No clustered data found. Run clustering first: python main.py --mode cluster")
        return {}

    # Remove unclustered items (key = -1)
    clustered_groups = {k: v for k, v in grouped.items() if k >= 0}

    if not clustered_groups:
        log.warning("No cluster assignments found in DB. Run: python main.py --mode cluster")
        return {}

    log.info("Generating reports for %d clusters...", len(clustered_groups))
    reports = {}

    for cluster_id, rows in sorted(clustered_groups.items()):
        cluster_label = rows[0].get("cluster_label") or f"Cluster {cluster_id}"
        log.info("=" * 60)
        log.info("Generating report for Cluster %d: '%s' (%d items)", cluster_id, cluster_label, len(rows))

        bullet_log = _build_bullet_log(rows)
        prompt = CLUSTER_REPORT_PROMPT.format(
            label=cluster_label,
            count=len(rows),
            log_data=bullet_log,
        )

        report_text = _call_ollama(prompt, max_tokens=600)

        if report_text:
            reports[cluster_id] = report_text
            filename = f"cluster_{cluster_id}_{cluster_label.lower().replace(' ', '_')}.txt"
            _save_report(filename, report_text)

            # Print to console
            for line in report_text.splitlines():
                log.info(line)
        else:
            log.warning("Failed to generate report for Cluster %d", cluster_id)

        # Small pause between cluster reports
        if cluster_id < max(clustered_groups.keys()):
            time.sleep(2)

    log.info("=" * 60)
    log.info("Per-cluster reports saved to: ./%s/", REPORTS_DIR)
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

    # Build a short summary of each cluster for the master prompt
    grouped = load_results_grouped_by_cluster(db_path)
    cluster_summaries = []

    for cluster_id, rows in sorted({k: v for k, v in grouped.items() if k >= 0}.items()):
        cluster_label = rows[0].get("cluster_label") or f"Cluster {cluster_id}"
        top_summaries = [r.get("summary", "") for r in rows[:5] if r.get("summary")]
        cluster_summaries.append(
            f"CLUSTER {cluster_id} — '{cluster_label}' ({len(rows)} items):\n" +
            "\n".join(f"  • {s}" for s in top_summaries)
        )

    prompt = MASTER_REPORT_PROMPT.format(
        cluster_summaries="\n\n".join(cluster_summaries)
    )

    log.info("=" * 60)
    log.info("Generating MASTER report across all clusters...")

    report_text = _call_ollama(prompt, max_tokens=1000)

    if report_text:
        _save_report("master.txt", report_text)
        log.info("=" * 60)
        log.info("MASTER STRATEGIC REPORT")
        log.info("=" * 60)
        for line in report_text.splitlines():
            log.info(line)
        log.info("=" * 60)
        log.info("Master report saved to: ./%s/master.txt", REPORTS_DIR)

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
        label = rows[0].get("cluster_label") or (f"Cluster {cluster_id}" if cluster_id >= 0 else "Unclustered")
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
