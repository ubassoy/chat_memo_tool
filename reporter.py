"""
reporter.py — Generates strategic project reports from classified data.

Two modes:
  1. AI report   — Sends a summary log to local Ollama for a rich strategic report
  2. Offline report — Prints a local summary without any model calls
"""

import requests
from datetime import datetime
from typing import Optional

from config import cfg
from logger import log
from storage import load_results


# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_timestamp(ts) -> str:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    if isinstance(ts, str):
        return ts[:10]
    return "Unknown Date"


def _build_bullet_log(rows: list[dict]) -> str:
    """Converts DB rows into a bullet-point context block for the prompt."""
    lines = []
    for item in rows:
        date_str = _format_timestamp(item.get("timestamp"))
        category = item.get("category") or "General"
        summary = item.get("summary") or "No summary available."
        lines.append(f"- [{date_str}] [{category}]: {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI-powered report via Ollama
# ---------------------------------------------------------------------------

AI_REPORT_PROMPT = """You are a senior technical project manager reviewing a developer's work history. Always respond in English only.

Based ONLY on the work log below, write a clear 'Strategic Status Report' with these 4 sections:

## 1. CURRENT FOCUS
What is the developer actively working on right now?

## 2. KEY DECISIONS MADE
List the most impactful architectural or design choices.

## 3. POTENTIAL BLOCKERS
Identify risks, unresolved issues, or areas of uncertainty.

## 4. SUGGESTED NEXT STEPS
Suggest exactly 3 concrete, actionable tasks to move the project forward.

WORK LOG:
{log_data}"""


def generate_ai_report(
    db_path: str,
    limit: Optional[int] = None,
) -> Optional[str]:
    """
    Fetches the latest N results, builds a prompt, and asks local Ollama
    to generate a strategic project status report.
    """
    item_limit = limit or cfg.report_item_limit
    rows = load_results(db_path, limit=item_limit)

    if not rows:
        log.warning("No results found in DB — cannot generate report.")
        return None

    bullet_log = _build_bullet_log(rows)

    # Print raw summary first as a sanity check
    log.info("=" * 60)
    log.info("RAW SUMMARY — last %d items", len(rows))
    log.info("=" * 60)
    for line in bullet_log.splitlines()[:20]:
        log.info(line)
    if len(bullet_log.splitlines()) > 20:
        log.info("... (%d more lines)", len(bullet_log.splitlines()) - 20)

    prompt = AI_REPORT_PROMPT.format(log_data=bullet_log)

    log.info("Requesting AI strategic report from local Ollama...")

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 800,   # Enough for a full structured report
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        report_text = response.json().get("response", "")

        log.info("=" * 60)
        log.info("STRATEGIC STATUS REPORT")
        log.info("=" * 60)
        for line in report_text.splitlines():
            log.info(line)
        log.info("=" * 60)

        # Also save to a clean file
        with open("report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        log.info("Report saved to: report.txt")

        return report_text

    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama. Make sure it is running.")
        return None
    except Exception as exc:
        log.error("Report generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Offline report (no model calls)
# ---------------------------------------------------------------------------

def generate_offline_report(db_path: str, limit: Optional[int] = None) -> None:
    """
    Prints a human-readable summary of stored results
    without making any model calls.
    """
    item_limit = limit or cfg.report_item_limit
    rows = load_results(db_path, limit=item_limit)

    if not rows:
        log.warning("No results found in the database.")
        return

    log.info("=" * 60)
    log.info("OFFLINE REPORT — last %d items", len(rows))
    log.info("=" * 60)

    for idx, item in enumerate(rows, start=1):
        date_str = _format_timestamp(item.get("timestamp"))
        title = item.get("title", "Untitled")
        category = item.get("category") or "?"
        priority = item.get("priority") or "?"

        summary = item.get("summary")
        if summary and not summary.startswith("[Classification failed"):
            content_line = f"[{category} / {priority}] {summary}"
        else:
            raw = item.get("original_text", "")
            content_line = "[Raw] " + raw[:200].replace("\n", " ") + "..."

        log.info("%3d. [%s] %s", idx, date_str, title)
        log.info("      %s", content_line)

    log.info("=" * 60)
    log.info("Total: %d results shown.", len(rows))
