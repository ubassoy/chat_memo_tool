"""
reporter.py — Generates strategic project reports from classified data.

Two modes:
  1. AI report   — Sends a summary log to Gemini for a rich strategic report
  2. Offline report — Prints a local summary without any API calls
"""

import time
from datetime import datetime
from typing import Optional

from config import cfg
from logger import log
from storage import load_results


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
    """Converts DB rows into a bullet-point context block for the AI prompt."""
    lines = []
    for item in rows:
        date_str = _format_timestamp(item.get("timestamp"))
        category = item.get("category") or "General"
        summary = item.get("summary") or "No summary available."
        lines.append(f"- [{date_str}] [{category}]: {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI-powered report
# ---------------------------------------------------------------------------

AI_REPORT_PROMPT = """
You are a senior technical project manager reviewing a developer's work history.

Below is a chronological log of tasks and decisions on project '{prefix}'.
Based ONLY on this data, produce a concise 'Strategic Status Report'.

The report MUST follow this exact structure:

## 1. CURRENT FOCUS
(What is the developer actively working on right now?)

## 2. KEY DECISIONS MADE
(List the most impactful architectural or design choices.)

## 3. POTENTIAL BLOCKERS
(Identify risks, unresolved issues, or areas of uncertainty.)

## 4. SUGGESTED NEXT STEPS
(Suggest exactly 3 concrete, actionable tasks to move the project forward.)

---
WORK LOG ({count} entries):
{log_data}
"""


def generate_ai_report(
    db_path: str,
    limit: Optional[int] = None,
    pre_call_pause: int = 60,
) -> Optional[str]:
    """
    Fetches the latest N results, builds a prompt, and asks Gemini
    to generate a strategic project status report.

    Returns the report text, or None on failure.
    """
    from google import genai

    item_limit = limit or cfg.report_item_limit
    rows = load_results(db_path, limit=item_limit)

    if not rows:
        log.warning("No results found in DB — cannot generate report.")
        return None

    bullet_log = _build_bullet_log(rows)

    log.info("=" * 60)
    log.info("RAW SUMMARY — last %d items", len(rows))
    log.info("=" * 60)
    # Print first 20 lines for a quick sanity-check without flooding the terminal
    preview_lines = bullet_log.splitlines()[:20]
    for line in preview_lines:
        log.info(line)
    if len(bullet_log.splitlines()) > 20:
        log.info("... (%d more lines)", len(bullet_log.splitlines()) - 20)

    prompt = AI_REPORT_PROMPT.format(
        prefix=cfg.title_prefix.upper(),
        count=len(rows),
        log_data=bullet_log,
    )

    if pre_call_pause > 0:
        log.info(
            "Pausing %ds before report generation to respect API quota...",
            pre_call_pause,
        )
        time.sleep(pre_call_pause)

    log.info("Requesting AI strategic report from Gemini...")

    try:
        client = genai.Client(api_key=cfg.gemini_api_key)
        response = client.models.generate_content(
            model=cfg.gemini_model,
            contents=prompt,
        )
        report_text: str = response.text

        log.info("=" * 60)
        log.info("STRATEGIC STATUS REPORT")
        log.info("=" * 60)
        # Print each line through the logger so it lands in the log file too
        for line in report_text.splitlines():
            log.info(line)
        log.info("=" * 60)

        return report_text

    except Exception as exc:
        log.error("Gemini report generation failed: %s", exc)
        log.info(
            "TIP: You have the raw bullet log above — "
            "paste it into any AI assistant manually."
        )
        return None


# ---------------------------------------------------------------------------
# Offline report (no API calls)
# ---------------------------------------------------------------------------

def generate_offline_report(db_path: str, limit: Optional[int] = None) -> None:
    """
    Prints a human-readable summary of stored results
    without making any API calls.
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
        if summary and summary != "[Classification failed: unknown]":
            content_line = f"[{category} / {priority}] {summary}"
        else:
            raw = item.get("original_text", "")
            content_line = "[Raw] " + raw[:200].replace("\n", " ") + "..."

        log.info("%3d. [%s] %s", idx, date_str, title)
        log.info("      %s", content_line)

    log.info("=" * 60)
    log.info("Total: %d results shown.", len(rows))
