"""
classifier.py — Sends message pairs to Gemini for structured classification.

Features:
  - Automatic retry with exponential backoff (via tenacity)
  - Strict JSON parsing with fallback
  - Typed ClassificationResult output
  - Rate-limit sleep between calls
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import cfg
from logger import log


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    summary: str
    category: str
    priority: str
    raw: dict

    @classmethod
    def empty(cls, reason: str = "unknown") -> "ClassificationResult":
        return cls(
            summary=f"[Classification failed: {reason}]",
            category="Unknown",
            priority="Unknown",
            raw={},
        )


# ---------------------------------------------------------------------------
# Gemini client (lazy singleton)
# ---------------------------------------------------------------------------

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=cfg.gemini_api_key)
        log.debug("Gemini client initialised.")
    return _client


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """
You are a project analyst extracting structured insights from AI conversation logs.

Analyze the conversation below and return ONLY a JSON object — no markdown, no preamble.

Required JSON schema:
{{
  "summary":  "<one concise sentence describing what was accomplished or decided>",
  "category": "<one of: Architecture, Bug Fix, Feature, Research, Planning, Integration, Testing, Documentation, Other>",
  "priority": "<one of: High, Medium, Low>"
}}

Conversation:
{text}
"""


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    before_sleep=before_sleep_log(log, log.level),
    reraise=False,
)
def _call_gemini_with_retry(text: str) -> str:
    """Raw Gemini call — retried automatically on failure."""
    client = _get_client()
    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents=CLASSIFICATION_PROMPT.format(text=text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return response.text


def _parse_json_safely(raw: str) -> dict:
    """
    Strips markdown fences if present, then parses JSON.
    Returns empty dict on failure instead of raising.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` fences
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed: %s | Raw: %.200s", exc, raw)
        return {}


def classify_message(text: str, max_chars: int = 15000) -> ClassificationResult:
    """
    Public interface: classify a message string and return a typed result.
    Truncates text, handles retries, and always returns a valid object.
    """
    safe_text = text[:max_chars]

    raw_response = _call_gemini_with_retry(safe_text)

    if not raw_response:
        log.error("Gemini returned an empty response.")
        return ClassificationResult.empty("empty response")

    parsed = _parse_json_safely(raw_response)

    if not parsed or "summary" not in parsed:
        log.warning("Parsed JSON missing required fields: %s", parsed)
        return ClassificationResult.empty("missing fields")

    return ClassificationResult(
        summary=parsed.get("summary", ""),
        category=parsed.get("category", "Other"),
        priority=parsed.get("priority", "Medium"),
        raw=parsed,
    )


# ---------------------------------------------------------------------------
# Batch classification with progress
# ---------------------------------------------------------------------------

def classify_batch(
    records: list,  # list[MessageRecord]
    on_classified: callable,
    rate_limit_seconds: int = 10,
) -> None:
    """
    Iterates over a list of MessageRecord objects, classifies each,
    and calls `on_classified(record, result)` after each success.

    The caller controls persistence — this function only classifies.
    """
    total = len(records)

    for idx, record in enumerate(records, start=1):
        log.info(
            "[%d/%d] Classifying: '%s'",
            idx,
            total,
            record.conversation_title,
        )

        result = classify_message(
            text=record.full_context,
            max_chars=cfg.max_content_chars,
        )

        on_classified(record, result)

        if idx < total:
            log.debug("Rate-limit pause: %ds", rate_limit_seconds)
            time.sleep(rate_limit_seconds)

    log.info("Batch classification complete. %d records processed.", total)
