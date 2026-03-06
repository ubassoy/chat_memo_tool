"""
classifier.py — Sends message pairs to local Ollama for structured classification.

Uses Ollama's REST API directly — no API key, no cost, fully offline.
Features:
  - Automatic retry with exponential backoff (via tenacity)
  - Strict JSON parsing with fallback
  - Typed ClassificationResult output
  - Rate-limit sleep between calls
"""

import json
import time
import requests
from dataclasses import dataclass
from typing import Optional

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
# Ollama config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"  # Change this if you switch models later


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """You are a project analyst. Analyze the conversation below.
Return ONLY a JSON object — no explanation, no markdown, no extra text.

Required format:
{{
  "summary":  "<one concise sentence describing what was accomplished or decided>",
  "category": "<one of: Architecture, Bug Fix, Feature, Research, Planning, Integration, Testing, Documentation, Other>",
  "priority": "<one of: High, Medium, Low>"
}}

Conversation:
{text}"""


# ---------------------------------------------------------------------------
# Ollama API call with retry
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    before_sleep=before_sleep_log(log, 20),
    reraise=False,
)
def _call_ollama_with_retry(text: str) -> str:
    """Calls local Ollama REST API — retried automatically on failure."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": CLASSIFICATION_PROMPT.format(text=text),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json_safely(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed: %s | Raw: %.200s", exc, raw)
        return {}


# ---------------------------------------------------------------------------
# Public classify interface
# ---------------------------------------------------------------------------

def classify_message(text: str, max_chars: int = 15000) -> ClassificationResult:
    safe_text = text[:max_chars]
    raw_response = _call_ollama_with_retry(safe_text)

    if not raw_response:
        log.error("Ollama returned an empty response.")
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
# Batch classification
# ---------------------------------------------------------------------------

def classify_batch(
    records: list,
    on_classified: callable,
    rate_limit_seconds: int = 2,  # Much shorter — local model, no API quota
) -> None:
    total = len(records)

    for idx, record in enumerate(records, start=1):
        log.info("[%d/%d] Classifying: '%s'", idx, total, record.conversation_title)

        result = classify_message(
            text=record.full_context,
            max_chars=cfg.max_content_chars,
        )

        on_classified(record, result)

        if idx < total:
            log.debug("Pause: %ds", rate_limit_seconds)
            time.sleep(rate_limit_seconds)

    log.info("Batch classification complete. %d records processed.", total)


# ---------------------------------------------------------------------------
# Quick connection test
# ---------------------------------------------------------------------------

def test_ollama_connection() -> bool:
    """Call this on startup to verify Ollama is reachable before processing."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]

        if OLLAMA_MODEL not in models:
            log.warning(
                "Model '%s' not found. Available: %s — Run: ollama pull %s",
                OLLAMA_MODEL, models, OLLAMA_MODEL,
            )
            return False

        log.info("Ollama connected. Model '%s' ready.", OLLAMA_MODEL)
        return True

    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama at 127.0.0.1:11434. Make sure Ollama is running.")
        return False