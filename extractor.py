"""
extractor.py — Loads and parses the ChatGPT conversations.json export.

Responsibilities:
  1. Load and validate the JSON file.
  2. Filter conversations by title prefix.
  3. Walk the node tree to pair user prompts with their AI responses.
  4. Skip already-processed node IDs.
  5. Return clean, typed MessageRecord dicts.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logger import log


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class MessageRecord:
    node_id: str
    conversation_title: str
    user_text: str
    ai_text: str
    timestamp: Optional[float]

    @property
    def full_context(self) -> str:
        """Combined prompt+response string sent for classification."""
        return (
            f"MY PROMPT:\n{self.user_text}\n\n"
            f"AI RESPONSE:\n{self.ai_text}"
        )


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_conversations(path: str) -> list[dict]:
    """
    Loads a ChatGPT export JSON.
    Accepts both list format and {"conversations": [...]} format.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Conversations file not found: {path}")

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    else:
        raise ValueError(
            "Unexpected JSON structure. "
            "Expected a list or {'conversations': [...]}."
        )

    log.info("Loaded %d conversations from %s", len(conversations), path)
    return conversations


# ---------------------------------------------------------------------------
# Tree walking helpers
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: dict) -> str:
    """Safely joins 'parts' from a message content block."""
    parts = content.get("parts", [])
    return "".join(str(p) for p in parts if isinstance(p, str))


def _find_best_ai_response(
    mapping: dict,
    user_node: dict,
    preview_chars: int,
) -> str:
    """
    Walks ALL children of a user node (not just [0]) to find
    the first assistant response. Handles regenerated branches.
    """
    children_ids: list[str] = user_node.get("children", [])

    for child_id in children_ids:
        child_node = mapping.get(child_id, {})
        child_msg = child_node.get("message")
        if not child_msg:
            continue

        role = child_msg.get("author", {}).get("role", "")
        if role == "assistant":
            content = child_msg.get("content", {})
            text = _extract_text_from_content(content)
            return text[:preview_chars]

    # If no direct assistant child found, do a shallow BFS one level deeper
    for child_id in children_ids:
        child_node = mapping.get(child_id, {})
        grandchildren_ids = child_node.get("children", [])
        for gc_id in grandchildren_ids:
            gc_node = mapping.get(gc_id, {})
            gc_msg = gc_node.get("message")
            if not gc_msg:
                continue
            if gc_msg.get("author", {}).get("role") == "assistant":
                content = gc_msg.get("content", {})
                return _extract_text_from_content(content)[:preview_chars]

    return ""


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_new_messages(
    conversations: list[dict],
    prefix: str,
    processed_ids: set[str],
    preview_chars: int = 5000,
) -> list[MessageRecord]:
    """
    Filters conversations by prefix and returns only unprocessed
    user+AI message pairs as MessageRecord objects.
    """
    records: list[MessageRecord] = []
    skipped_processed = 0
    skipped_no_text = 0

    for conv in conversations:
        title: str = conv.get("title", "")

        if not title or not title.lower().startswith(prefix.lower()):
            continue

        mapping: dict = conv.get("mapping", {})

        for node_id, node in mapping.items():
            message = node.get("message")
            if not message:
                continue

            role = message.get("author", {}).get("role", "")
            if role != "user":
                continue

            # Skip already processed
            if node_id in processed_ids:
                skipped_processed += 1
                continue

            content = message.get("content", {})
            user_text = _extract_text_from_content(content)

            if not user_text.strip():
                skipped_no_text += 1
                continue

            ai_text = _find_best_ai_response(mapping, node, preview_chars)
            timestamp = message.get("create_time")

            records.append(
                MessageRecord(
                    node_id=node_id,
                    conversation_title=title,
                    user_text=user_text,
                    ai_text=ai_text,
                    timestamp=timestamp,
                )
            )

    log.info(
        "Extraction complete — new: %d | skipped (processed): %d | skipped (empty): %d",
        len(records),
        skipped_processed,
        skipped_no_text,
    )
    return records
