"""
config.py — Central configuration loader.
Reads all settings from environment variables (.env file).
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    # --- Paths ---
    conversations_path: str = field(
        default_factory=lambda: os.getenv("CONVERSATIONS_PATH", "conversations.json")
    )
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "memo.db"))

    # --- Filtering ---
    title_prefix: str = field(default_factory=lambda: os.getenv("TITLE_PREFIX", "bjk"))

    # --- Behaviour ---
    rate_limit_seconds: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_SECONDS", "2"))
    )
    report_item_limit: int = field(
        default_factory=lambda: int(os.getenv("REPORT_ITEM_LIMIT", "25"))
    )
    max_content_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTENT_CHARS", "15000"))
    )
    ai_response_preview_chars: int = field(
        default_factory=lambda: int(os.getenv("AI_RESPONSE_PREVIEW_CHARS", "5000"))
    )
    num_clusters: int = field(
        default_factory=lambda: int(os.getenv("NUM_CLUSTERS", "5"))
    )

    def validate(self) -> None:
        """Raises ValueError if any required field is missing."""
        if not self.conversations_path:
            raise ValueError("CONVERSATIONS_PATH is not set in .env")


# Singleton — import this everywhere
cfg = AppConfig()