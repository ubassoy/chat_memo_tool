"""
config.py — Central configuration loader.
Reads all settings from environment variables (.env file).
Never hardcode secrets here.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    # --- API ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))

    # --- Paths ---
    conversations_path: str = field(
        default_factory=lambda: os.getenv("CONVERSATIONS_PATH", "conversations.json")
    )
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "memo.db"))

    # --- Filtering ---
    title_prefix: str = field(default_factory=lambda: os.getenv("TITLE_PREFIX", "bjk"))

    # --- Behaviour ---
    rate_limit_seconds: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_SECONDS", "10"))
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
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Create a .env file and add: GEMINI_API_KEY=your_key_here"
            )
        if not self.conversations_path:
            raise ValueError("CONVERSATIONS_PATH is not set in .env")


# Singleton — import this everywhere
cfg = AppConfig()
