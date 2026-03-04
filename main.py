"""
main.py — Entry point and CLI for the Chat Memo Tool.

Usage:
    python main.py                  # Full pipeline: extract → classify → cluster → report
    python main.py --mode classify  # Extract + classify only (no report)
    python main.py --mode report    # Generate AI report from existing data
    python main.py --mode offline   # Print offline report (no API calls)
    python main.py --mode cluster   # Run clustering only
    python main.py --mode cleanup   # Remove failed/empty results from DB
    python main.py --limit 50       # Override report item limit
"""

import argparse
import sys

from config import cfg
from logger import log


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_extract_and_classify() -> None:
    """Step 1+2: Load conversations, extract new messages, classify them."""
    from extractor import load_conversations, extract_new_messages
    from classifier import classify_batch
    from storage import (
        init_db,
        load_processed_ids,
        mark_id_processed,
        save_result,
        get_result_count,
    )

    init_db(cfg.db_path)

    processed_ids = load_processed_ids(cfg.db_path)
    log.info("Previously processed: %d messages.", len(processed_ids))

    conversations = load_conversations(cfg.conversations_path)

    new_records = extract_new_messages(
        conversations=conversations,
        prefix=cfg.title_prefix,
        processed_ids=processed_ids,
        preview_chars=cfg.ai_response_preview_chars,
    )

    if not new_records:
        log.info("No new messages found — everything is up to date.")
        return

    log.info("Starting classification of %d new messages...", len(new_records))

    # Closure captures DB path so classify_batch stays pure
    def on_classified(record, result) -> None:
        entry = {
            "message_id": record.node_id,
            "title": record.conversation_title,
            "original_text": record.full_context,
            "timestamp": record.timestamp,
            "analysis": {
                "summary": result.summary,
                "category": result.category,
                "priority": result.priority,
            },
        }
        save_result(cfg.db_path, entry)
        mark_id_processed(cfg.db_path, record.node_id)
        log.debug("Saved: %s", record.node_id)

    classify_batch(
        records=new_records,
        on_classified=on_classified,
        rate_limit_seconds=cfg.rate_limit_seconds,
    )

    total = get_result_count(cfg.db_path)
    log.info("Classification done. Total results in DB: %d", total)


def step_cluster() -> None:
    """Step 3 (optional): Semantic clustering of all results."""
    from clustering import run_clustering

    log.info("Running clustering...")
    run_clustering(db_path=cfg.db_path, num_clusters=cfg.num_clusters)


def step_ai_report(limit: int) -> None:
    """Step 4: Generate AI strategic report via Gemini."""
    from reporter import generate_ai_report

    generate_ai_report(db_path=cfg.db_path, limit=limit)


def step_offline_report(limit: int) -> None:
    """Alternative Step 4: Offline report with no API calls."""
    from reporter import generate_offline_report

    generate_offline_report(db_path=cfg.db_path, limit=limit)


def step_cleanup() -> None:
    """Utility: Remove failed/empty classification rows."""
    from storage import delete_failed_results

    removed = delete_failed_results(cfg.db_path)
    log.info("Cleanup complete. Removed %d failed entries.", removed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat Memo Tool — Extract, classify, and report on ChatGPT conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "classify", "report", "offline", "cluster", "cleanup"],
        default="full",
        help="Pipeline mode to run (default: full)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of items to include in the report",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    report_limit = args.limit or cfg.report_item_limit

    log.info("=" * 60)
    log.info("Chat Memo Tool — mode: %s", args.mode)
    log.info("Project prefix : %s", cfg.title_prefix)
    log.info("Database       : %s", cfg.db_path)
    log.info("=" * 60)

    try:
        cfg.validate()
    except ValueError as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    if args.mode == "full":
        step_extract_and_classify()
        step_cluster()
        step_ai_report(limit=report_limit)

    elif args.mode == "classify":
        step_extract_and_classify()

    elif args.mode == "report":
        step_ai_report(limit=report_limit)

    elif args.mode == "offline":
        step_offline_report(limit=report_limit)

    elif args.mode == "cluster":
        step_cluster()

    elif args.mode == "cleanup":
        step_cleanup()

    log.info("Done.")


if __name__ == "__main__":
    main()
