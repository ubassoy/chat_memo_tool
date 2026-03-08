"""
main.py — Entry point and CLI for the Chat Memo Tool.

Usage:
    python main.py                      # Full pipeline
    python main.py --mode classify      # Extract + classify only
    python main.py --mode cluster       # Run clustering only
    python main.py --mode report        # Per-cluster reports + master report
    python main.py --mode report_cluster  # Per-cluster reports only
    python main.py --mode report_master   # Master report only
    python main.py --mode offline       # Offline grouped summary
    python main.py --mode cleanup       # Remove failed rows
"""

import argparse
import sys

from config import cfg
from logger import log


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_extract_and_classify() -> None:
    from extractor import load_conversations, extract_new_messages
    from classifier import classify_batch, test_ollama_connection
    from storage import (
        init_db, load_processed_ids, mark_id_processed,
        save_result, get_result_count,
    )

    init_db(cfg.db_path)

    # Verify Ollama is reachable before starting
    if not test_ollama_connection():
        log.error("Ollama is not running. Start it and try again.")
        sys.exit(1)

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

    classify_batch(
        records=new_records,
        on_classified=on_classified,
        rate_limit_seconds=cfg.rate_limit_seconds,
    )

    total = get_result_count(cfg.db_path)
    log.info("Classification done. Total results in DB: %d", total)


def step_cluster() -> None:
    from clustering import run_clustering
    log.info("Running clustering...")
    run_clustering(db_path=cfg.db_path, num_clusters=None, auto_find=True)


def step_cluster_reports() -> dict:
    from reporter import generate_cluster_reports
    return generate_cluster_reports(db_path=cfg.db_path)


def step_master_report(cluster_reports: dict = None) -> None:
    from reporter import generate_master_report
    generate_master_report(db_path=cfg.db_path, cluster_reports=cluster_reports)


def step_offline_report(limit: int) -> None:
    from reporter import generate_offline_report
    generate_offline_report(db_path=cfg.db_path, limit=limit)


def step_cleanup() -> None:
    from storage import delete_failed_results
    removed = delete_failed_results(cfg.db_path)
    log.info("Cleanup complete. Removed %d failed entries.", removed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat Memo Tool — Extract, classify, cluster and report on ChatGPT conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "classify", "cluster", "report", "report_cluster", "report_master", "offline", "cleanup"],
        default="full",
        help="Pipeline mode to run (default: full)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to show in offline report",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

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
        cluster_reports = step_cluster_reports()
        step_master_report(cluster_reports)

    elif args.mode == "classify":
        step_extract_and_classify()

    elif args.mode == "cluster":
        step_cluster()

    elif args.mode == "report":
        # Full report: per-cluster + master
        cluster_reports = step_cluster_reports()
        step_master_report(cluster_reports)

    elif args.mode == "report_cluster":
        step_cluster_reports()

    elif args.mode == "report_master":
        step_master_report()

    elif args.mode == "offline":
        step_offline_report(limit=args.limit)

    elif args.mode == "cleanup":
        step_cleanup()

    log.info("Done.")


if __name__ == "__main__":
    main()
