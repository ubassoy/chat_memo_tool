"""
logger.py — Centralised logging setup.
All modules import `log` from here instead of using print().
"""

import logging
import sys


def _build_logger(name: str = "chat_memo") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # File handler — DEBUG and above (full trace)
    file_handler = logging.FileHandler("chat_memo.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  [%(module)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


log = _build_logger()
