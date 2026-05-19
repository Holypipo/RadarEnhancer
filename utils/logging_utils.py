"""
Project logging helpers.

CLI scripts keep their normal console output, while every stdout/stderr line is
also copied into a timestamped file under logs/.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from config import LOG_DIR


class _TeeStream:
    """Write text to the original terminal stream and to a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def configure_logging(name: str) -> logging.Logger:
    """
    Configure terminal + file logging for a script run.

    Existing print() calls are intentionally preserved and captured. The
    returned logger is available for new structured logging when needed.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{name}_{timestamp}.log"
    log_file = open(log_path, "a", encoding="utf-8")

    original_stdout = getattr(sys.stdout, "terminal", sys.stdout)
    original_stderr = getattr(sys.stderr, "terminal", sys.stderr)
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialized: %s", Path(log_path).resolve())
    print(f"Log file: {Path(log_path).resolve()}")
    return logger
