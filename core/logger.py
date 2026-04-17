from __future__ import annotations

import logging


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure app-wide logging once."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(level=level, format=_LOG_FORMAT)
