"""Shared logger factory for all pipeline modules."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(f"credit_scoring.{name}")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        log.addHandler(handler)
        log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return log
