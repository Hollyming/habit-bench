"""Logging module - unified logging configuration."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def setup_logger(
    name: str = "personalization_lab",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console output
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File output
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_eval_logger(
    method_name: str,
    dataset_name: str,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Get evaluation logger."""
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{method_name}_{dataset_name}_{timestamp}.log"

    logger_name = f"eval.{method_name}.{dataset_name}"
    return setup_logger(logger_name, log_file=log_file)


# Global logger
_main_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get global logger."""
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_logger("personalization_lab")
    return _main_logger
