"""
Logging configuration for Adaptive LLM Routing
"""

import os
import logging
import sys

def setup_logger(name: str = "adaptive_llm", log_file: str = "results/app.log", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass

    return logger

logger = setup_logger()
