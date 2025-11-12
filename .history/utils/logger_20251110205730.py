"""
Centralized logger utility.
"""

import logging
import os
from datetime import datetime

from .config import LOG_DIR, LOG_LEVEL

def get_logger(name: str):
    """Return a logger with file and console handlers."""
    log_file = os.path.join(LOG_DIR, f"project_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger(name)
    
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
