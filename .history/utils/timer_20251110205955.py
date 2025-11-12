"""
Timer decorator for measuring execution time.
"""

import time
import os
import sys
from functools import wraps

# Handle both package and direct imports
try:
    from .logger import get_logger
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.logger import get_logger

log = get_logger(__name__)

def timer(func):
    """Decorator to measure runtime of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        log.info(f"⏱️  {func.__name__} executed in {time.time()-start:.2f}s")
        return result
    return wrapper
