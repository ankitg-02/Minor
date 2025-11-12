"""
Custom exceptions and handler for structured error management.
"""

import traceback
from utils.logger import get_logger

log = get_logger(__name__)

class YouTubeAPIError(Exception): pass
class DataIngestionError(Exception): pass
class DataProcessingError(Exception): pass
class ConfigurationError(Exception): pass

def handle_exception(e, context: str):
    """Log error message and traceback."""
    log.error(f"❌ Exception in {context}: {type(e).__name__} - {e}")
    log.debug("Traceback:\n" + "".join(traceback.format_exception(None, e, e.__traceback__)))
