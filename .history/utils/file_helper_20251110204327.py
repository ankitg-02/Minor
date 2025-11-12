"""
Reusable file helper utilities.
"""

import os
from datetime import datetime

def get_latest_file(folder: str, ext: str = ".csv"):
    """Return the most recently modified file path."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)]
    if not files:
        raise FileNotFoundError(f"No {ext} files found in {folder}")
    return max(files, key=os.path.getmtime)

def safe_filename(prefix: str, keyword: str, ext: str = ".csv"):
    """Generate timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{keyword.replace(' ', '_')}_{timestamp}{ext}"
