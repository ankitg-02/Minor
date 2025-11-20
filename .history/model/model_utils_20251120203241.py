"""
Utilities for saving & loading ML models.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path so `utils` imports work when running the file directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import os
from utils.logger import get_logger

log = get_logger(__name__)

def save_model(model, vectorizer, path="model/sentiment_model.pkl"):
    """Save ML model + vectorizer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, path)
    log.info(f"💾 Model saved at {path}")

def load_model(path="model/sentiment_model.pkl"):
    """Load trained ML model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    data = joblib.load(path)
    log.info("📥 Loaded trained ML model.")
    return data["model"], data["vectorizer"]
