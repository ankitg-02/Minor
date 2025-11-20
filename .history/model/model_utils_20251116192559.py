"""
Utilities for saving & loading ML models.
"""

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
