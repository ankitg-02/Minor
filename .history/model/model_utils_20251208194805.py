"""
Utilities for saving & loading ML models and loading processed data with metadata features.
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path so `utils` imports work when running the file directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import os
import pandas as pd
from utils.logger import get_logger

log = get_logger(__name__)

def save_model(model, vectorizer, path="model/sentiment_model.pkl", **kwargs):
    """Save ML model + vectorizer (supports both traditional ML and deep learning models)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Check if this is an LSTM model (has tokenizer and label_encoder)
    if "tokenizer" in kwargs and "label_encoder" in kwargs:

def load_processed_data(csv_path):
    """
    Load processed comments data CSV with cleaned text and metadata features.

    Args:
        csv_path (str): Path to the processed CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing cleaned_text and metadata columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed data file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    log.info(f"📥 Loaded processed data from {csv_path} with {len(df)} rows")

    # Validate there is a cleaned_text column
    if "cleaned_text" not in df.columns:
        raise ValueError("Processed data must contain 'cleaned_text' column")

    # Optional metadata columns typically from ingestion
    metadata_cols = [col for col in ["like_count", "view_count", "published_at"] if col in df.columns]

    # We return the whole dataframe; caller can select columns as needed
    return df

