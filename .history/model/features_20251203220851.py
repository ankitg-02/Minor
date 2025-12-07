"""
Feature extraction using TF-IDF and optional metadata features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import hstack

def get_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a configurable TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

def combine_text_and_metadata_features(text_features, metadata_features) -> Any:
    """
    Combine sparse text features and dense metadata features into a single feature matrix.

    Args:
        text_features: Sparse matrix from TF-IDF vectorizer.
        metadata_features: 2D numpy array or similar for metadata features like counts.

    Returns:
        Combined feature matrix in sparse or dense form, depending on input.
    """
    if metadata_features is None or len(metadata_features) == 0:
        return text_features
    return hstack([text_features, metadata_features])
