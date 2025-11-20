"""
Feature extraction using TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer():
    """Return a configurable TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
