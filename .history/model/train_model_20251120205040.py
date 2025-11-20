"""
Train a sentiment model using cleaned YouTube comments.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path
import sys

# Ensure project root is on sys.path so `utils` imports work when running the file directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.config import PROCESSED_DIR
from utils.file_helper import get_latest_file
from model.features import get_tfidf_vectorizer
from model.model_utils import save_model
from utils.exception import DataProcessingError, handle_exception
from utils.timer import timer

import os

log = get_logger(__name__)

def rule_based_sentiment(text):
    """Simple lexicon-based sentiment labeling."""
    positive_words = ["good", "great", "love", "amazing", "nice", "best"]
    negative_words = ["bad", "hate", "terrible", "worst", "awful"]

    text_lower = text.lower()

    score = 0
    for w in positive_words:
        if w in text_lower:
            score += 1
    for w in negative_words:
        if w in text_lower:
            score -= 1

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

@timer
def train_sentiment_model():
    try:
        log.info("🎓 Training sentiment model...")

        cleaned_path = get_latest_file(PROCESSED_DIR)
        df = pd.read_csv(cleaned_path)

        if "cleaned_text" not in df.columns:
            raise DataProcessingError("cleaned_text column missing.")

        log.info("🔧 Labeling data using rule-based method...")
        df["sentiment"] = df["cleaned_text"].apply(rule_based_sentiment)

        log.info("🔠 Extracting TF-IDF features...")
        vectorizer = get_tfidf_vectorizer()
        X = vectorizer.fit_transform(df["cleaned_text"])
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        log.info(f"📊 Model Accuracy: {acc:.3f}")

        save_model(model, vectorizer, path="result/sentiment_model.pkl")

    except Exception as e:
        handle_exception(e, "model_training")
        raise
