"""
Train a sentiment model using cleaned YouTube comments.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

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
        return "good"
    elif score < 0:
        return "bad"
    else:
        return "neutral"

@timer
def train_sentiment_model() -> None:
    """Train a sentiment classification model using TF-IDF and numeric features."""
    try:
        log.info("🎓 Training sentiment model...")

        cleaned_path = get_latest_file(PROCESSED_DIR)
        df = pd.read_csv(cleaned_path)

        required_columns = {"cleaned_text", "like_count", "view_count"}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise DataProcessingError(f"Missing columns in data: {missing_cols}")

        log.info("🔧 Labeling data using rule-based method...")
        df["sentiment"] = df["cleaned_text"].apply(rule_based_sentiment)

        log.info("🔠 Extracting TF-IDF features...")
        vectorizer = get_tfidf_vectorizer()
        X_text = vectorizer.fit_transform(df["cleaned_text"])

        log.info("🔢 Processing numeric features...")
        numeric_features = df[["like_count", "view_count"]].fillna(0)
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(numeric_features)

        # Combine TF-IDF sparse matrix with scaled numeric features
        X_combined = hstack([X_text, X_numeric])

        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        log.info(f"📊 Model Accuracy: {acc:.3f}")

        save_model(model, vectorizer, path="model/sentiment_model.pkl")
        joblib.dump(scaler, "model/feature_scaler.pkl")
        log.info("✅ Model, vectorizer, and scaler saved successfully.")

    except Exception as e:
        handle_exception(e, "model_training")
        raise
