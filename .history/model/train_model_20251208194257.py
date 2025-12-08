"""
Train multiple sentiment models using cleaned YouTube comments and compare their performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from scipy.sparse import hstack
import joblib
import json
import numpy as np
from typing import Tuple, List, Set, Dict, Any

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will be skipped.")

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

def get_models() -> Dict[str, Any]:
    """Get dictionary of models to train and compare."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42),
        "NaiveBayes": MultinomialNB()
    }

def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted')
    }

    log.info(f"📊 {model_name} Metrics:")
    log.info(f"   Accuracy: {metrics['accuracy']:.4f}")
    log.info(f"   F1 Macro: {metrics['f1_macro']:.4f}")
    log.info(f"   F1 Weighted: {metrics['f1_weighted']:.4f}")

    return metrics

@timer
def compare_models() -> Dict[str, Any]:
    """Train and compare multiple sentiment classification models."""
    try:
        log.info("🎓 Training and comparing sentiment models...")

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
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )

        models = get_models()
        results = {}
        trained_models = {}

        log.info("🏃 Training and evaluating models...")

        for model_name, model in models.items():
            log.info(f"🔄 Training {model_name}...")

            # Special handling for Naive Bayes (needs non-negative features)
            if model_name == "NaiveBayes":
                # Use MaxAbsScaler for Naive Bayes since it needs non-negative values
                nb_scaler = MaxAbsScaler()
                X_numeric_nb = nb_scaler.fit_transform(numeric_features)
                X_nb = hstack([X_text, X_numeric_nb])
                X_train_nb, X_test_nb = train_test_split(X_nb, test_size=0.2, random_state=42, stratify=y)[0:2]

                model.fit(X_train_nb, y_train)
                metrics = evaluate_model(model, X_test_nb, y_test, model_name)
                trained_models[model_name] = {
                    "model": model,
                    "scaler": nb_scaler,
                    "X_train": X_train_nb,
                    "X_test": X_test_nb
                }
            else:
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test, model_name)
                trained_models[model_name] = {
                    "model": model,
                    "scaler": scaler,
                    "X_train": X_train,
                    "X_test": X_test
                }

            results[model_name] = metrics

        # Select best model based on F1 weighted score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
        best_metrics = results[best_model_name]

        log.info(f"🏆 Best Model: {best_model_name} (F1 Weighted: {best_metrics['f1_weighted']:.4f})")

        # Save comparison results
        comparison_results = {
            "models": results,
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "training_info": {
                "dataset_size": len(df),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "sentiment_distribution": df["sentiment"].value_counts().to_dict()
            }
        }

        with open("model/model_comparison.json", "w") as f:
            json.dump(comparison_results, f, indent=2)

        # Save best model
        best_model_data = trained_models[best_model_name]
        save_model(best_model_data["model"], vectorizer, path="model/sentiment_model.pkl")
        joblib.dump(best_model_data["scaler"], "model/feature_scaler.pkl")

        log.info("✅ Best model, vectorizer, and scaler saved successfully.")
        log.info("📊 Model comparison results saved to model/model_comparison.json")

        return comparison_results

    except Exception as e:
        handle_exception(e, "model_comparison")
        raise

@timer
def train_sentiment_model() -> None:
    """Train the best sentiment classification model based on comparison."""
    try:
        log.info("🎓 Training best sentiment model...")

        # Run model comparison and get results
        comparison_results = compare_models()

        best_model = comparison_results["best_model"]
        best_metrics = comparison_results["best_metrics"]

        log.info(f"✅ Training completed. Best model: {best_model}")
        log.info(f"📊 Best model F1 Score: {best_metrics['f1_weighted']:.4f}")

    except Exception as e:
        handle_exception(e, "model_training")
        raise
