"""
Train multiple sentiment models using cleaned YouTube comments and compare their performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from scipy.sparse import hstack
import joblib
import json
from typing import Tuple, List, Set, Dict, Any

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
