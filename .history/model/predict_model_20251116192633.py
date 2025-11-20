"""
Use trained model to predict sentiment on processed comments.
"""

import pandas as pd
import os

from utils.logger import get_logger
from utils.file_helper import get_latest_file
from utils.config import PROCESSED_DIR
from model.model_utils import load_model
from utils.timer import timer
from utils.exceptions import handle_exception

log = get_logger(__name__)

@timer
def generate_predictions():
    try:
        log.info("🔮 Generating sentiment predictions...")

        model, vectorizer = load_model("model/sentiment_model.pkl")
        cleaned_path = get_latest_file(PROCESSED_DIR)
        df = pd.read_csv(cleaned_path)

        X = vectorizer.transform(df["cleaned_text"])
        df["predicted_sentiment"] = model.predict(X)

        out_path = os.path.join(
            PROCESSED_DIR, 
            cleaned_path.replace(".csv", "_sentiment.csv")
        )

        df.to_csv(out_path, index=False)
        log.info(f"📈 Sentiment results saved at: {out_path}")

        return out_path

    except Exception as e:
        handle_exception(e, "model_prediction")
        raise
