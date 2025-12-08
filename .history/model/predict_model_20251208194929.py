"""
Use trained model to predict sentiment on processed comments.
"""

import pandas as pd
import os
from pathlib import Path
import sys

# Ensure project root is on sys.path so `utils` imports work when running the file directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.file_helper import get_latest_file
from utils.config import PROCESSED_DIR
from model.model_utils import load_model
from utils.timer import timer
from utils.exception import handle_exception

log = get_logger(__name__)

@timer
def generate_predictions():
    try:
        log.info("🔮 Generating sentiment predictions...")

        # Load model - could be traditional ML or LSTM
        model_components = load_model("model/sentiment_model.pkl")
        cleaned_path = get_latest_file(PROCESSED_DIR)
        df = pd.read_csv(cleaned_path)

        # Check if this is an LSTM model (returns 4 components) or traditional ML (returns 2)
        if len(model_components) == 4:
            # LSTM model: model, tokenizer, label_encoder, max_len
            model, tokenizer, label_encoder, max_len = model_components

            # Preprocess text for LSTM
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
            X = pad_sequences(sequences, maxlen=max_len)

            # Make predictions
            predictions = model.predict(X)
            predicted_classes = predictions.argmax(axis=1)
            predicted_labels = label_encoder.inverse_transform(predicted_classes)

            df["predicted_sentiment"] = predicted_labels

        else:
            # Traditional ML model: model, vectorizer
            model, vectorizer = model_components
            X = vectorizer.transform(df["cleaned_text"])
            df["predicted_sentiment"] = model.predict(X)

        out_path = os.path.join(
            PROCESSED_DIR,
            os.path.basename(cleaned_path).replace(".csv", "_sentiment.csv")
        )

        df.to_csv(out_path, index=False)
        log.info(f"📈 Sentiment results saved at: {out_path}")

        return out_path

    except Exception as e:
        handle_exception(e, "model_prediction")
        raise
