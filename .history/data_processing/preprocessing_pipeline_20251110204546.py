"""
Preprocess raw YouTube comments into cleaned format.
"""

from utils.logger import get_logger
from utils.config import RAW_DIR, PROCESSED_DIR
from utils.file_helper import get_latest_file
from utils.exceptions import DataProcessingError, handle_exception
from utils.timer import timer
from data_processing.text_cleaning import clean_text

import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas()

log = get_logger(__name__)

@timer
def run_preprocessing():
    """Run preprocessing: load latest raw CSV, clean text, save processed CSV."""
    try:
        log.info("🧹 Starting preprocessing pipeline...")
        raw_path = get_latest_file(RAW_DIR)
        log.info(f"📄 Latest raw file: {raw_path}")

        df = pd.read_csv(raw_path)
        if df.empty:
            raise DataProcessingError("Raw CSV is empty.")

        df = df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        df["cleaned_text"] = df["text"].progress_apply(clean_text)
        df = df[df["cleaned_text"].str.strip() != ""]

        out_path = os.path.join(PROCESSED_DIR, f"cleaned_{os.path.basename(raw_path)}")
        df.to_csv(out_path, index=False)
        log.info(f"✅ Cleaned data saved to {out_path}")
        return out_path

    except Exception as e:
        handle_exception(e, "data_processing")
        raise
