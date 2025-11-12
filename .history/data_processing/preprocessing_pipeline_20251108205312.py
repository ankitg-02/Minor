"""
preprocessing_pipeline.py
----------------------------------
Main preprocessing pipeline for YouTube comments.
"""

import pandas as pd
import os
from tqdm import tqdm
from data_processing.text_cleaning import clean_text

tqdm.pandas()  # Enable progress bar for pandas apply


def load_raw_data(raw_path: str):
    """Loads raw comments data from CSV."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"❌ File not found: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"📥 Loaded {len(df)} comments from {raw_path}")
    return df


def preprocess_data(df: pd.DataFrame):
    """Performs text cleaning and deduplication."""
    print("🧹 Starting preprocessing...")

    # Drop duplicates and nulls
    df = df.drop_duplicates(subset=["text"])
    df = df.dropna(subset=["text"])

    # Clean comments
    df["cleaned_text"] = df["text"].progress_apply(clean_text)

    # Drop empty cleaned text
    df = df[df["cleaned_text"].str.strip() != ""]
    print(f"✅ Preprocessing complete: {len(df)} valid comments")
    return df


def save_processed_data(df: pd.DataFrame, keyword: str):
    """Saves processed comments to data/processed folder."""
    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/cleaned_comments_{keyword}.csv"
    df.to_csv(out_path, index=False)
    print(f"💾 Cleaned data saved to: {out_path}")


def run_preprocessing_pipeline(raw_path: str, keyword: str = "YouTube"):
    """
    Full preprocessing pipeline:
      1. Load raw CSV
      2. Clean text
      3. Save processed CSV
    """
    df = load_raw_data(raw_path)
    df_clean = preprocess_data(df)
    save_processed_data(df_clean, keyword)
    print("🚀 Preprocessing pipeline executed successfully!")
    return df_clean


# ✅ Run manually for testing
if __name__ == "__main__":
    latest_file = "data/raw/comments_Instagram update_20251030_193045.csv"
    run_preprocessing_pipeline(latest_file, keyword="Instagram update")
