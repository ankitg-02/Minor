import sys
from pathlib import Path

# Set up project root for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data_processing.text_cleaning import clean_text
    print("✓ Text cleaning import successful")

    from model.train_model import rule_based_sentiment
    print("✓ Sentiment analysis import successful")

    from utils.file_helper import get_latest_file
    from utils.config import PROCESSED_DIR
    print("✓ Utils import successful")

    # Test text cleaning
    test_text = "Hello! This is a test comment with emojis 😀 and URLs https://example.com"
