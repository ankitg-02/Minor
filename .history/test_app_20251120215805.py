eimport sys
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
    cleaned = clean_text(test_text)
    print(f"✓ Text cleaning works: '{test_text}' -> '{cleaned}'")

    # Test sentiment
    sentiment = rule_based_sentiment(cleaned)
    print(f"✓ Sentiment analysis works: '{cleaned}' -> {sentiment}")

    # Test data loading
    cleaned_path = get_latest_file(PROCESSED_DIR)
    import pandas as pd
    df = pd.read_csv(cleaned_path)
    print(f"✓ Data loading works: Loaded {len(df)} comments from {cleaned_path}")

    # Test sentiment computation
    df['sentiment'] = df['cleaned_text'].apply(rule_based_sentiment)
    sentiment_counts = df['sentiment'].value_counts()
    print(f"✓ Sentiment computation works: {dict(sentiment_counts)}")

    print("\n🎉 All tests passed! The app should work correctly.")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
