import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Set up project root for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.text_cleaning import clean_text
from model.train_model import rule_based_sentiment
from utils.file_helper import get_latest_file
from utils.config import PROCESSED_DIR

st.title("Instagram Comments Analysis Dashboard")

# Text Cleaning Demo Section
st.header("🔧 Text Cleaning Demo")
st.write("Enter a comment to see how the text cleaning process works step-by-step.")

user_input = st.text_area("Enter raw comment text:", height=100)

if st.button("Clean Text"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        st.subheader("Cleaning Results:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Text:**")
            st.code(user_input, language="")
        with col2:
            st.write("**Cleaned Text:**")
            st.code(cleaned, language="")
    else:
        st.warning("Please enter some text to clean.")

# Data Insights Section
st.header("📊 Data Insights & User Satisfaction")

try:
    # Load the latest cleaned comments data
    cleaned_path = get_latest_file(PROCESSED_DIR)
    df = pd.read_csv(cleaned_path)

    # Compute sentiments using rule-based method
    df['sentiment'] = df['cleaned_text'].apply(rule_based_sentiment)

    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_comments = len(df)
        st.metric("Total Comments", total_comments)
    with col2:
        unique_videos = df['video_id'].nunique()
        st.metric("Unique Videos", unique_videos)
    with col3:
        total_likes = df['like_count'].sum()
        st.metric("Total Likes", total_likes)

    # Sentiment Distribution
    st.subheader("User Satisfaction (Sentiment Distribution)")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Sentiment percentages
    st.write("Sentiment Breakdown:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_comments) * 100
        st.write(f"- {sentiment}: {count} ({percentage:.1f}%)")

    # Video Breakdown
    st.subheader("Video Breakdown")
    video_stats = df.groupby('video_id').agg(
        comments=('text', 'count'),
        total_likes=('like_count', 'sum'),
        avg_likes=('like_count', 'mean'),
        positive_comments=('sentiment', lambda x: (x == 'Positive').sum()),
        negative_comments=('sentiment', lambda x: (x == 'Negative').sum()),
        neutral_comments=('sentiment', lambda x: (x == 'Neutral').sum())
    ).reset_index()

    st.dataframe(video_stats.style.format({
        'avg_likes': '{:.2f}',
        'positive_comments': '{:.0f}',
        'negative_comments': '{:.0f}',
        'neutral_comments': '{:.0f}'
    }))

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure you have processed comment data in the data/processed/ directory.")
