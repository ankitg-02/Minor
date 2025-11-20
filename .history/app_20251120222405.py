import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
from datetime import datetime

# Set up project root for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.text_cleaning import clean_text
from model.train_model import rule_based_sentiment
from model.model_utils import load_model
from utils.file_helper import get_latest_file
from utils.config import PROCESSED_DIR, RAW_DIR

# Page configuration
st.set_page_config(page_title="YouTube Comments Analysis Dashboard", page_icon="📊", layout="wide")

# Sidebar for data exploration
st.sidebar.title("📂 Data Explorer")

# Raw Data Section
st.sidebar.header("Raw Data (API Fetched)")
try:
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    raw_files.sort(key=lambda x: os.path.getmtime(os.path.join(RAW_DIR, x)), reverse=True)
    if raw_files:
        selected_raw = st.sidebar.selectbox("Select Raw Data File", raw_files)
        if st.sidebar.button("View Raw Data"):
            raw_path = os.path.join(RAW_DIR, selected_raw)
            raw_df = pd.read_csv(raw_path)
            st.sidebar.write(f"**Latest Raw Data:** {selected_raw}")
            st.sidebar.write(f"**Timestamp:** {datetime.fromtimestamp(os.path.getmtime(raw_path)).strftime('%Y-%m-%d %H:%M:%S')}")
            st.sidebar.dataframe(raw_df.head(10))
    else:
        st.sidebar.write("No raw data files found.")
except Exception as e:
    st.sidebar.error(f"Error loading raw data: {str(e)}")

# Processed Data Section
st.sidebar.header("Processed Data")
try:
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv')]
    processed_files.sort(key=lambda x: os.path.getmtime(os.path.join(PROCESSED_DIR, x)), reverse=True)
    if processed_files:
        selected_processed = st.sidebar.selectbox("Select Processed Data File", processed_files, key="processed_select")
        if st.sidebar.button("View Processed Data", key="view_processed"):
            processed_path = os.path.join(PROCESSED_DIR, selected_processed)
            processed_df = pd.read_csv(processed_path)
            st.sidebar.write(f"**Latest Processed Data:** {selected_processed}")
            st.sidebar.write(f"**Timestamp:** {datetime.fromtimestamp(os.path.getmtime(processed_path)).strftime('%Y-%m-%d %H:%M:%S')}")
            st.sidebar.dataframe(processed_df.head(10))
            # Show demographics
            st.sidebar.subheader("Demographics")
            total_comments = len(processed_df)
            st.sidebar.metric("Total Comments", total_comments)
            sentiment_counts = processed_df['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total_comments) * 100
                st.sidebar.write(f"{sentiment}: {count} ({percentage:.1f}%)")
    else:
        st.sidebar.write("No processed data files found.")
except Exception as e:
    st.sidebar.error(f"Error loading processed data: {str(e)}")

# Main content
st.title("📊 YouTube Comments Analysis Dashboard")

# Text Cleaning Demo Section
st.header("🔧 Text Cleaning & Sentiment Prediction")
st.write("Enter a comment to see how the text cleaning process works and get a sentiment prediction (good/bad/neutral).")

user_input = st.text_area("Enter raw comment text:", height=100)

if st.button("Clean Text & Predict Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)

        # Try to use trained ML model, fallback to rule-based
        try:
            model, vectorizer = load_model("model/sentiment_model.pkl")
            X = vectorizer.transform([cleaned])
            sentiment = model.predict(X)[0]
            method = "ML Model"
        except FileNotFoundError:
            sentiment = rule_based_sentiment(cleaned)
            method = "Rule-based"

        st.subheader("Cleaning Results:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Text:**")
            st.code(user_input, language="")
        with col2:
            st.write("**Cleaned Text:**")
            st.code(cleaned, language="")

        st.subheader(f"Sentiment Prediction ({method}):")
        if sentiment == "good":
            st.success(f"😊 The comment is **{sentiment}**")
        elif sentiment == "bad":
            st.error(f"😞 The comment is **{sentiment}**")
        elif sentiment == "neutral":
            st.info(f"😐 The comment is **{sentiment}**")
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

    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.multiselect("Filter by Sentiment", options=["good", "bad", "neutral"], default=["good", "bad", "neutral"])
    with col2:
        video_filter = st.multiselect("Filter by Video ID", options=df['video_id'].unique(), default=df['video_id'].unique()[:5] if len(df['video_id'].unique()) > 5 else df['video_id'].unique())

    # Apply filters
    filtered_df = df[df['sentiment'].isin(sentiment_filter) & df['video_id'].isin(video_filter)]

    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_comments = len(filtered_df)
        st.metric("Total Comments", total_comments)
    with col2:
        unique_videos = filtered_df['video_id'].nunique()
        st.metric("Unique Videos", unique_videos)
    with col3:
        total_likes = filtered_df['like_count'].sum()
        st.metric("Total Likes", total_likes)

    # Sentiment Distribution
    st.subheader("User Satisfaction (Sentiment Distribution)")
    sentiment_counts = filtered_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Sentiment percentages
    st.write("Sentiment Breakdown:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_comments) * 100 if total_comments > 0 else 0
        st.write(f"- {sentiment}: {count} ({percentage:.1f}%)")

    # Video Breakdown
    st.subheader("Video Breakdown")
    video_stats = filtered_df.groupby('video_id').agg(
        comments=('text', 'count'),
        total_likes=('like_count', 'sum'),
        avg_likes=('like_count', 'mean'),
        good_comments=('sentiment', lambda x: (x == 'good').sum()),
        bad_comments=('sentiment', lambda x: (x == 'bad').sum()),
        neutral_comments=('sentiment', lambda x: (x == 'neutral').sum())
    ).reset_index()

    st.dataframe(video_stats.style.format({
        'avg_likes': '{:.2f}',
        'good_comments': '{:.0f}',
        'bad_comments': '{:.0f}',
        'neutral_comments': '{:.0f}'
    }))

    # Interactive Table
    st.subheader("Detailed Comments View")
    st.dataframe(filtered_df[['text', 'cleaned_text', 'sentiment', 'like_count', 'video_id']].head(50))

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure you have processed comment data in the data/processed/ directory.")
