import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# Set up project root for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.text_cleaning import clean_text
from model.train_model import rule_based_sentiment
from model.model_utils import load_model
from utils.file_helper import get_latest_file
from utils.config import PROCESSED_DIR, RAW_DIR

# Enhanced Page configuration with theme
st.set_page_config(
    page_title="🎯 YouTube Comments Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': 'YouTube Comments Analysis Dashboard v2.0 - Optimized & Dynamic'
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Sidebar with better organization
st.sidebar.markdown('<div class="sidebar-header"><h2>🎯 Data Explorer</h2></div>', unsafe_allow_html=True)
st.sidebar.markdown("**Explore raw and processed YouTube comments data**")

# Add refresh button for dynamic updates
if st.sidebar.button("🔄 Refresh Data", key="refresh_data"):
    st.cache_data.clear()
    st.rerun()

# Add data summary in sidebar
try:
    latest_processed = get_latest_file(PROCESSED_DIR)
    if latest_processed:
        df_summary = pd.read_csv(latest_processed)
        st.sidebar.success(f"📊 {len(df_summary)} comments loaded")
        st.sidebar.info(f"📅 Last updated: {datetime.fromtimestamp(os.path.getmtime(latest_processed)).strftime('%Y-%m-%d %H:%M')}")
except:
    st.sidebar.warning("No processed data available")

@st.cache_data
def load_raw_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_processed_data(file_path):
    return pd.read_csv(file_path)

# Raw Data Section
st.sidebar.header("Raw Data (API Fetched)")
try:
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    raw_files.sort(key=lambda x: os.path.getmtime(os.path.join(RAW_DIR, x)), reverse=True)
    if raw_files:
        selected_raw = st.sidebar.selectbox("Select Raw Data File", raw_files, key="raw_select")
        if st.sidebar.button("View Raw Data", key="view_raw"):
            raw_path = os.path.join(RAW_DIR, selected_raw)
            raw_df = load_raw_data(raw_path)
            st.sidebar.write(f"**Latest Raw Data:** {selected_raw}")
            st.sidebar.write(f"**Timestamp:** {datetime.fromtimestamp(os.path.getmtime(raw_path)).strftime('%Y-%m-%d %H:%M:%S')}")
            st.sidebar.dataframe(raw_df.head(10))
    else:
        st.sidebar.info("No raw data files found. Please run data ingestion.")
except Exception as e:
    st.sidebar.error(f"Error loading raw data: {str(e)}")

# Processed Data Section
st.sidebar.header("Processed Data")

# Function to get all processed files recursively
def get_all_processed_files():
    all_files = []
    for root, dirs, files in os.walk(PROCESSED_DIR):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROCESSED_DIR)
                mtime = os.path.getmtime(full_path)
                all_files.append({
                    'path': full_path,
                    'rel_path': rel_path,
                    'name': file,
                    'mtime': mtime,
                    'timestamp': datetime.fromtimestamp(mtime)
                })
    return sorted(all_files, key=lambda x: x['mtime'], reverse=True)

try:
    all_processed_files = get_all_processed_files()

    if all_processed_files:
        # Show total files count
        st.sidebar.success(f"📊 Found {len(all_processed_files)} processed files")

        # Show recent files (last 24 hours)
        now = datetime.now()
        recent_files = [f for f in all_processed_files if (now - f['timestamp']).days < 1]
        if recent_files:
            st.sidebar.info(f"🆕 {len(recent_files)} files updated in last 24h")

        # File browser with timestamps
        st.sidebar.subheader("📁 File Browser")

        # Group files by date for better organization
        files_by_date = {}
        for file_info in all_processed_files:
            date_key = file_info['timestamp'].strftime('%Y-%m-%d')
            if date_key not in files_by_date:
                files_by_date[date_key] = []
            files_by_date[date_key].append(file_info)

        # Display files grouped by date
        for date in sorted(files_by_date.keys(), reverse=True):
            with st.sidebar.expander(f"📅 {date} ({len(files_by_date[date])} files)", expanded=(date == max(files_by_date.keys()))):
                for file_info in files_by_date[date]:
                    # Highlight recent files
                    is_recent = (now - file_info['timestamp']).days < 1
                    icon = "🆕" if is_recent else "📄"

                    if st.button(f"{icon} {file_info['name']}", key=f"view_{file_info['path']}"):
                        try:
                            processed_df = load_processed_data(file_info['path'])
                            st.sidebar.write(f"**File:** {file_info['rel_path']}")
                            st.sidebar.write(f"**Last Modified:** {file_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

                            # Show file stats
                            total_comments = len(processed_df)
                            st.sidebar.metric("Total Comments", total_comments)

                            if 'sentiment' in processed_df.columns:
                                sentiment_counts = processed_df['sentiment'].value_counts()
                                all_sentiments = {'good': 0, 'bad': 0, 'neutral': 0}
                                all_sentiments.update(sentiment_counts.to_dict())

                                st.sidebar.subheader("Sentiment Breakdown")
                                for sentiment, count in all_sentiments.items():
                                    percentage = (count / total_comments) * 100 if total_comments > 0 else 0
                                    emoji = {"good": "😊", "bad": "😞", "neutral": "😐"}[sentiment]
                                    st.sidebar.write(f"{emoji} {sentiment}: {count} ({percentage:.1f}%)")

                            # Show preview
                            st.sidebar.subheader("Preview (first 5 rows)")
                            st.sidebar.dataframe(processed_df.head(5), use_container_width=True)

                        except Exception as e:
                            st.sidebar.error(f"Error loading file: {str(e)}")

        # Summary statistics
        st.sidebar.subheader("📈 Summary Statistics")
        total_comments_all = 0
        total_files = len(all_processed_files)

        for file_info in all_processed_files:
            try:
                df_temp = pd.read_csv(file_info['path'])
                total_comments_all += len(df_temp)
            except:
                continue

        st.sidebar.metric("Total Files", total_files)
        st.sidebar.metric("Total Comments Across All Files", total_comments_all)

    else:
        st.sidebar.info("No processed data files found. Please run data processing pipeline.")
except Exception as e:
    st.sidebar.error(f"Error loading processed data: {str(e)}")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🔧 Text Analysis", "📊 Data Insights", "📈 Advanced Analytics"])

with tab1:
    st.markdown('<div class="main-header">🔧 Text Cleaning & Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown("**Test our AI-powered text cleaning and sentiment analysis pipeline**")

    # Enhanced text input section
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area(
            "📝 Enter raw comment text:",
            value=st.session_state.get('user_input', ''),
            height=120,
            placeholder="Type or paste a YouTube comment here...",
            help="Enter any text to see how our cleaning pipeline processes it"
        )

    with col2:
        st.markdown("### 🎯 Quick Examples")
        if st.button("💡 Try Positive Comment", key="positive_example"):
            st.session_state.user_input = "This video is amazing! Great content and very helpful. Love it! 👍"
            st.rerun()
        if st.button("😞 Try Negative Comment", key="negative_example"):
            st.session_state.user_input = "This is terrible content. Waste of time, very disappointing."
            st.rerun()
        if st.button("😐 Try Neutral Comment", key="neutral_example"):
            st.session_state.user_input = "This video shows some information about the topic."
            st.rerun()

    if st.button("🚀 Clean Text & Predict Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("🔄 Processing your text..."):
                time.sleep(0.5)  # Small delay for better UX
                cleaned = clean_text(user_input)

                # Try to use trained ML model, fallback to rule-based with caching
                @st.cache_resource
                def load_ml_model():
                    try:
                        return load_model("model/sentiment_model.pkl")
                    except FileNotFoundError:
                        return None

                ml_model = load_ml_model()
                if ml_model:
                    model, vectorizer = ml_model
                    X = vectorizer.transform([cleaned])
                    sentiment = model.predict(X)[0]
                    method = "🤖 ML Model"
                    confidence = max(model.predict_proba(X)[0]) if hasattr(model, 'predict_proba') else 0.8
                else:
                    sentiment = rule_based_sentiment(cleaned)
                    method = "📋 Rule-based"
                    confidence = 0.7

            # Enhanced results display
            st.success("✅ Analysis Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📝 Original Text")
                st.text_area("", value=user_input, height=100, disabled=True, key="original")

            with col2:
                st.markdown("### 🧹 Cleaned Text")
                st.text_area("", value=cleaned, height=100, disabled=True, key="cleaned")

            # Sentiment result with enhanced styling
            st.markdown("### 🎭 Sentiment Analysis Result")
            sentiment_colors = {"good": "#00ff00", "bad": "#ff0000", "neutral": "#ffff00"}
            sentiment_emojis = {"good": "😊", "bad": "😞", "neutral": "😐"}

            if sentiment == "good":
                st.success(f"{sentiment_emojis[sentiment]} **POSITIVE** - This comment shows approval and satisfaction!")
            elif sentiment == "bad":
                st.error(f"{sentiment_emojis[sentiment]} **NEGATIVE** - This comment expresses dissatisfaction or criticism.")
            elif sentiment == "neutral":
                st.info(f"{sentiment_emojis[sentiment]} **NEUTRAL** - This comment is objective and balanced.")

            st.markdown(f"**Method Used:** {method}")
            if 'confidence' in locals():
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.markdown("### 📊 Data Insights & Analytics")
    st.markdown("Explore comprehensive sentiment analysis with interactive filters and visualizations.")

    try:
        # Get all processed files for selection
        all_processed_files = get_all_processed_files()
        file_options = {f"{f['rel_path']} ({f['timestamp'].strftime('%Y-%m-%d %H:%M')})": f['path'] for f in all_processed_files}

        # File selector for analytics
        selected_file_display = st.selectbox(
            "📁 Select Processed Data File for Analysis",
            options=list(file_options.keys()),
            index=0,  # Default to latest file
            help="Choose which processed data file to analyze. Video IDs and metrics will update dynamically."
        )
        selected_file_path = file_options[selected_file_display]

        # Load and cache data based on selected file
        @st.cache_data
        def load_analytics_data(file_path):
            df = pd.read_csv(file_path)
            df['sentiment'] = df['cleaned_text'].apply(rule_based_sentiment)
            if 'published_at' in df.columns:
                df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            return df

        df = load_analytics_data(selected_file_path)

        # Advanced Filters with real-time updates
        st.markdown("#### 🎛️ Interactive Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            sentiment_filter = st.multiselect(
                "🎭 Sentiment Filter",
                options=["good", "bad", "neutral"],
                default=["good", "bad", "neutral"],
                help="Filter comments by sentiment"
            )

        with col2:
            video_options = df['video_id'].unique()
            video_filter = st.multiselect(
                "🎬 Video Filter",
                options=video_options,
                default=list(video_options),
                help="Select specific videos to analyze"
            )

        with col3:
            date_range = st.date_input(
                "📅 Date Range",
                value=(df['published_at'].min().date() if 'published_at' in df.columns and df['published_at'].notnull().any() else datetime.now().date() - timedelta(days=30),
                       df['published_at'].max().date() if 'published_at' in df.columns and df['published_at'].notnull().any() else datetime.now().date()),
                help="Filter comments by date range"
            )

        # Apply filters
        filtered_df = df[df['sentiment'].isin(sentiment_filter) & df['video_id'].isin(video_filter)]
        if 'published_at' in filtered_df.columns and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['published_at'].dt.date >= date_range[0]) &
                (filtered_df['published_at'].dt.date <= date_range[1])
            ]

        # Enhanced Key Metrics with custom styling
        st.markdown("#### 📈 Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📝 Total Comments", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎬 Unique Videos", filtered_df['video_id'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👍 Total Likes", int(filtered_df['like_count'].sum()))
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            avg_sentiment_score = (filtered_df['sentiment'].map({'good': 1, 'neutral': 0, 'bad': -1}).mean() + 1) / 2 * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("😊 Sentiment Score", f"{avg_sentiment_score:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Interactive Sentiment Distribution with Plotly
        st.markdown("#### 🎭 Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment'].value_counts()

        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'good': '#00ff00', 'bad': '#ff0000', 'neutral': '#ffff00'}
        )
        fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # Time Series Analysis with Plotly
        if 'published_at' in filtered_df.columns and filtered_df['published_at'].notnull().any():
            st.markdown("#### 📈 Temporal Analysis")

            # Comments over time
            daily_comments = filtered_df.set_index('published_at').resample('D').size().reset_index()
            daily_comments.columns = ['Date', 'Comments']

            fig_time = px.line(
                daily_comments,
                x='Date',
                y='Comments',
                title='Comments Volume Over Time',
                markers=True
            )
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Comments")
            st.plotly_chart(fig_time, use_container_width=True)

            # Sentiment trend over time
            sentiment_trend = filtered_df.groupby([pd.Grouper(key='published_at', freq='D'), 'sentiment']).size().unstack(fill_value=0).reset_index()
            sentiment_trend = pd.melt(sentiment_trend, id_vars=['published_at'], var_name='Sentiment', value_name='Count')

            fig_trend = px.area(
                sentiment_trend,
                x='published_at',
                y='Count',
                color='Sentiment',
                title='Sentiment Trends Over Time',
                color_discrete_map={'good': '#00ff00', 'bad': '#ff0000', 'neutral': '#ffff00'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Enhanced Video Analysis
        st.markdown("#### 🎬 Video Performance Analysis")
        video_stats = filtered_df.groupby('video_id').agg(
            comments=('text', 'count'),
            total_likes=('like_count', 'sum'),
            avg_likes=('like_count', 'mean'),
            good_comments=('sentiment', lambda x: (x == 'good').sum()),
            bad_comments=('sentiment', lambda x: (x == 'bad').sum()),
            neutral_comments=('sentiment', lambda x: (x == 'neutral').sum())
        ).reset_index()

        # Sort by total comments for better visualization
        video_stats = video_stats.sort_values('comments', ascending=False)

        # Interactive bar chart for video comparison
        fig_videos = px.bar(
            video_stats.head(10),
            x='video_id',
            y='comments',
            title='Top 10 Videos by Comment Count',
            color='total_likes',
            color_continuous_scale='viridis'
        )
        fig_videos.update_layout(xaxis_title="Video ID", yaxis_title="Comment Count")
        st.plotly_chart(fig_videos, use_container_width=True)

        # Detailed video statistics table
        st.markdown("**Detailed Video Statistics:**")
        st.dataframe(
            video_stats.style.format({
                'avg_likes': '{:.2f}',
                'total_likes': '{:,.0f}',
                'good_comments': '{:.0f}',
                'bad_comments': '{:.0f}',
                'neutral_comments': '{:.0f}'
            }).background_gradient(cmap='RdYlGn', subset=['good_comments', 'bad_comments']),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ Error loading analytics data: {str(e)}")
        st.info("💡 Make sure you have processed comment data in the data/processed/ directory.")

with tab3:
    st.markdown("### 📈 Advanced Analytics & Insights")
    st.markdown("Deep-dive into word frequency analysis and advanced metrics.")

    try:
        # Get all processed files for selection (reuse from tab2 if available)
        if 'all_processed_files' not in locals():
            all_processed_files = get_all_processed_files()
        file_options = {f"{f['rel_path']} ({f['timestamp'].strftime('%Y-%m-%d %H:%M')})": f['path'] for f in all_processed_files}

        # File selector for advanced analytics
        selected_file_display_advanced = st.selectbox(
            "📁 Select Processed Data File for Advanced Analysis",
            options=list(file_options.keys()),
            index=0,  # Default to latest file
            help="Choose which processed data file to analyze. Word frequencies and heatmaps will update dynamically.",
            key="advanced_file_select"
        )
        selected_file_path_advanced = file_options[selected_file_display_advanced]

        # Load data for advanced analytics
        df_advanced = pd.read_csv(selected_file_path_advanced)

        # Word Frequency Analysis
        st.markdown("#### 📊 Word Frequency Analysis")
        all_text = ' '.join(df_advanced['cleaned_text'].fillna(''))
        words = all_text.split()
        word_freq = Counter(words).most_common(20)

        if word_freq:
            words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

            fig_words = px.bar(
                words_df,
                x='Word',
                y='Frequency',
                title='Top 20 Most Frequent Words',
                color='Frequency',
                color_continuous_scale='blues'
            )
            fig_words.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_words, use_container_width=True)

        # Sentiment by Video Heatmap
        st.markdown("#### 🔥 Sentiment Heatmap by Video")
        sentiment_pivot = df_advanced.pivot_table(
            index='video_id',
            columns='sentiment',
            values='text',
            aggfunc='count',
            fill_value=0
        ).head(10)  # Top 10 videos

        if not sentiment_pivot.empty:
            fig_heatmap = px.imshow(
                sentiment_pivot,
                title='Sentiment Distribution by Video (Top 10)',
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Engagement Analysis
        st.markdown("#### 📈 Engagement Analysis")
        engagement_df = df_advanced.groupby('video_id').agg({
            'like_count': ['sum', 'mean', 'max'],
            'text': 'count'
        }).round(2)
        engagement_df.columns = ['Total Likes', 'Avg Likes', 'Max Likes', 'Comment Count']
        engagement_df = engagement_df.sort_values('Total Likes', ascending=False).head(10)

        st.dataframe(engagement_df.style.format({
            'Total Likes': '{:,.0f}',
            'Avg Likes': '{:.2f}',
            'Max Likes': '{:,.0f}',
            'Comment Count': '{:,.0f}'
        }), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error in advanced analytics: {str(e)}")
        st.info("💡 Ensure you have processed data available for analysis.")

with tab4:
    st.markdown("### 🤖 Model Performance Comparison")
    st.markdown("Compare the performance of different machine learning models for sentiment analysis.")

    try:
        # Load model comparison results
        import json
        import os

        comparison_file = "model/model_comparison.json"
        if os.path.exists(comparison_file):
            with open(comparison_file, "r") as f:
                comparison_data = json.load(f)

            # Display best model
            best_model = comparison_data["best_model"]
            best_metrics = comparison_data["best_metrics"]

            st.success(f"🏆 **Best Model Selected:** {best_model}")
            st.info(f"**Best F1 Score:** {best_metrics['f1_weighted']:.4f}")

            # Model comparison table
            st.markdown("#### 📊 Model Performance Metrics")

            models_data = []
            for model_name, metrics in comparison_data["models"].items():
                models_data.append({
                    "Model": model_name,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "F1 Macro": f"{metrics['f1_macro']:.4f}",
                    "F1 Weighted": f"{metrics['f1_weighted']:.4f}",
                    "Precision (Weighted)": f"{metrics['precision_weighted']:.4f}",
                    "Recall (Weighted)": f"{metrics['recall_weighted']:.4f}"
                })

            comparison_df = pd.DataFrame(models_data)
            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision (Weighted)', 'Recall (Weighted)']), use_container_width=True)

            # Training information
            st.markdown("#### 📋 Training Information")
            training_info = comparison_data["training_info"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Dataset Size", training_info["dataset_size"])
            with col2:
                st.metric("🎯 Training Size", training_info["train_size"])
            with col3:
                st.metric("🧪 Test Size", training_info["test_size"])
            with col4:
                st.metric("📈 Classes", len(training_info["sentiment_distribution"]))

            # Sentiment distribution
            st.markdown("#### 🎭 Sentiment Distribution")
            sentiment_dist = training_info["sentiment_distribution"]
            dist_df = pd.DataFrame(list(sentiment_dist.items()), columns=["Sentiment", "Count"])
            st.bar_chart(dist_df.set_index("Sentiment"))

            # Model details
            st.markdown("#### 🔍 Model Details")
            with st.expander("View detailed metrics for each model"):
                for model_name, metrics in comparison_data["models"].items():
                    st.markdown(f"**{model_name}**")
                    detailed_metrics = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision (Macro)", "Precision (Weighted)", "Recall (Macro)", "Recall (Weighted)", "F1 (Macro)", "F1 (Weighted)"],
                        "Value": [
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['precision_macro']:.4f}",
                            f"{metrics['precision_weighted']:.4f}",
                            f"{metrics['recall_macro']:.4f}",
                            f"{metrics['recall_weighted']:.4f}",
                            f"{metrics['f1_macro']:.4f}",
                            f"{metrics['f1_weighted']:.4f}"
                        ]
                    })
                    st.table(detailed_metrics)
                    st.markdown("---")

        else:
            st.warning("🤖 No model comparison data found.")
            st.info("💡 Run the model training to generate comparison results: `python -c 'from model.train_model import train_sentiment_model; train_sentiment_model()'`")

            # Button to trigger training
            if st.button("🚀 Train Models & Compare", type="primary"):
                with st.spinner("🔄 Training and comparing models... This may take a few minutes."):
                    try:
                        from model.train_model import train_sentiment_model
                        train_sentiment_model()
                        st.success("✅ Model training completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error loading model comparison: {str(e)}")
        st.info("💡 Ensure model comparison data is available.")
