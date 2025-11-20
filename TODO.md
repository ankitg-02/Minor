# TODO: Update User Satisfaction Categories to "good", "neutral", "bad"

## Tasks
- [x] Update `model/train_model.py`: Change `rule_based_sentiment` function to return "good", "bad", "neutral" instead of "Positive", "Negative", "Neutral".
- [x] Update `app.py`: Change sentiment prediction display conditions to check for "good", "bad", "neutral".
- [x] Update `app.py`: Modify video stats aggregation to count 'good', 'bad', 'neutral' comments.
- [x] Update `app.py`: Adjust column names in video_stats dataframe and formatting.
- [x] Retrain the sentiment model to use the new labels.
