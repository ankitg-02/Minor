st.subheader(f"Sentiment Prediction ({method}):")
        if sentiment == "good":
            st.success(f"😊 The comment is **{sentiment}**")
        elif sentiment == "bad":
            st.error(f"😞 The comment is **{sentiment}**")
        elif sentiment == "neutral":
            st.info(f"😐 The comment is **{sentiment}**")
