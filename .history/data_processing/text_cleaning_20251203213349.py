"""
text_cleaning.py
----------------------------------
Handles text preprocessing for YouTube comments.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# Download required NLTK assets (only once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_emojis(text: str) -> str:
    """Removes emojis from text."""
    return emoji.replace_emoji(text, replace='')


def clean_text(text: str) -> str:
    """
    Cleans and preprocesses a single comment.
    Steps:
      1. Remove URLs
      2. Remove emojis
      3. Remove special characters & digits
      4. Lowercase conversion
      5. Remove stopwords
      6. Lemmatize words
    """
    if not isinstance(text, str):
        return ""

    # 1️⃣ Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # 2️⃣ Remove emojis
    text = remove_emojis(text)

    # 3️⃣ Remove special characters & numbers
    text = re.sub(r"[^A-Za-z\s]", " ", text)

    # 4️⃣ Lowercase
    text = text.lower()

    # 5️⃣ Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 6️⃣ Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)
