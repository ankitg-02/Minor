"""
Cleans and normalizes YouTube comments.
"""

import re, nltk, emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"[^A-Za-z\s]", " ", text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMA.lemmatize(t) for t in tokens]
    return " ".join(tokens)
