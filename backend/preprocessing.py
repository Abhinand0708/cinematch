"""
preprocessing.py
-----------------
Text preprocessing pipeline — exact replica of the logic used in the
original Movie_Recommendation_Model.ipynb notebook.

Rules:
  - overview  : remove stopwords = True
  - genres    : remove stopwords = False
  - keywords  : remove stopwords = True
  - Negation words are PRESERVED even when stopwords are removed.
  - Contractions are expanded before tokenisation.
  - Tokens are POS-tagged and lemmatised with WordNetLemmatizer.

This module is imported by recommender.py to process external TMDB movies
at query time. It is never used to re-fit the TF-IDF vectorizer.
"""

import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# ---------------------------------------------------------------------------
# Download required NLTK data (runs only if not already downloaded)
# ---------------------------------------------------------------------------

def download_nltk_resources():
    """Download NLTK resources silently if missing."""
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
        ("corpora/wordnet",           "wordnet"),
        ("corpora/omw-1.4",           "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


download_nltk_resources()

# ---------------------------------------------------------------------------
# Initialise lemmatizer and stopword set
# ---------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()

_stop_words = set(stopwords.words("english"))

# Words that must NOT be removed — they flip the meaning of a sentence
_negative_words = {
    "no", "not", "never", "neither", "nor",
    "without", "nobody", "nothing", "nowhere",
    "cannot", "hardly", "barely", "scarcely",
}

_stop_words = _stop_words - _negative_words

# ---------------------------------------------------------------------------
# Contraction map (matches the notebook exactly)
# ---------------------------------------------------------------------------

_contractions = {
    "can't":    "cannot",
    "won't":    "will not",
    "don't":    "do not",
    "doesn't":  "does not",
    "didn't":   "did not",
    "isn't":    "is not",
    "aren't":   "are not",
    "wasn't":   "was not",
    "weren't":  "were not",
    "haven't":  "have not",
    "hasn't":   "has not",
    "hadn't":   "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
}


def _expand_contractions(text: str) -> str:
    for contraction, expanded in _contractions.items():
        text = re.sub(re.escape(contraction), expanded, text)
    return text


# ---------------------------------------------------------------------------
# POS tag → WordNet format converter
# ---------------------------------------------------------------------------

def _get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------

def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocess a single text string.

    Parameters
    ----------
    text : str
        Raw input text (overview, genres, or keywords).
    remove_stopwords : bool
        True  → remove stopwords (but keep negation words).
        False → keep all words (used for genres).

    Returns
    -------
    str
        Cleaned, lemmatised text ready for TF-IDF or semantic embedding.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Expand contractions
    text = _expand_contractions(text)

    # 3. Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenise
    tokens = word_tokenize(text)

    # 6. Optionally remove stopwords (negation words are preserved)
    if remove_stopwords:
        tokens = [w for w in tokens if w not in _stop_words]

    # 7. POS-tag and lemmatise
    tagged = pos_tag(tokens)
    lemmatised = [
        lemmatizer.lemmatize(word, pos=_get_wordnet_pos(tag))
        for word, tag in tagged
    ]

    return " ".join(lemmatised)


# ---------------------------------------------------------------------------
# Convenience: build the combined processed_recommendation_text string
# That is: processed_overview + " " + processed_genres + " " + processed_keywords
# This is what the TF-IDF vectorizer was trained on.
# ---------------------------------------------------------------------------

def build_processed_tags(overview: str, genres: str, keywords: str) -> str:
    """
    Build the processed recommendation text used as TF-IDF input.

    overview  → stopwords removed
    genres    → stopwords kept
    keywords  → stopwords removed
    """
    p_overview  = preprocess_text(overview,  remove_stopwords=True)
    p_genres    = preprocess_text(genres,    remove_stopwords=False)
    p_keywords  = preprocess_text(keywords,  remove_stopwords=True)

    combined = f"{p_overview} {p_genres} {p_keywords}"
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


# ---------------------------------------------------------------------------
# Genre set builder
# ---------------------------------------------------------------------------

def build_genre_set(genres: str) -> set:
    """
    Convert a genres string like 'Action, Science Fiction' into a Python set:
    {'Action', 'Science Fiction'}

    Multiword genres are preserved exactly as they appear.
    """
    if not isinstance(genres, str) or not genres.strip():
        return set()
    return {g.strip() for g in genres.split(",") if g.strip()}
