"""
model_loader.py
---------------
Loads every ML artefact once when the Flask application starts.
All other modules import from this module — nothing reloads models per request.

Loaded artefacts
----------------
recommendation_df     : pandas DataFrame  — 19,995 rows, recommendation features
final_movie_df        : pandas DataFrame  — 19,999 rows, display info (poster, cast, director, etc.)
movie_embeddings      : np.ndarray        — shape (19995, 384), float32
tfidf_vectorizer      : TfidfVectorizer   — fitted, 30,000-feature vocabulary
tfidf_matrix          : csr_matrix        — shape (19995, 30000)
genre_sets            : list[set]         — 19,995 genre sets aligned with recommendation_df
config                : dict              — final_hybrid_config.pkl contents
sentence_model        : SentenceTransformer — all-MiniLM-L6-v2
local_id_to_index     : dict              — {tmdb_movie_id: row_index} for O(1) look-up
"""

import os
# Fix for Windows DLL conflicts (Intel OpenMP loaded by multiple libs)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz

# Import sentence_transformers AFTER setting env vars
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR   = os.path.join(_HERE, "..", "models")
_DATASET_DIR  = os.path.join(_HERE, "..", "dataset")


def _path(directory: str, filename: str) -> str:
    return os.path.join(directory, filename)


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

def load_all():
    """
    Load all model artefacts and return them as a dict.
    Call this once at Flask app startup.
    """
    print("=" * 60)
    print("Loading ML model artefacts…")
    print("=" * 60)

    # 1. final_hybrid_config (V2 — the active configuration)
    print("[1/8] Loading final_hybrid_config.pkl …")
    config = joblib.load(_path(_MODELS_DIR, "final_hybrid_config.pkl"))
    print(f"      Model: {config.get('model_name')} {config.get('model_version')}")
    print(f"      Weights — semantic: {config['semantic_weight']}, "
          f"tfidf: {config['tfidf_weight']}, genre: {config['genre_weight']}")

    # 2. Recommendation dataset (ML features)
    print("[2/8] Loading recommendation_model_dataset.csv …")
    recommendation_df = pd.read_csv(
        _path(_DATASET_DIR, "recommendation_model_dataset.csv")
    )
    print(f"      Shape: {recommendation_df.shape}")

    # 3. Final movie dataset (display info)
    print("[3/8] Loading final_movie_dataset.csv …")
    final_movie_df = pd.read_csv(
        _path(_DATASET_DIR, "final_movie_dataset.csv")
    )
    print(f"      Shape: {final_movie_df.shape}")

    # 4. Sentence Transformer embeddings
    print("[4/8] Loading movie_embeddings.npy …")
    movie_embeddings = np.load(_path(_MODELS_DIR, "movie_embeddings.npy"))
    print(f"      Shape: {movie_embeddings.shape}, dtype: {movie_embeddings.dtype}")

    # 5. TF-IDF vectorizer
    print("[5/8] Loading tfidf_vectorizer.pkl …")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress sklearn version warning
        tfidf_vectorizer = joblib.load(_path(_MODELS_DIR, "tfidf_vectorizer.pkl"))
    print(f"      Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    # 6. TF-IDF matrix
    print("[6/8] Loading tfidf_matrix.npz …")
    tfidf_matrix = load_npz(_path(_MODELS_DIR, "tfidf_matrix.npz"))
    print(f"      Shape: {tfidf_matrix.shape}")

    # 7. Genre sets
    print("[7/8] Loading genre_sets.pkl …")
    genre_sets = joblib.load(_path(_MODELS_DIR, "genre_sets.pkl"))
    print(f"      Count: {len(genre_sets)}")

    # 8. Sentence Transformer model
    print("[8/8] Loading Sentence Transformer (all-MiniLM-L6-v2) …")
    print("      This may take a moment on first run …")
    sentence_model = SentenceTransformer(config["sentence_transformer"])
    dim = sentence_model.get_embedding_dimension() if hasattr(sentence_model, "get_embedding_dimension") else sentence_model.get_sentence_embedding_dimension()
    print(f"      Embedding dim: {dim}")

    # ---------------------------------------------------------------------------
    # Build TMDB-ID → row-index lookup for O(1) local-dataset membership checks
    # ---------------------------------------------------------------------------
    local_id_to_index = {
        int(row["id"]): idx
        for idx, row in recommendation_df.iterrows()
    }

    # ---------------------------------------------------------------------------
    # Build TMDB-ID → row lookup for the display dataset
    # ---------------------------------------------------------------------------
    final_movie_id_to_row = {
        int(row["id"]): row
        for _, row in final_movie_df.iterrows()
    }

    # ---------------------------------------------------------------------------
    # Pre-build genre binary matrix for fast vectorised Jaccard similarity.
    # genre_matrix shape: (19995, num_unique_genres) — boolean numpy array.
    # This is computed ONCE at startup so genre scoring takes <5ms per request.
    # ---------------------------------------------------------------------------
    print("Building genre matrix for fast Jaccard scoring…")
    all_genre_labels = sorted({g for gs in genre_sets for g in gs})
    genre_label_index = {g: i for i, g in enumerate(all_genre_labels)}
    n_movies = len(genre_sets)
    n_genres = len(all_genre_labels)
    genre_matrix = np.zeros((n_movies, n_genres), dtype=np.float32)
    for row_i, gs in enumerate(genre_sets):
        for g in gs:
            if g in genre_label_index:
                genre_matrix[row_i, genre_label_index[g]] = 1.0
    print(f"      Genre matrix: {genre_matrix.shape}, unique genres: {n_genres}")

    # Pre-compute row sums (|A| for each movie) — needed for Jaccard denominator
    genre_row_sums = genre_matrix.sum(axis=1)   # shape (19995,)

    print("=" * 60)
    print("All artefacts loaded successfully.")
    print("=" * 60)

    return {
        "config":               config,
        "recommendation_df":    recommendation_df,
        "final_movie_df":       final_movie_df,
        "movie_embeddings":     movie_embeddings,
        "tfidf_vectorizer":     tfidf_vectorizer,
        "tfidf_matrix":         tfidf_matrix,
        "genre_sets":           genre_sets,
        "sentence_model":       sentence_model,
        "local_id_to_index":    local_id_to_index,
        "final_movie_id_to_row": final_movie_id_to_row,
        # Fast genre scoring structures
        "genre_matrix":         genre_matrix,       # (19995, n_genres) float32
        "genre_row_sums":       genre_row_sums,     # (19995,) float32
        "genre_label_index":    genre_label_index,  # {genre_name: col_index}
        "all_genre_labels":     all_genre_labels,   # list of genre names
    }


# ---------------------------------------------------------------------------
# Module-level singleton — imported by app.py and other modules
# ---------------------------------------------------------------------------

# This is populated by app.py calling initialise()
_MODELS = None


def initialise():
    """Call once from app.py before the first request."""
    global _MODELS
    if _MODELS is None:
        _MODELS = load_all()
    return _MODELS


def get():
    """Return the loaded models dict. Raises if initialise() was not called."""
    if _MODELS is None:
        raise RuntimeError(
            "model_loader.get() called before initialise(). "
            "Call model_loader.initialise() at app startup."
        )
    return _MODELS
