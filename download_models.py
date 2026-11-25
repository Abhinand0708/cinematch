"""
download_models.py
------------------
Downloads model and dataset files from Hugging Face Hub into the
correct local folders.

Run this once after cloning the repo:
    python download_models.py

Requires:  pip install huggingface_hub
"""

import os
from huggingface_hub import hf_hub_download

# ── Configure these two values ────────────────────────────────────────────
HF_REPO_ID   = "abhinand8386/cinematch-models"   # e.g. "abhin/cinematch-models"
REPO_TYPE    = "dataset"
# ──────────────────────────────────────────────────────────────────────────

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

MODEL_FILES = [
    ("models/final_hybrid_config.pkl",         MODELS_DIR),
    ("models/hybrid_config.pkl",               MODELS_DIR),
    ("models/movie_embeddings.npy",            MODELS_DIR),
    ("models/tfidf_vectorizer.pkl",            MODELS_DIR),
    ("models/tfidf_matrix.npz",               MODELS_DIR),
    ("models/genre_sets.pkl",                  MODELS_DIR),
    ("dataset/recommendation_model_dataset.csv", DATASET_DIR),
    ("dataset/final_movie_dataset.csv",        DATASET_DIR),
]

print(f"Downloading files from: {HF_REPO_ID}")
print()

for filename, dest_dir in MODEL_FILES:
    local_name = os.path.basename(filename)
    dest_path  = os.path.join(dest_dir, local_name)

    if os.path.exists(dest_path):
        print(f"  ✓ Already exists: {local_name}")
        continue

    print(f"  ↓ Downloading: {local_name} ...", end=" ", flush=True)
    try:
        hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = filename,
            repo_type = REPO_TYPE,
            local_dir = os.path.dirname(__file__),
        )
        print("done")
    except Exception as e:
        print(f"FAILED — {e}")

print()
print("All files downloaded. You can now run:  cd backend && python app.py")
