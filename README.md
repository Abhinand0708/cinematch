# 🎬 CineMatch — AI-Powered Movie Recommendation Website

A modern, cinematic movie recommendation website powered by a **Hybrid V2 Recommendation System** combining Sentence Transformers, TF-IDF and Genre Jaccard Similarity. Built on a custom dataset of **19,999 movies** collected via the TMDB API (1990–2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhinand0708/cinematch/blob/main/CineMatch_Deploy.ipynb)

> Click the badge above to launch the live app on Google Colab — no installation needed.

---

## 📸 Features

- 🔍 **Live autocomplete search** — searches TMDB as you type with poster previews
- 🎬 **Full movie details** — poster, director, cast, genres, rating, overview
- 🤖 **AI-powered recommendations** — 7 movies ranked by a hybrid similarity score
- 🔄 **Recursive exploration** — click any recommendation to get a fresh set
- 📊 **Dual candidate sources** — 19,995 local movies + live TMDB Similar & Recommendations APIs
- 📱 **Fully responsive** — works on mobile, tablet and desktop

---

## 🤖 Recommendation Model Architecture

The recommendation engine is a **Hybrid V2 System** that combines three similarity signals:

| Component | Model / Algorithm | Input Features | Weight |
|---|---|---|---|
| **Semantic Similarity** | Sentence Transformer (`all-MiniLM-L6-v2`) | Raw movie overview | **60%** |
| **TF-IDF Similarity** | Cosine similarity on TF-IDF vectors | Processed overview + genres + keywords | **20%** |
| **Genre Similarity** | Jaccard Index | Genre sets (multiword genres preserved) | **20%** |

### Hybrid Scoring Formula

```
hybrid_score = 0.60 × semantic_similarity
             + 0.20 × tfidf_similarity
             + 0.20 × genre_similarity
```

No min-max normalisation is applied. The selected movie is excluded from its own recommendations.

### Candidate Sources

For every selected movie, recommendations are collected from two sources:

1. **Local dataset** — all 19,995 movies scored using pre-saved embeddings and TF-IDF matrix
2. **TMDB API** — `/movie/{id}/similar` and `/movie/{id}/recommendations` endpoints

Both sources compete using the same hybrid formula. The top 7 by score are returned.

### Performance Optimisations

- Genre Jaccard computed via **vectorised NumPy matrix multiply** (not a Python loop) — ~100× faster
- External TMDB candidates **batch-encoded** in a single `sentence_model.encode()` call
- Genre **pre-filter** skips external candidates with zero genre overlap before fetching keywords
- Keyword lookups are **LRU-cached** per movie ID across requests
- All ML artefacts loaded **once at startup** — never reloaded per request

---

## 🗂️ Dataset

- **19,999 unique movies** collected from TMDB API (1990–2026)
- Features: title, overview, genres, cast, director, keywords, poster path, release date, ratings
- Preprocessing: lowercasing, contraction expansion, stopword removal (negation words preserved), POS-tagged lemmatisation
- Stored on Hugging Face Hub: [`abhinand8386/cinematch-models`](https://huggingface.co/datasets/abhinand8386/cinematch-models)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, Flask, Flask-CORS |
| **ML / NLP** | Sentence Transformers, Scikit-learn (TF-IDF), NLTK, NumPy, SciPy, Pandas |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Data** | TMDB API, custom dataset |
| **Deployment** | Google Colab + ngrok (free) |

---

## 📁 Project Structure

```
cinematch/
├── backend/
│   ├── app.py                  Flask entry point — 3 REST API routes
│   ├── model_loader.py         Loads all ML artefacts once at startup
│   ├── recommender.py          Hybrid V2 scoring pipeline (optimised)
│   ├── tmdb_client.py          All TMDB API calls with retry logic
│   └── preprocessing.py        NLP text preprocessing pipeline
├── frontend/
│   ├── templates/
│   │   └── index.html          Single-page UI
│   └── static/
│       ├── css/style.css       Cinematic dark theme
│       └── js/main.js          Autocomplete, details, recommendations, interactive cards
├── CineMatch_Deploy.ipynb      Google Colab deployment notebook
├── download_models.py          Downloads model files from HF Hub for local use
├── requirements.txt
└── README.md
```

> Model files (`models/`) and datasets (`dataset/`) are hosted on Hugging Face Hub —  
> not stored in this repo due to file size limits.

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/search?query=inception` | Autocomplete movie search via TMDB |
| GET | `/api/movie/<id>` | Full movie details (title, poster, cast, director, genres, rating) |
| GET | `/api/recommendations/<id>` | 7 hybrid recommendations with source (Local / TMDB) and score |

---

## ▶️ Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/Abhinand0708/cinematch.git
cd cinematch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model files from Hugging Face Hub
```bash
python download_models.py
```

### 4. Add your TMDB token
Create a `.env` file in the project root:
```
TMDB_ACCESS_TOKEN=your_tmdb_bearer_token_here
```
Get your free token at: https://www.themoviedb.org/settings/api

### 5. Run the app
```bash
# Windows
.\run.bat
```
Open **http://localhost:5000** in your browser.

---

## ☁️ Deployment (Google Colab + ngrok)

The app is deployed using **Google Colab** as a free cloud server and **ngrok** to expose it publicly — no paid hosting required.

### How it works

1. Google Colab provides a **free cloud runtime with 12GB RAM** — enough for the ML models
2. The Colab notebook clones this repo, downloads model files from HF Hub, and starts Flask
3. ngrok creates a **secure public tunnel** to the Flask server
4. The resulting `https://xxxx.ngrok-free.app` URL works for anyone worldwide

### To launch the live app

1. Click the **Open in Colab** badge at the top of this README
2. In **Cell 4**, paste your TMDB access token
3. In **Cell 6**, paste your ngrok authtoken (free at https://ngrok.com)
4. Click **Runtime → Run all**
5. Copy the public URL from the last cell and share it

> The session lasts up to 12 hours on the free Colab tier. Keep the tab open while sharing.

---

## 📊 Saved Model Files (Hugging Face Hub)

| File | Description |
|---|---|
| `movie_embeddings.npy` | Sentence Transformer embeddings — shape (19995, 384) |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer — 30,000 feature vocabulary |
| `tfidf_matrix.npz` | Sparse TF-IDF matrix — shape (19995, 30000) |
| `genre_sets.pkl` | Python sets of genres per movie — 19,995 entries |
| `final_hybrid_config.pkl` | V2 model configuration (weights, metadata) |
| `recommendation_model_dataset.csv` | Preprocessed features for 19,995 movies |
| `final_movie_dataset.csv` | Full display info for 19,999 movies |
