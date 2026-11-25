# 🎬 CineMatch — AI-Powered Movie Recommendation Website

A modern, cinematic movie recommendation website powered by a **Hybrid V2 Recommendation System** combining Sentence Transformers, TF-IDF and Genre Jaccard Similarity.

> **To run the live demo** → open `CineMatch_Deploy.ipynb` in Google Colab and run all cells. You'll get a public URL in ~5 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhinand0708/cinematch/blob/main/CineMatch_Deploy.ipynb)

---

## Features

- 🔍 **Live autocomplete search** — searches TMDB as you type
- 🎬 **Movie details** — poster, director, cast, genres, rating, overview
- 🤖 **AI recommendations** — 7 movies ranked by hybrid score
- 🔄 **Recursive exploration** — click any recommendation to get new ones
- 📊 **Dual sources** — 19,995 local movies + live TMDB data
- 📱 **Responsive** — works on mobile, tablet and desktop

---

## Recommendation Model

| Component | Input | Weight |
|---|---|---|
| Sentence Transformer (`all-MiniLM-L6-v2`) | Movie overview | 60% |
| TF-IDF Cosine Similarity | Overview + Genres + Keywords | 20% |
| Genre Jaccard Similarity | Genre sets | 20% |

```
hybrid_score = 0.60 × semantic + 0.20 × tfidf + 0.20 × genre
```

---

## Project Structure

```
cinematch/
├── backend/
│   ├── app.py                  Flask entry point + API routes
│   ├── model_loader.py         Loads all ML artefacts once at startup
│   ├── recommender.py          Hybrid V2 scoring pipeline
│   ├── tmdb_client.py          TMDB API calls
│   └── preprocessing.py        NLP preprocessing
├── frontend/
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── CineMatch_Deploy.ipynb      Google Colab deployment notebook
├── download_models.py          Downloads model files from HF Hub
├── requirements.txt
└── README.md
```

> Model files (`models/`) and datasets (`dataset/`) are hosted on Hugging Face Hub at  
> [`abhinand8386/cinematch-models`](https://huggingface.co/datasets/abhinand8386/cinematch-models) — not stored in this repo.

---

## Quick Demo (Google Colab — Free)

The easiest way to run CineMatch publicly:

1. Click the **Open in Colab** badge above
2. Click **Runtime → Run all**
3. Wait ~5 minutes for models to load
4. Copy the `https://xxxx.ngrok-free.app` URL from the last cell
5. Share that URL — anyone worldwide can use the website

Keep the Colab tab open while sharing. The session lasts up to 12 hours on the free tier.

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Abhinand0708/cinematch.git
cd cinematch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model files
```bash
python download_models.py
```

### 4. Add your TMDB token
Create a `.env` file in the project root:
```
TMDB_ACCESS_TOKEN=your_tmdb_bearer_token_here
```
Get your token at: https://www.themoviedb.org/settings/api

### 5. Run
```bash
# Windows
.\run.bat
```

Open **http://localhost:5000** in your browser.

---

## Tech Stack

**Backend:** Python, Flask, Pandas, NumPy, Scikit-learn, Sentence Transformers, NLTK, SciPy

**Frontend:** HTML5, CSS3, Vanilla JavaScript

**Data:** TMDB API, custom 19,999-movie dataset (1990–2026)

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/search?query=inception` | Autocomplete search via TMDB |
| `GET /api/movie/<id>` | Full movie details |
| `GET /api/recommendations/<id>` | 7 hybrid recommendations |
