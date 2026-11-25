"""
app.py
------
Flask entry point for the AI-Powered Hybrid Movie Recommendation Website.

API Endpoints
-------------
  GET /                                   → serves index.html
  GET /api/search?query=<text>            → TMDB movie autocomplete
  GET /api/movie/<movie_id>               → full movie details
  GET /api/recommendations/<movie_id>     → 7 hybrid recommendations

Usage
-----
    cd backend
    python app.py
Then open http://localhost:5000 in your browser.
"""

import os
# Fix for Windows DLL conflicts (Intel OpenMP loaded by numpy/scipy AND torch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

# Use Windows system certificate store to avoid SSL conflicts with torch
try:
    import truststore
    truststore.set_default_ssl_context()
except Exception:
    pass  # truststore is optional, fall back to default SSL

# ---------------------------------------------------------------------------
# Load .env before any other import that reads environment variables
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

# .env is one level up from backend/
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

from flask import Flask, jsonify, request, render_template, abort
from flask_cors import CORS

import model_loader
import tmdb_client
import recommender

# Simple in-memory cache for TMDB movie details (avoids repeated API calls)
_movie_details_cache: dict = {}

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

# templates/ and static/ live inside frontend/, one level up from backend/
_FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(_FRONTEND, "templates"),
    static_folder=os.path.join(_FRONTEND, "static"),
)

CORS(app)   # allow requests from any origin during development


# ---------------------------------------------------------------------------
# Load ML models once at startup (before the first request is served)
# ---------------------------------------------------------------------------

with app.app_context():
    model_loader.initialise()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return render_template("index.html")


# ── Search ──────────────────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    """
    GET /api/search?query=inception

    Returns up to 20 movie suggestions from TMDB.

    Response:
        {
          "results": [
            {"id": 27205, "title": "Inception", "release_year": "2010", "poster_url": "https://..."},
            ...
          ]
        }
    """
    query = request.args.get("query", "").strip()

    if not query:
        return jsonify({"results": [], "error": "query parameter is required"}), 400

    results = tmdb_client.search_movies(query)
    return jsonify({"results": results})


# ── Movie details ────────────────────────────────────────────────────────────

@app.route("/api/movie/<int:movie_id>")
def api_movie(movie_id: int):
    """
    GET /api/movie/27205

    Returns full movie details for display (title, poster, cast, director, etc.).
    First checks the local display dataset, falls back to TMDB API.

    Response:
        {
          "id": 27205,
          "title": "Inception",
          "overview": "...",
          "genres": "Action, Science Fiction, Adventure",
          "poster_url": "https://...",
          "release_date": "2010-07-15",
          "vote_average": 8.4,
          "original_language": "en",
          "cast": ["Leonardo DiCaprio", ...],
          "director": "Christopher Nolan",
          "source": "local" | "tmdb"
        }
    """
    models      = model_loader.get()
    id_to_row   = models["final_movie_id_to_row"]

    # Try local display dataset first (faster, no API call)
    if movie_id in id_to_row:
        row = id_to_row[movie_id]

        def safe_str(val, default=""):
            """Convert pandas value to string, treating NaN/None as empty."""
            import math
            if val is None:
                return default
            try:
                if math.isnan(float(val)):
                    return default
            except (TypeError, ValueError):
                pass
            s = str(val).strip()
            return s if s not in ("nan", "None", "NaN") else default

        cast_raw = safe_str(row.get("cast", ""))
        cast     = [c.strip() for c in cast_raw.split(",") if c.strip()][:6]

        poster_path = safe_str(row.get("poster_path", ""))
        raw_date    = safe_str(row.get("release_date", ""))

        try:
            vote = round(float(row.get("vote_average", 0) or 0), 1)
            if vote != vote:   # NaN check
                vote = 0.0
        except (TypeError, ValueError):
            vote = 0.0

        result = {
            "id":                movie_id,
            "title":             safe_str(row.get("title", ""), "Unknown"),
            "overview":          safe_str(row.get("overview", "")),
            "genres":            safe_str(row.get("genres", "")),
            "keywords":          safe_str(row.get("keywords", "")),
            "poster_url":        tmdb_client._poster_url(poster_path),
            "release_date":      raw_date,
            "vote_average":      vote,
            "original_language": safe_str(row.get("original_language", "")),
            "cast":              cast,
            "director":          safe_str(row.get("director", "")),
            "source":            "local",
        }
        return jsonify(result)

    # Not in local dataset — fetch from TMDB (with simple cache)
    if movie_id in _movie_details_cache:
        return jsonify(_movie_details_cache[movie_id])

    details = tmdb_client.get_movie_details(movie_id)
    if not details:
        return jsonify({"error": f"Movie {movie_id} not found"}), 404

    details["source"] = "tmdb"
    _movie_details_cache[movie_id] = details
    return jsonify(details)


# ── Recommendations ──────────────────────────────────────────────────────────

@app.route("/api/recommendations/<int:movie_id>")
def api_recommendations(movie_id: int):
    """
    GET /api/recommendations/27205

    Runs the full Hybrid V2 pipeline and returns 7 recommendations.

    Response:
        {
          "movie_id": 27205,
          "recommendations": [
            {
              "id": 157336,
              "title": "Interstellar",
              "poster_url": "https://...",
              "release_date": "2014-11-05",
              "genres": "Adventure, Drama, Science Fiction",
              "hybrid_score": 0.6234,
              "source": "Local"
            },
            ...
          ]
        }
    """
    try:
        recs = recommender.get_recommendations(movie_id)
    except Exception as e:
        print(f"[app] recommendations error for movie {movie_id}: {e}")
        return jsonify({"error": "Failed to generate recommendations", "detail": str(e)}), 500

    return jsonify({
        "movie_id":        movie_id,
        "recommendations": recs,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5000)))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
