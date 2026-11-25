"""
tmdb_client.py
--------------
All TMDB API interactions.

Every function reads TMDB_ACCESS_TOKEN from the environment (.env file).
The token is never exposed to the frontend.

Endpoints used
--------------
  /search/movie                               → autocomplete search
  /movie/{id}                                 → movie details
  /movie/{id}/credits                         → cast and director
  /movie/{id}/keywords                        → keywords
  /movie/{id}/similar                         → similar movies
  /movie/{id}/recommendations                 → TMDB recommendations
"""

import os
import requests
import time
from functools import lru_cache

# Use Windows system certificate store to avoid SSL conflicts with torch
try:
    import truststore
    truststore.set_default_ssl_context()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL   = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"   # poster URL prefix

TIMEOUT = 15   # seconds per request
MAX_RETRIES = 3   # retry on transient connection errors
RETRY_DELAY = 1   # seconds between retries


def _get_headers() -> dict:
    """Build auth headers from environment. Never hardcode the token."""
    token = os.environ.get("TMDB_ACCESS_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "TMDB_ACCESS_TOKEN not found in environment. "
            "Add it to your .env file."
        )
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _poster_url(poster_path: str) -> str:
    """Convert a TMDB poster_path to a full image URL, or return empty string."""
    if poster_path and isinstance(poster_path, str):
        return f"{IMAGE_BASE}{poster_path}"
    return ""


def _request_with_retry(url: str, params: dict = None) -> dict:
    """
    Make a GET request with retry logic for transient connection errors.
    Returns the JSON response dict, or empty dict on failure.
    """
    headers = _get_headers()
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[tmdb_client] Request failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[tmdb_client] Request failed after {MAX_RETRIES} attempts: {e}")
                return {}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_movies(query: str, page: int = 1) -> list[dict]:
    """
    Search TMDB for movies matching `query`.

    Returns a list of dicts with keys:
        id, title, release_year, poster_url
    """
    if not query or not query.strip():
        return []

    data = _request_with_retry(f"{BASE_URL}/search/movie", params={
        "query": query.strip(),
        "include_adult": "false",
        "language": "en-US",
        "page": page,
    })
    results = data.get("results", [])

    suggestions = []
    for item in results:
        raw_date  = item.get("release_date") or ""
        year      = raw_date[:4] if len(raw_date) >= 4 else ""
        suggestions.append({
            "id":           item.get("id"),
            "title":        item.get("title", ""),
            "release_year": year,
            "poster_url":   _poster_url(item.get("poster_path", "")),
        })

    return suggestions


# ---------------------------------------------------------------------------
# Movie details
# ---------------------------------------------------------------------------

def get_movie_details(movie_id: int) -> dict:
    """
    Fetch full movie details including credits and keywords in one call.

    Returns a dict with keys:
        id, title, overview, genres, keywords,
        poster_url, release_date, vote_average, original_language,
        cast (list of up to 6 names), director (str)
    """
    data = _request_with_retry(f"{BASE_URL}/movie/{movie_id}", params={
        "append_to_response": "credits,keywords",
        "language": "en-US",
    })
    if not data:
        return {}

    # Genres → comma-separated string
    genres_str = ", ".join(
        g["name"] for g in data.get("genres", [])
    )

    # Keywords
    kw_list = data.get("keywords", {}).get("keywords", [])
    keywords_str = ", ".join(kw["name"] for kw in kw_list)

    # Credits
    credits  = data.get("credits", {})
    cast_raw = credits.get("cast", [])
    crew_raw = credits.get("crew", [])

    cast = [m["name"] for m in cast_raw[:6]]   # top 6 cast members for display

    director = ""
    for person in crew_raw:
        if person.get("job") == "Director":
            director = person["name"]
            break

    raw_date = data.get("release_date") or ""

    return {
        "id":                movie_id,
        "title":             data.get("title", ""),
        "overview":          data.get("overview", ""),
        "genres":            genres_str,
        "keywords":          keywords_str,
        "poster_url":        _poster_url(data.get("poster_path", "")),
        "release_date":      raw_date,
        "vote_average":      round(data.get("vote_average", 0), 1),
        "original_language": data.get("original_language", ""),
        "cast":              cast,
        "director":          director,
    }


# ---------------------------------------------------------------------------
# External recommendation candidates
# ---------------------------------------------------------------------------

def get_similar_movies(movie_id: int) -> list[dict]:
    """
    Call /movie/{id}/similar and return raw result items.
    Each item has: id, title, overview, genres (from genre_ids), poster_path, release_date.
    Note: genre_ids are integers — convert with get_genre_names() if needed,
    or fetch full details per movie.
    """
    return _fetch_candidate_page(f"{BASE_URL}/movie/{movie_id}/similar")


def get_tmdb_recommendations(movie_id: int) -> list[dict]:
    """
    Call /movie/{id}/recommendations and return raw result items.
    """
    return _fetch_candidate_page(f"{BASE_URL}/movie/{movie_id}/recommendations")


def _fetch_candidate_page(url: str) -> list[dict]:
    """Fetch page 1 of a TMDB endpoint that returns a results list."""
    data = _request_with_retry(url, params={"language": "en-US", "page": 1})
    return data.get("results", [])


def get_movie_keywords(movie_id: int) -> str:
    """
    Fetch keywords for a movie and return as comma-separated string.
    Used when enriching external candidates that lack keyword data.
    """
    data = _request_with_retry(f"{BASE_URL}/movie/{movie_id}/keywords")
    kw_list = data.get("keywords", [])
    return ", ".join(kw["name"] for kw in kw_list)


# ---------------------------------------------------------------------------
# Genre ID → name mapping (TMDB returns genre_ids in list endpoints)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_genre_map() -> dict:
    """
    Fetch the TMDB genre list once and cache it.
    Returns {genre_id: genre_name}.
    """
    data = _request_with_retry(
        f"{BASE_URL}/genre/movie/list", params={"language": "en-US"}
    )
    genres = data.get("genres", [])
    return {g["id"]: g["name"] for g in genres}


def genre_ids_to_string(genre_ids: list) -> str:
    """Convert a list of integer genre IDs to 'Action, Drama' style string."""
    genre_map = _get_genre_map()
    return ", ".join(
        genre_map[gid] for gid in genre_ids if gid in genre_map
    )
