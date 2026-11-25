"""
recommender.py  —  Hybrid V2 Recommendation Pipeline  (optimised)
------------------------------------------------------------------

Speed optimisations vs original:
  1. Genre Jaccard — vectorised NumPy instead of Python set loop (~100× faster)
  2. External TMDB candidates — batch-encoded in one sentence_model.encode() call
  3. Keyword fetching — LRU-cached + only fetched for candidates that pass a
     genre pre-filter (candidates with zero genre overlap are skipped early)
  4. Local candidate list built from pre-sorted numpy argsort, only top-N×4
     candidates are assembled into dicts (avoids building 19,995-item list)

Formula (unchanged):
    hybrid_score = 0.60 × semantic + 0.20 × tfidf + 0.20 × genre  (no normalisation)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from scipy.sparse import issparse

import model_loader
import tmdb_client
from preprocessing import build_processed_tags, build_genre_set

TOP_N      = 7       # recommendations to return
LOCAL_POOL = TOP_N * 6   # top local candidates to assemble into dicts (keeps list small)


# ---------------------------------------------------------------------------
# Keyword cache — avoids re-fetching the same movie's keywords
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _cached_keywords(movie_id: int) -> str:
    return tmdb_client.get_movie_keywords(movie_id)


# ---------------------------------------------------------------------------
# Vectorised genre Jaccard
# Uses the pre-built binary genre_matrix from model_loader.
# |A ∩ B| / |A ∪ B|  =  dot(q, M) / (|q| + row_sums - dot(q, M))
# ---------------------------------------------------------------------------

def _genre_scores_vectorised(query_genre_set: set, models: dict) -> np.ndarray:
    """
    Return Jaccard similarity between query_genre_set and every row of genre_matrix.
    Shape: (n_movies,)  — computed entirely in NumPy, no Python loop.
    """
    genre_matrix    = models["genre_matrix"]      # (n_movies, n_genres) float32
    genre_row_sums  = models["genre_row_sums"]    # (n_movies,) float32
    genre_label_idx = models["genre_label_index"] # {name: col}

    if not query_genre_set:
        return np.zeros(genre_matrix.shape[0], dtype=np.float32)

    # Build query vector
    q = np.zeros(genre_matrix.shape[1], dtype=np.float32)
    for g in query_genre_set:
        if g in genre_label_idx:
            q[genre_label_idx[g]] = 1.0

    q_sum = q.sum()
    if q_sum == 0:
        return np.zeros(genre_matrix.shape[0], dtype=np.float32)

    # Intersection sizes via dot product
    intersection = genre_matrix @ q          # shape (n_movies,)

    # Union = |q| + |M_i| - |intersection_i|
    union = q_sum + genre_row_sums - intersection
    union = np.maximum(union, 1e-9)          # avoid div-by-zero

    return (intersection / union).astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_recommendations(movie_id: int) -> list[dict]:
    """
    Generate 7 hybrid recommendations for the given TMDB movie_id.
    """
    models = model_loader.get()

    config            = models["config"]
    rec_df            = models["recommendation_df"]
    embeddings        = models["movie_embeddings"]
    tfidf_vectorizer  = models["tfidf_vectorizer"]
    tfidf_matrix      = models["tfidf_matrix"]
    sentence_model    = models["sentence_model"]
    local_id_to_index = models["local_id_to_index"]
    final_id_to_row   = models["final_movie_id_to_row"]

    semantic_w = config["semantic_weight"]   # 0.60
    tfidf_w    = config["tfidf_weight"]      # 0.20
    genre_w    = config["genre_weight"]      # 0.20

    # -----------------------------------------------------------------------
    # Step 1 — Resolve the query movie's features
    # -----------------------------------------------------------------------

    movie_index = local_id_to_index.get(int(movie_id))

    if movie_index is not None:
        query_row      = rec_df.iloc[movie_index]
        query_overview = str(query_row.get("overview", "") or "")
        query_genres   = str(query_row.get("genres",   "") or "")
        # Use saved embedding and TF-IDF row (no encoding needed)
        query_embedding = embeddings[movie_index].reshape(1, -1)
        query_tfidf_vec = tfidf_matrix[movie_index]
    else:
        tmdb_data      = tmdb_client.get_movie_details(int(movie_id))
        query_overview = tmdb_data.get("overview",  "") or ""
        query_genres   = tmdb_data.get("genres",    "") or ""
        query_keywords = tmdb_data.get("keywords",  "") or ""

        query_embedding = (
            sentence_model.encode([query_overview], convert_to_numpy=True)
            if query_overview.strip()
            else np.zeros((1, embeddings.shape[1]), dtype=np.float32)
        )
        processed_tags  = build_processed_tags(query_overview, query_genres, query_keywords)
        query_tfidf_vec = tfidf_vectorizer.transform([processed_tags])

    query_genre_set = build_genre_set(query_genres)

    # -----------------------------------------------------------------------
    # Step 2 — Score ALL 19,995 local movies (fully vectorised)
    # -----------------------------------------------------------------------

    # 2a. Semantic — one cosine call against the full embedding matrix
    sem_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # 2b. TF-IDF — sparse cosine, still one call
    tfidf_scores = cosine_similarity(query_tfidf_vec, tfidf_matrix).flatten()

    # 2c. Genre — vectorised Jaccard (no Python loop)
    genre_scores = _genre_scores_vectorised(query_genre_set, models)

    # 2d. Weighted hybrid
    hybrid_scores = (
        semantic_w * sem_scores +
        tfidf_w    * tfidf_scores +
        genre_w    * genre_scores
    )

    # Exclude the query movie itself
    if movie_index is not None:
        hybrid_scores[movie_index] = -1.0

    # -----------------------------------------------------------------------
    # Step 3 — Build local candidate list (only top LOCAL_POOL entries)
    #   Avoids constructing a 19,995-item dict list every request
    # -----------------------------------------------------------------------

    top_local_indices = np.argsort(hybrid_scores)[::-1][:LOCAL_POOL]

    local_candidates = []
    for idx in top_local_indices:
        idx = int(idx)
        row     = rec_df.iloc[idx]
        tmdb_id = int(row["id"])

        display      = final_id_to_row.get(tmdb_id)
        poster_path  = ""
        release_date = ""
        if display is not None:
            _pp = display.get("poster_path", "")
            poster_path  = str(_pp) if _pp and str(_pp) not in ("nan", "None", "") else ""
            _rd = display.get("release_date", "")
            release_date = str(_rd) if _rd and str(_rd) not in ("nan", "None", "") else ""

        local_candidates.append({
            "id":           tmdb_id,
            "title":        str(row.get("title", "")),
            "poster_url":   tmdb_client._poster_url(poster_path),
            "release_date": release_date,
            "genres":       str(row.get("genres", "") or ""),
            "hybrid_score": float(hybrid_scores[idx]),
            "source":       "Local",
        })

    # -----------------------------------------------------------------------
    # Step 4 — Collect external TMDB candidates
    # -----------------------------------------------------------------------

    similar_raw     = tmdb_client.get_similar_movies(int(movie_id))
    recommend_raw   = tmdb_client.get_tmdb_recommendations(int(movie_id))

    # Deduplicate by ID, skip query movie and movies already in local dataset
    seen = set()
    external_raw = []
    for item in (similar_raw + recommend_raw):
        cid = item.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if int(cid) == int(movie_id):
            continue
        if int(cid) in local_id_to_index:
            continue   # already scored locally
        external_raw.append(item)

    # -----------------------------------------------------------------------
    # Step 5 — Score external candidates (batch encode + genre pre-filter)
    # -----------------------------------------------------------------------

    external_candidates = []

    if external_raw:
        # 5a. Genre pre-filter — skip candidates with zero genre overlap
        #     This eliminates keyword fetching for irrelevant movies
        filtered_external = []
        for item in external_raw:
            genre_ids  = item.get("genre_ids", [])
            genres_str = tmdb_client.genre_ids_to_string(genre_ids) if genre_ids else ""
            cand_gs    = build_genre_set(genres_str)
            # Keep if there is ANY genre overlap OR query has no genres (can't filter)
            if not query_genre_set or (query_genre_set & cand_gs):
                item["_genres_str"] = genres_str
                item["_genre_set"]  = cand_gs
                filtered_external.append(item)

        # 5b. Batch encode all remaining external overviews in ONE call
        overviews = [item.get("overview", "") or "" for item in filtered_external]
        if any(o.strip() for o in overviews):
            batch_embeddings = sentence_model.encode(
                overviews,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )
        else:
            batch_embeddings = np.zeros(
                (len(filtered_external), embeddings.shape[1]), dtype=np.float32
            )

        # 5c. Score each filtered external candidate
        for i, item in enumerate(filtered_external):
            cid         = item["id"]
            genres_str  = item["_genres_str"]
            cand_gs     = item["_genre_set"]
            overview    = overviews[i]

            # Semantic
            cand_emb = batch_embeddings[i].reshape(1, -1)
            cand_sem = float(cosine_similarity(query_embedding, cand_emb).flatten()[0])

            # TF-IDF — only fetch keywords now (after genre pre-filter)
            keywords_str   = _cached_keywords(int(cid))
            processed_tags = build_processed_tags(overview, genres_str, keywords_str)
            cand_tfidf_vec = tfidf_vectorizer.transform([processed_tags])
            cand_tfidf     = float(cosine_similarity(query_tfidf_vec, cand_tfidf_vec).flatten()[0])

            # Genre Jaccard
            union      = query_genre_set | cand_gs
            cand_genre = len(query_genre_set & cand_gs) / len(union) if union else 0.0

            cand_hybrid = (
                semantic_w * cand_sem +
                tfidf_w    * cand_tfidf +
                genre_w    * cand_genre
            )

            raw_date = item.get("release_date") or ""
            external_candidates.append({
                "id":           int(cid),
                "title":        item.get("title", ""),
                "poster_url":   tmdb_client._poster_url(item.get("poster_path", "") or ""),
                "release_date": raw_date,
                "genres":       genres_str,
                "hybrid_score": cand_hybrid,
                "source":       "TMDB",
            })

    # -----------------------------------------------------------------------
    # Step 6 — Combine, deduplicate, sort, return top 7
    # -----------------------------------------------------------------------

    deduped = {}
    for c in (local_candidates + external_candidates):
        cid = c["id"]
        if cid not in deduped or c["hybrid_score"] > deduped[cid]["hybrid_score"]:
            deduped[cid] = c

    ranked = sorted(deduped.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:TOP_N]
