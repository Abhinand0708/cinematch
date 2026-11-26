/**
 * CineMatch — main.js
 * Phases 3-6: Autocomplete search, movie details, hybrid recommendations,
 * interactive recommendation cards.
 */

"use strict";

/* ═══════════════════════════════════════════════════════
   Constants & State
   ═══════════════════════════════════════════════════════ */

const API = {
  search: (q)  => `/api/search?query=${encodeURIComponent(q)}`,
  movie:  (id) => `/api/movie/${id}`,
  recs:   (id) => `/api/recommendations/${id}`,
};

const DEBOUNCE_MS = 350;
const MIN_QUERY   = 2;

let debounceTimer  = null;
let currentMovieId = null;

/* ═══════════════════════════════════════════════════════
   DOM References
   ═══════════════════════════════════════════════════════ */

const $ = (id) => document.getElementById(id);

const searchInput         = $("searchInput");
const searchClear         = $("searchClear");
const suggestionsDropdown = $("suggestionsDropdown");
const suggestionsList     = $("suggestionsList");
const suggestionsLoading  = $("suggestionsLoading");
const suggestionsEmpty    = $("suggestionsEmpty");

const movieDetailsSection = $("movieDetailsSection");
const detailsLoading      = $("detailsLoading");
const detailsError        = $("detailsError");
const movieCard           = $("movieCard");

const moviePoster         = $("moviePoster");
const moviePosterFallback = $("moviePosterFallback");
const movieTitle          = $("movieTitle");
const metaYear            = $("metaYear");
const metaRating          = $("metaRating");
const metaLang            = $("metaLang");
const movieGenres         = $("movieGenres");
const movieOverview       = $("movieOverview");
const movieDirector       = $("movieDirector");
const movieCast           = $("movieCast");
const directorRow         = $("directorRow");
const castRow             = $("castRow");

const recommendationsSection = $("recommendationsSection");
const recsLoading            = $("recsLoading");
const recsError              = $("recsError");
const recsGrid               = $("recsGrid");
const recsSubtitle           = $("recsSubtitle");
const welcomeSection         = $("welcomeSection");

/* ═══════════════════════════════════════════════════════
   Utility helpers
   ═══════════════════════════════════════════════════════ */

function debounce(fn, ms) {
  return (...args) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => fn(...args), ms);
  };
}

const show     = (el) => el.classList.remove("hidden");
const hide     = (el) => el.classList.add("hidden");
const active   = (el) => el.classList.add("active");
const inactive = (el) => el.classList.remove("active");

function yearFromDate(d) {
  if (!d || typeof d !== "string") return "";
  return d.slice(0, 4);
}

function safeText(val) {
  if (val == null) return "";
  const s = String(val).trim();
  return (s === "nan" || s === "None" || s === "NaN" || s === "undefined") ? "" : s;
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(text || ""));
  return d.innerHTML;
}

function scrollTo(el, offset = 80) {
  const top = el.getBoundingClientRect().top + window.scrollY - offset;
  window.scrollTo({ top, behavior: "smooth" });
}

/* ═══════════════════════════════════════════════════════
   Phase 3 — Autocomplete Search
   ═══════════════════════════════════════════════════════ */

function openDropdown()  { suggestionsDropdown.classList.add("open"); }
function closeDropdown() {
  suggestionsDropdown.classList.remove("open");
  inactive(suggestionsLoading);
  inactive(suggestionsEmpty);
}

function renderSuggestions(results) {
  suggestionsList.innerHTML = "";
  inactive(suggestionsLoading);

  if (!results || results.length === 0) {
    active(suggestionsEmpty);
    openDropdown();
    return;
  }
  inactive(suggestionsEmpty);

  results.forEach((movie) => {
    const item = document.createElement("div");
    item.className = "suggestion-item";
    item.setAttribute("role", "option");
    item.setAttribute("data-id", movie.id);

    const posterHtml = movie.poster_url
      ? `<img class="suggestion-poster" src="${escapeHtml(movie.poster_url)}" alt="${escapeHtml(movie.title)}" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex'" />
         <div class="suggestion-poster-fallback" style="display:none">🎬</div>`
      : `<div class="suggestion-poster-fallback">🎬</div>`;

    item.innerHTML = `
      ${posterHtml}
      <div class="suggestion-info">
        <div class="suggestion-title">${escapeHtml(movie.title)}</div>
        <div class="suggestion-year">${movie.release_year || "Unknown year"}</div>
      </div>
    `;
    item.addEventListener("click", () => selectMovie(movie.id, movie.title));
    suggestionsList.appendChild(item);
  });

  openDropdown();
}

async function fetchSuggestions(query) {
  if (query.length < MIN_QUERY) { closeDropdown(); return; }

  suggestionsList.innerHTML = "";
  inactive(suggestionsEmpty);
  active(suggestionsLoading);
  openDropdown();

  try {
    const res  = await fetch(API.search(query));
    const data = await res.json();
    renderSuggestions(data.results || []);
  } catch (err) {
    console.error("Search error:", err);
    inactive(suggestionsLoading);
    active(suggestionsEmpty);
  }
}

const debouncedSearch = debounce(fetchSuggestions, DEBOUNCE_MS);

searchInput.addEventListener("input", (e) => {
  const q = e.target.value.trim();
  searchClear.classList.toggle("visible", q.length > 0);
  if (q.length < MIN_QUERY) { closeDropdown(); return; }
  debouncedSearch(q);
});

searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDropdown();
});

searchClear.addEventListener("click", () => {
  searchInput.value = "";
  searchClear.classList.remove("visible");
  closeDropdown();
  searchInput.focus();
});

document.addEventListener("click", (e) => {
  if (!e.target.closest("#searchWrapper")) closeDropdown();
});

/* ═══════════════════════════════════════════════════════
   Phase 4 — Movie Details Display
   ═══════════════════════════════════════════════════════ */

async function loadMovieDetails(movieId) {
  show(movieDetailsSection);
  hide(welcomeSection);

  // Reset states
  hide(detailsError);
  hide(movieCard);

  // Show loading spinner
  show(detailsLoading);
  detailsLoading.style.display = "flex";

  try {
    const res = await fetch(API.movie(movieId));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderMovieDetails(await res.json());
  } catch (err) {
    console.error("Movie details error:", err);
    hide(detailsLoading);
    detailsLoading.style.display = "";
    show(detailsError);
    detailsError.style.display = "flex";
  }
}

function renderMovieDetails(data) {
  // Hide loading
  hide(detailsLoading);
  detailsLoading.style.display = "";

  // Poster
  const posterUrl = safeText(data.poster_url);
  if (posterUrl) {
    moviePoster.src = posterUrl;
    moviePoster.alt = safeText(data.title);
    moviePoster.style.display = "block";
    hide(moviePosterFallback);
    moviePoster.onerror = () => { hide(moviePoster); show(moviePosterFallback); };
  } else {
    hide(moviePoster);
    show(moviePosterFallback);
  }

  // Title
  const title = safeText(data.title) || "Unknown Title";
  movieTitle.textContent = title;
  document.title = `${title} — CineMatch`;

  // Meta badges
  const year = yearFromDate(safeText(data.release_date));
  metaYear.textContent = year;
  metaYear.style.display = year ? "" : "none";

  const rating = parseFloat(data.vote_average) || 0;
  metaRating.textContent = rating > 0 ? `⭐ ${rating.toFixed(1)}` : "";
  metaRating.style.display = rating > 0 ? "" : "none";

  const lang = safeText(data.original_language);
  metaLang.textContent = lang.toUpperCase();
  metaLang.style.display = lang ? "" : "none";

  // Genres
  movieGenres.innerHTML = "";
  const genresStr = safeText(data.genres);
  if (genresStr) {
    genresStr.split(",").forEach((g) => {
      const t = g.trim();
      if (!t) return;
      const pill = document.createElement("span");
      pill.className = "genre-pill";
      pill.textContent = t;
      movieGenres.appendChild(pill);
    });
  }

  // Overview
  const overview = safeText(data.overview);
  movieOverview.textContent = overview || "No overview available.";

  // Director
  const director = safeText(data.director);
  if (director) {
    movieDirector.textContent = director;
    show(directorRow);
  } else {
    hide(directorRow);
  }

  // Cast
  const cast = Array.isArray(data.cast) ? data.cast.filter(Boolean) : [];
  if (cast.length > 0) {
    movieCast.textContent = cast.join(", ");
    show(castRow);
  } else {
    hide(castRow);
  }

  show(movieCard);
}

/* ═══════════════════════════════════════════════════════
   Phase 5 — Hybrid Recommendations
   ═══════════════════════════════════════════════════════ */

async function loadRecommendations(movieId, titleText) {
  show(recommendationsSection);
  recsSubtitle.textContent = `Based on "${titleText}"`;

  hide(recsError);
  recsGrid.innerHTML = "";

  // Show loading spinner
  show(recsLoading);
  recsLoading.style.display = "flex";

  const loadingP = recsLoading.querySelector("p");
  if (loadingP) loadingP.textContent = "Generating AI recommendations…";

  try {
    const res = await fetch(API.recs(movieId));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    hide(recsLoading);
    recsLoading.style.display = "";
    renderRecommendations(data.recommendations || []);
  } catch (err) {
    console.error("Recommendations error:", err);
    hide(recsLoading);
    recsLoading.style.display = "";
    show(recsError);
    recsError.style.display = "flex";
  }
}

function renderRecommendations(recs) {
  recsGrid.innerHTML = "";

  if (!recs || recs.length === 0) {
    recsGrid.innerHTML = `<p style="color:var(--text-muted);padding:20px 0;">No recommendations found.</p>`;
    return;
  }

  recs.forEach((movie) => {
    const card = document.createElement("div");
    card.className = "rec-card";
    card.setAttribute("data-id", movie.id);
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `View recommendations for ${movie.title}`);

    const year        = yearFromDate(safeText(movie.release_date));
    const sourceClass = (movie.source || "Local").toLowerCase() === "local" ? "local" : "tmdb";
    const sourceLbl   = safeText(movie.source) || "Local";
    const title       = safeText(movie.title) || "Unknown";
    const posterUrl   = safeText(movie.poster_url);

    const posterHtml = posterUrl
      ? `<img class="rec-poster" src="${escapeHtml(posterUrl)}" alt="${escapeHtml(title)}" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex'" />
         <div class="rec-poster-fallback" style="display:none">🎬</div>`
      : `<div class="rec-poster-fallback">🎬</div>`;

    card.innerHTML = `
      <div class="rec-poster-wrap">
        ${posterHtml}
        <span class="rec-source-badge ${sourceClass}">${escapeHtml(sourceLbl)}</span>
        <div class="rec-overlay"><span class="rec-overlay-text">More like this</span></div>
      </div>
      <div class="rec-info">
        <div class="rec-title">${escapeHtml(title)}</div>
        <div class="rec-year">${escapeHtml(year)}</div>
      </div>
    `;

    card.addEventListener("click",   () => selectMovie(movie.id, title));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMovie(movie.id, title); }
    });

    recsGrid.appendChild(card);
  });
}

/* ═══════════════════════════════════════════════════════
   Phase 6 — Unified selectMovie
   ═══════════════════════════════════════════════════════ */

async function selectMovie(movieId, movieTitleText) {
  // Prevent double-clicks on the same movie
  if (movieId === currentMovieId) return;
  currentMovieId = movieId;

  // Update search bar
  searchInput.value = movieTitleText || "";
  searchClear.classList.add("visible");
  closeDropdown();

  // Show sections immediately so scroll works
  show(movieDetailsSection);
  show(recommendationsSection);
  hide(welcomeSection);

  // Scroll to movie details section
  scrollTo(movieDetailsSection, 70);

  // Dim existing rec cards while loading new ones
  recsGrid.querySelectorAll(".rec-card").forEach((c) => {
    c.style.opacity = "0.4";
    c.style.pointerEvents = "none";
  });

  // Fade current movie card
  movieCard.style.opacity = "0.3";

  // Load details and recommendations in parallel
  await Promise.all([
    loadMovieDetails(movieId),
    loadRecommendations(movieId, movieTitleText || "this movie"),
  ]);

  // Restore opacity
  movieCard.style.opacity = "1";

  // Allow same movie to be re-selected after load completes
  currentMovieId = movieId;
}

/* ═══════════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════════ */
show(welcomeSection);
console.log("CineMatch initialised.");
