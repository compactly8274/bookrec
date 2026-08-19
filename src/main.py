from pathlib import Path
import sqlite3
import json
import logging
import os
import asyncio
import random
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bookrec")

CONFIG_DIR = Path("/config")
DB_PATH = CONFIG_DIR / "bookrec.db"
CALIBRE_DB = Path("/calibre/metadata.db")
INDEX_PATH = CONFIG_DIR / "index.faiss"
INDEX_META_PATH = CONFIG_DIR / "index_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Bump whenever embed_text() or MODEL_NAME changes, to force an index rebuild.
EMBED_VERSION = 3

OLLAMA_URL = ""
OLLAMA_MODEL = "gemma3:4b"

# Positive-signal weights: a "like" is a strong signal, "to read" is a soft one.
LIKE_WEIGHT = 1.0
TOREAD_WEIGHT = 0.5
READ_WEIGHT = 1.0

# Dislikes actively push candidates away (not just exclude them).
DISLIKE_PENALTY = 0.5

# Books marked "seen" become eligible again after this many days.
SEEN_TTL_DAYS = 30

# Fraction of each batch that is random exploration (serendipity) rather than
# nearest-neighbour exploitation. Set to 0.0 to disable.
EXPLORATION_RATE = 0.15

# Embedding field weighting. all-MiniLM-L6-v2 has a ~256-token input limit, so
# the description must be truncated to leave room for the more discriminative
# title/tags. Repetition approximates per-field weighting (the model has no
# native weighting); title/tags are repeated, description is not.
DESCRIPTION_MAX_WORDS = 120
TITLE_REPEAT = 2
TAGS_REPEAT = 2
AUTHOR_REPEAT = 1
DESCRIPTION_REPEAT = 1

# Relevance boost for a book that is later in a series the user has liked an
# earlier entry of. Additive to the relevance score (typically 0–1).
SERIES_BOOST = 0.15


class Feedback(BaseModel):
    book_id: int
    action: str = Field(..., pattern="^(like|dislike|skip|toread|read)$")
    reason_shown: str = ""


# ── Calibre reading ──────────────────────────────────────────────────────────

def get_books():
    conn = sqlite3.connect(str(CALIBRE_DB))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT b.id, b.title, b.author_sort, b.path, b.has_cover,
               c.text AS description,
               r.rating AS rating_val
        FROM books b
        LEFT JOIN comments c ON c.book = b.id
        LEFT JOIN books_ratings_link brl ON brl.book = b.id
        LEFT JOIN ratings r ON r.id = brl.rating
        """
    )
    books = []
    for row in cur.fetchall():
        books.append(dict(row))
    conn.close()

    # Authors, tags, and series
    conn = sqlite3.connect(str(CALIBRE_DB))
    cur = conn.cursor()
    cur.execute("SELECT bal.book, a.name FROM books_authors_link bal JOIN authors a ON a.id=bal.author")
    authors: dict = {}
    for bid, name in cur.fetchall():
        authors.setdefault(bid, []).append(name)
    cur.execute("SELECT btl.book, t.name FROM books_tags_link btl JOIN tags t ON t.id=btl.tag")
    tags: dict = {}
    for bid, name in cur.fetchall():
        tags.setdefault(bid, []).append(name)
    cur.execute(
        "SELECT bsl.book, s.name, bsl.series_index "
        "FROM books_series_link bsl JOIN series s ON s.id=bsl.series"
    )
    series: dict = {}
    for bid, sname, sidx in cur.fetchall():
        series[bid] = {"name": sname, "index": sidx}
    conn.close()

    for b in books:
        b["authors"] = authors.get(b["id"], [])
        b["tags"] = tags.get(b["id"], [])
        s = series.get(b["id"])
        b["series"] = s["name"] if s else None
        b["series_index"] = s["index"] if s else None
    return books


def _truncate_words(text, max_words):
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def embed_text(book):
    """Build the embedding input with field weighting.

    Title and tags are the most discriminative fields for "is this book like
    that book", so they're repeated (weighted up) and placed first. The
    description is truncated and placed last, so if the model's token window
    overflows, it eats the description — never the title/tags.
    """
    parts = []

    title = book.get("title") or ""
    if title:
        parts.extend([title] * TITLE_REPEAT)

    tags = ", ".join(book.get("tags", []))
    if tags:
        parts.extend([tags] * TAGS_REPEAT)

    authors = ", ".join(book.get("authors", []))
    if authors:
        parts.extend([authors] * AUTHOR_REPEAT)

    description = _truncate_words(book.get("description") or "", DESCRIPTION_MAX_WORDS)
    if description:
        parts.extend([description] * DESCRIPTION_REPEAT)

    return " | ".join(parts)


# ── State ────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.books: list = []
        self.book_ids: list = []
        self.id_to_idx: dict = {}
        self.db_ready = False


STATE = AppState()


# ── Caches ────────────────────────────────────────────────────────────────────

# reason cache: (book_id, like_signature) -> reason string.
# invalidate on every write to likes/dislikes.
REASON_CACHE: dict = {}
# likes/dislikes/toread/read are cached (no TTL); seen is always queried fresh so
# the TTL window is respected.
STATE_CACHE: dict = {}  # {"likes":..., "dislikes":..., "toread":..., "read":..., "loaded":False}
REBUILD_LOCK = asyncio.Lock()
DB_LOCK = asyncio.Lock()


def init_db():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                reason_shown TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                book_id INTEGER PRIMARY KEY,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS likes (
                book_id INTEGER PRIMARY KEY,
                liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dislikes (
                book_id INTEGER PRIMARY KEY,
                disliked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS toread (
                book_id INTEGER PRIMARY KEY,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS read (
                book_id INTEGER PRIMARY KEY,
                read_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


async def load_state_from_db():
    if not STATE_CACHE.get("loaded"):
        async with DB_LOCK:
            conn = sqlite3.connect(str(DB_PATH))
            try:
                cur = conn.cursor()
                cur.execute("SELECT book_id FROM likes")
                likes = {r[0] for r in cur.fetchall()}
                cur.execute("SELECT book_id FROM dislikes")
                dislikes = {r[0] for r in cur.fetchall()}
                cur.execute("SELECT book_id FROM toread")
                toread = {r[0] for r in cur.fetchall()}
                cur.execute("SELECT book_id FROM read")
                read_books = {r[0] for r in cur.fetchall()}
            finally:
                conn.close()
        STATE_CACHE["likes"] = likes
        STATE_CACHE["dislikes"] = dislikes
        STATE_CACHE["toread"] = toread
        STATE_CACHE["read"] = read_books
        STATE_CACHE["loaded"] = True
    # seen is always queried fresh so the TTL window is respected
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT book_id FROM seen WHERE last_seen >= datetime('now', ?)",
                (f"-{SEEN_TTL_DAYS} days",),
            )
            seen = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
    return STATE_CACHE["likes"], STATE_CACHE["dislikes"], seen, STATE_CACHE["toread"], STATE_CACHE["read"]


async def record_feedback(fb: Feedback):
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO feedback (book_id, action, reason_shown) VALUES (?, ?, ?)",
                (fb.book_id, fb.action, fb.reason_shown),
            )
            if fb.action == "like":
                cur.execute("INSERT OR REPLACE INTO likes (book_id) VALUES (?)", (fb.book_id,))
                cur.execute("DELETE FROM dislikes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM toread WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM read WHERE book_id=?", (fb.book_id,))
            elif fb.action == "dislike":
                cur.execute("INSERT OR REPLACE INTO dislikes (book_id) VALUES (?)", (fb.book_id,))
                cur.execute("DELETE FROM likes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM toread WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM read WHERE book_id=?", (fb.book_id,))
            elif fb.action == "toread":
                cur.execute("INSERT OR REPLACE INTO toread (book_id) VALUES (?)", (fb.book_id,))
                cur.execute("DELETE FROM likes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM dislikes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM read WHERE book_id=?", (fb.book_id,))
            elif fb.action == "read":
                cur.execute("INSERT OR REPLACE INTO read (book_id) VALUES (?)", (fb.book_id,))
                cur.execute("DELETE FROM likes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM dislikes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM toread WHERE book_id=?", (fb.book_id,))
            cur.execute("INSERT OR REPLACE INTO seen (book_id) VALUES (?)", (fb.book_id,))
            conn.commit()
        finally:
            conn.close()
    if STATE_CACHE.get("loaded"):
        STATE_CACHE["likes"].discard(fb.book_id)
        STATE_CACHE["dislikes"].discard(fb.book_id)
        STATE_CACHE["toread"].discard(fb.book_id)
        STATE_CACHE["read"].discard(fb.book_id)
        if fb.action == "like":
            STATE_CACHE["likes"].add(fb.book_id)
        elif fb.action == "dislike":
            STATE_CACHE["dislikes"].add(fb.book_id)
        elif fb.action == "toread":
            STATE_CACHE["toread"].add(fb.book_id)
        elif fb.action == "read":
            STATE_CACHE["read"].add(fb.book_id)
    REASON_CACHE.clear()


# ── Indexing ─────────────────────────────────────────────────────────────────

def _read_index_meta():
    """Read the index metadata sidecar, or {} if missing/corrupt."""
    try:
        if INDEX_META_PATH.exists():
            return json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _build_index_sync():
    """Synchronous index build — run via asyncio.to_thread()."""
    logger.info("Loading Calibre library...")
    books = get_books()
    if not books:
        raise RuntimeError("No books found in Calibre DB")

    logger.info("Loading embedding model %s...", MODEL_NAME)
    model = STATE.model or SentenceTransformer(MODEL_NAME)

    texts = [embed_text(b) for b in books]
    logger.info("Embedding %d books...", len(texts))
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(INDEX_PATH))
    (CONFIG_DIR / "books.json").write_text(json.dumps(books, default=str), encoding="utf-8")
    INDEX_META_PATH.write_text(
        json.dumps({"embed_version": EMBED_VERSION, "model_name": MODEL_NAME}),
        encoding="utf-8",
    )

    new_state = AppState()
    new_state.model = model
    new_state.index = index
    new_state.books = books
    new_state.book_ids = [b["id"] for b in books]
    new_state.id_to_idx = {bid: i for i, bid in enumerate(new_state.book_ids)}
    return new_state


def _load_index_sync():
    """Synchronous index load — run via asyncio.to_thread()."""
    if not INDEX_PATH.exists() or not (CONFIG_DIR / "books.json").exists():
        return False
    logger.info("Loading existing index...")
    new_state = AppState()
    if STATE.model is None:
        new_state.model = SentenceTransformer(MODEL_NAME)
    else:
        new_state.model = STATE.model
    new_state.index = faiss.read_index(str(INDEX_PATH))
    new_state.books = json.loads((CONFIG_DIR / "books.json").read_text(encoding="utf-8"))
    new_state.book_ids = [b["id"] for b in new_state.books]
    new_state.id_to_idx = {bid: i for i, bid in enumerate(new_state.book_ids)}
    logger.info("Index loaded: %d books", len(new_state.books))
    return new_state


async def ensure_index():
    """Build or load the index and publish it atomically."""
    try:
        mtime = CALIBRE_DB.stat().st_mtime
        index_mtime = INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0
        meta = _read_index_meta()
        stale = (
            not INDEX_PATH.exists()
            or mtime > index_mtime
            or meta.get("embed_version") != EMBED_VERSION
            or meta.get("model_name") != MODEL_NAME
        )
        if stale:
            new_state = await asyncio.to_thread(_build_index_sync)
        else:
            new_state = await asyncio.to_thread(_load_index_sync)
            if new_state is False:
                new_state = await asyncio.to_thread(_build_index_sync)
        publish_state(new_state)
    except Exception as e:
        logger.exception("Failed to build/load index: %s", e)
        raise


def publish_state(new_state):
    """Atomically swap STATE for all readers."""
    global STATE
    STATE = new_state
    STATE_CACHE.clear()
    REASON_CACHE.clear()


# ── Recommendation ─────────────────────────────────────────────────────────

def cover_url(book):
    if not book.get("has_cover"):
        return ""
    return f"/cover/{book['id']}/cover.jpg"


def _rating_stars(book):
    """Calibre stores rating as 0 (unrated) or 2..10 (1..5 stars)."""
    rv = book.get("rating_val")
    if rv is None:
        return 0.0
    try:
        return float(rv) / 2.0
    except (TypeError, ValueError):
        return 0.0


def _dislike_penalty(state, book, dislike_embs):
    """Max cosine similarity between a book and the user's disliked books."""
    if not dislike_embs:
        return 0.0
    emb = state.index.reconstruct(int(state.id_to_idx[book["id"]]))
    return max(float(np.dot(emb, de)) for de in dislike_embs)


def _series_progress(state, likes):
    """Build a {series_name: set(series_index)} map from the user's likes."""
    progress = {}
    for bid in likes:
        idx = state.id_to_idx.get(bid)
        if idx is None:
            continue
        book = state.books[idx]
        sname = book.get("series")
        sidx = book.get("series_index")
        if sname is not None and sidx is not None:
            progress.setdefault(sname, set()).add(float(sidx))
    return progress


def _series_boost(book, series_progress):
    """Return SERIES_BOOST if this book is a later entry in a series the user
    has liked an earlier entry of, else 0.0."""
    sname = book.get("series")
    sidx = book.get("series_index")
    if sname is None or sidx is None:
        return 0.0
    liked_indices = series_progress.get(sname)
    if not liked_indices:
        return 0.0
    sidx = float(sidx)
    if any(liked_idx < sidx for liked_idx in liked_indices):
        return SERIES_BOOST
    return 0.0


def _relevance(book):
    """Combined relevance: similarity + rating boost − dislike penalty + series boost."""
    return (
        book.get("score", 0.0)
        + _rating_stars(book) * 0.03
        - DISLIKE_PENALTY * book.get("dislike_penalty", 0.0)
        + book.get("series_boost", 0.0)
    )


def _taste_centroids_sync(state, signal, max_clusters=5):
    """Synchronous taste centroid clustering — run via asyncio.to_thread()."""
    vecs = []
    for idx, w in signal:
        copies = max(1, int(round(w * 2)))
        v = state.index.reconstruct(int(idx))
        for _ in range(copies):
            vecs.append(v)
    vecs = np.vstack(vecs).astype("float32")
    # Cap clusters by DISTINCT signal points, not duplicated vectors.
    # Weight duplication is only a weighting approximation for k-means;
    # it must not inflate the cluster count.
    n_distinct = len({idx for idx, _ in signal})
    k = min(n_distinct, max_clusters)
    if k <= 1:
        centroid = vecs.mean(axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        return centroid
    kmeans = faiss.Kmeans(state.index.d, k, niter=20, verbose=False, seed=42)
    kmeans.train(vecs)
    centroids = kmeans.centroids
    faiss.normalize_L2(centroids)
    return centroids


def _candidate_pool_sync(state, likes, dislikes, seen, toread, read_books, limit=30, shuffle=True):
    """Synchronous candidate pool generation — run via asyncio.to_thread()."""
    excluded = dislikes | seen | toread | read_books

    dislike_embs = [
        state.index.reconstruct(int(state.id_to_idx[bid]))
        for bid in dislikes
        if bid in state.id_to_idx
    ]

    signal = []
    for bid in likes:
        idx = state.id_to_idx.get(bid)
        if idx is not None:
            signal.append((idx, LIKE_WEIGHT))
    for bid in toread:
        idx = state.id_to_idx.get(bid)
        if idx is not None:
            signal.append((idx, TOREAD_WEIGHT))
    for bid in read_books:
        idx = state.id_to_idx.get(bid)
        if idx is not None:
            signal.append((idx, READ_WEIGHT))

    series_prog = _series_progress(state, likes)

    if not signal:
        pool = []
        for b in state.books:
            if b["id"] in excluded or not b.get("description"):
                continue
            if "omnibus" in b["title"].lower() or "complete" in b["title"].lower() or "collection" in b["title"].lower():
                continue
            d = dict(b)
            d["dislike_penalty"] = _dislike_penalty(state, d, dislike_embs)
            pool.append(d)
        pool.sort(
            key=lambda x: len(x.get("description", ""))
            + len(x.get("tags", [])) * 50
            + _rating_stars(x) * 20
            - DISLIKE_PENALTY * x.get("dislike_penalty", 0.0) * 20,
            reverse=True,
        )
        pool = pool[:limit * 2]
        if shuffle:
            random.shuffle(pool)
        return pool[:limit]

    centroids = _taste_centroids_sync(state, signal)
    D, I = state.index.search(centroids, 200)

    pool = []
    found_ids = set()
    n_clusters = I.shape[0]
    for col in range(I.shape[1]):
        for row in range(n_clusters):
            idx = int(I[row, col])
            score = float(D[row, col])
            book = state.books[idx]
            bid = book["id"]
            if bid in likes or bid in excluded or bid in found_ids:
                continue
            book = dict(book)
            book["score"] = score
            book["dislike_penalty"] = _dislike_penalty(state, book, dislike_embs)
            book["series_boost"] = _series_boost(book, series_prog)
            pool.append(book)
            found_ids.add(bid)
            if len(pool) >= limit:
                break
        if len(pool) >= limit:
            break

    if shuffle:
        random.shuffle(pool)
    return pool


def _exploration_pool_sync(state, likes, dislikes, seen, toread, read_books, count):
    """Synchronous exploration pool — run via asyncio.to_thread()."""
    if count <= 0:
        return []
    excluded = likes | dislikes | seen | toread | read_books
    dislike_embs = [
        state.index.reconstruct(int(state.id_to_idx[bid]))
        for bid in dislikes
        if bid in state.id_to_idx
    ]
    eligible = []
    for b in state.books:
        if b["id"] in excluded:
            continue
        d = dict(b)
        d["score"] = 0.0
        d["dislike_penalty"] = _dislike_penalty(state, d, dislike_embs)
        eligible.append(d)
    random.shuffle(eligible)
    return eligible[:count]


def _diversify_sync(state, candidates, count, lambda_=0.7):
    """Synchronous MMR diversification — run via asyncio.to_thread()."""
    if len(candidates) <= count:
        return candidates
    embs = {c["id"]: state.index.reconstruct(int(state.id_to_idx[c["id"]])) for c in candidates}
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        best = None
        best_mmr = -float("inf")
        for c in remaining:
            rel = _relevance(c)
            if selected:
                ce = embs[c["id"]]
                max_sim = max(float(np.dot(ce, embs[s["id"]])) for s in selected)
            else:
                max_sim = 0.0
            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = c
        selected.append(best)
        remaining.remove(best)
    return selected


async def llm_reason(book, liked_titles, like_sig, similar_to=None):
    if not OLLAMA_URL:
        return ""
    cache_key = (book["id"], like_sig)
    if cache_key in REASON_CACHE:
        return REASON_CACHE[cache_key]
    if similar_to:
        prompt = f"""Recommend a book similar to "{similar_to}".
Recommend the book "{book['title']}" by {', '.join(book.get('authors', [])) or 'unknown'} in one short sentence (under 25 words). Mention a specific theme, style, or mood that connects it to "{similar_to}". Keep it casual."""
    else:
        prompt = f"""The user likes these books: {liked_titles}.
Recommend the book "{book['title']}" by {', '.join(book.get('authors', [])) or 'unknown'} in one short sentence (under 25 words). Mention a specific theme, style, or mood that connects it to what they like. Keep it casual."""
    try:
        base = OLLAMA_URL.rstrip("/")
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{base}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            data = r.json()
            reason = data.get("response", "").strip()
    except Exception as e:
        logger.debug("LLM reason failed: %s", e)
        return ""
    if reason:
        REASON_CACHE[cache_key] = reason
    return reason


def deterministic_reason(book, liked_titles, similar_to=None):
    tags = ", ".join(book.get("tags", [])[:5])
    authors = ", ".join(book.get("authors", [])[:2])
    if similar_to:
        return f"Similar to {similar_to}: {tags} by {authors}." if tags else f"Similar to {similar_to}."
    if liked_titles:
        return f"Because you liked {liked_titles[0]}: {tags} by {authors}." if tags else f"Because you liked {liked_titles[0]}."
    return f"{tags} by {authors}" if tags else f"by {authors}"


def _liked_titles(state, likes):
    titles = []
    for bid in list(likes)[:3]:
        idx = state.id_to_idx.get(bid)
        if idx is not None:
            titles.append(state.books[idx]["title"])
    return titles


async def _decorate_book(state, book, likes, liked_titles, like_sig, similar_to=None):
    """Attach reason + cover_url to a candidate book (mutates the dict copy)."""
    reason = await llm_reason(book, liked_titles, like_sig, similar_to)
    if not reason:
        reason = deterministic_reason(book, liked_titles, similar_to)
    book["reason"] = reason
    book["cover_url"] = cover_url(book)
    return book


async def _mark_seen(book, likes, dislikes, seen, toread, read_books):
    bid = book["id"]
    if bid in seen or bid in likes or bid in dislikes or bid in toread or bid in read_books:
        return
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO seen (book_id) VALUES (?)", (bid,))
            conn.commit()
        finally:
            conn.close()


async def batch_recommendations(state, count=10):
    """Return up to `count` randomized, diverse recommendations with reasons."""
    likes, dislikes, seen, toread, read_books = await load_state_from_db()

    n_explore = int(count * EXPLORATION_RATE)
    n_exploit = count - n_explore

    # Offload CPU-bound work to thread pool
    if likes or toread or read_books:
        pool = await asyncio.to_thread(
            _candidate_pool_sync, state, likes, dislikes, seen, toread, read_books,
            max(n_exploit * 5, 50), False
        )
        pool = await asyncio.to_thread(_diversify_sync, state, pool, n_exploit)
    else:
        pool = await asyncio.to_thread(
            _candidate_pool_sync, state, likes, dislikes, seen, toread, read_books,
            n_exploit, True
        )

    explore = await asyncio.to_thread(
        _exploration_pool_sync, state, likes, dislikes, seen, toread, read_books, n_explore
    )

    combined = pool + explore
    random.shuffle(combined)
    deduped = []
    seen_ids = set()
    for b in combined:
        if b["id"] in seen_ids:
            continue
        seen_ids.add(b["id"])
        deduped.append(b)
    combined = deduped[:count]

    if not combined:
        return []
    liked_titles = _liked_titles(state, likes)
    like_sig = tuple(sorted(likes))

    books = await asyncio.gather(
        *[_decorate_book(state, b, likes, liked_titles, like_sig) for b in combined]
    )

    for b in books:
        await _mark_seen(b, likes, dislikes, seen, toread, read_books)
    return books


async def next_recommendation(state):
    likes, dislikes, seen, toread, read_books = await load_state_from_db()
    pool = await asyncio.to_thread(
        _candidate_pool_sync, state, likes, dislikes, seen, toread, read_books, 30, True
    )
    if not pool:
        return None
    book = pool[0]
    liked_titles = _liked_titles(state, likes)
    like_sig = tuple(sorted(likes))

    book = await _decorate_book(state, book, likes, liked_titles, like_sig)
    await _mark_seen(book, likes, dislikes, seen, toread, read_books)
    return book


async def more_like_books(state, book_id, count=10):
    """One-shot "more like this": nearest neighbours of a single book."""
    idx = state.id_to_idx.get(book_id)
    if idx is None:
        return None
    likes, dislikes, seen, toread, read_books = await load_state_from_db()
    source = state.books[idx]

    def _search_sync():
        query = state.index.reconstruct(int(idx)).reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)
        D, I = state.index.search(query, 200)
        excluded = likes | dislikes | seen | toread | read_books | {book_id}
        pool = []
        for score, nidx in zip(D[0], I[0]):
            nidx = int(nidx)
            book = state.books[nidx]
            bid = book["id"]
            if bid in excluded:
                continue
            b = dict(book)
            b["score"] = float(score)
            pool.append(b)
            if len(pool) >= count:
                break
        return pool

    pool = await asyncio.to_thread(_search_sync)
    if not pool:
        return []

    like_sig = ("more", book_id)
    books = await asyncio.gather(
        *[_decorate_book(state, b, likes, [], like_sig, similar_to=source["title"]) for b in pool]
    )
    return books


# ── FastAPI app ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global OLLAMA_URL, OLLAMA_MODEL
    init_db()
    try:
        env = json.loads((CONFIG_DIR / "env.json").read_text()) if (CONFIG_DIR / "env.json").exists() else {}
    except Exception:
        env = {}
    OLLAMA_URL = env.get("OLLAMA_URL", "") or os.environ.get("OLLAMA_URL", "")
    OLLAMA_MODEL = env.get("OLLAMA_MODEL", "") or os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    logger.info("OLLAMA_URL=%s", OLLAMA_URL or "(disabled)")
    await ensure_index()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/api/recommend")
async def recommend():
    state = STATE
    book = await next_recommendation(state)
    if not book:
        return {"done": True}
    return {"done": False, "book": book}


@app.get("/api/recommendations")
async def recommendations(count: int = 10):
    state = STATE
    count = max(1, min(count, 50))
    books = await batch_recommendations(state, count=count)
    return {"done": len(books) == 0, "books": books}


@app.get("/api/more-like/{book_id}")
async def more_like(book_id: int, count: int = 10):
    state = STATE
    count = max(1, min(count, 50))
    books = await more_like_books(state, book_id, count=count)
    if books is None:
        return Response(status_code=404)
    return {"done": len(books) == 0, "books": books}


@app.get("/api/list/{action}")
async def list_by_action(action: str):
    valid_actions = {"like", "dislike", "skip", "toread", "read"}
    if action not in valid_actions:
        return Response(status_code=400, content=json.dumps({"error": "invalid action"}), media_type="application/json")

    state = STATE
    books = []
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT book_id FROM feedback WHERE action=? ORDER BY created_at DESC",
                (action,),
            )
            for row in cur.fetchall():
                bid = row[0]
                if bid in state.id_to_idx:
                    b = dict(state.books[state.id_to_idx[bid]])
                    b["why"] = f"{action.title()}d"
                    b["cover_url"] = cover_url(b)
                    books.append(b)
        finally:
            conn.close()

    return {"books": books, "done": len(books) == 0}


@app.post("/api/feedback")
async def feedback(fb: Feedback):
    if fb.book_id not in STATE.id_to_idx:
        return {"ok": False, "error": "unknown book_id"}
    await record_feedback(fb)
    return {"ok": True}


@app.post("/api/reset-seen")
async def reset_seen():
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM seen")
            conn.commit()
        finally:
            conn.close()
    return {"ok": True}


@app.get("/api/stats")
async def stats():
    state = STATE
    likes, dislikes, seen, toread, read_books = await load_state_from_db()
    return {
        "total": len(state.books),
        "liked": len(likes),
        "disliked": len(dislikes),
        "seen": len(seen),
        "toread": len(toread),
        "read": len(read_books),
    }


@app.get("/api/rebuild")
async def rebuild_index():
    async with REBUILD_LOCK:
        await ensure_index()
    return {"ok": True, "count": len(STATE.books)}


# ── Cover serving ────────────────────────────────────────────────────────────

LIBRARY_PATH = Path("/calibre").resolve()


@app.get("/cover/{book_id}/{filename}")
async def cover(book_id: int, filename: str):
    state = STATE
    idx = state.id_to_idx.get(book_id)
    if idx is None:
        return Response(status_code=404)
    b = state.books[idx]
    book_dir = (LIBRARY_PATH / b["path"]).resolve()
    try:
        book_dir.relative_to(LIBRARY_PATH)
    except ValueError:
        return Response(status_code=400)
    cover_path = book_dir / filename
    try:
        cover_path.resolve().relative_to(book_dir)
    except ValueError:
        return Response(status_code=400)
    if cover_path.is_file():
        return FileResponse(str(cover_path))
    return Response(status_code=404)
