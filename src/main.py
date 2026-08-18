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
EMBED_VERSION = 2

OLLAMA_URL = ""
OLLAMA_MODEL = "gemma3:4b"

# Positive-signal weights: a "like" is a strong signal, "to read" is a soft one.
LIKE_WEIGHT = 1.0
TOREAD_WEIGHT = 0.5

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


class Feedback(BaseModel):
    book_id: int
    action: str = Field(..., pattern="^(like|dislike|skip|toread)$")
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

    # Authors and tags
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
    conn.close()

    for b in books:
        b["authors"] = authors.get(b["id"], [])
        b["tags"] = tags.get(b["id"], [])
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
# likes/dislikes/toread are cached (no TTL); seen is always queried fresh so
# the TTL window is respected.
STATE_CACHE: dict = {}  # {"likes":..., "dislikes":..., "toread":..., "loaded":False}
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
            finally:
                conn.close()
        STATE_CACHE["likes"] = likes
        STATE_CACHE["dislikes"] = dislikes
        STATE_CACHE["toread"] = toread
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
    return STATE_CACHE["likes"], STATE_CACHE["dislikes"], seen, STATE_CACHE["toread"]


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
            elif fb.action == "dislike":
                cur.execute("INSERT OR REPLACE INTO dislikes (book_id) VALUES (?)", (fb.book_id,))
                cur.execute("DELETE FROM likes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM toread WHERE book_id=?", (fb.book_id,))
            elif fb.action == "toread":
                # "already own it, haven't read it" — a soft positive signal.
                # Record it, but don't treat it as a full like/dislike.
                cur.execute("INSERT OR REPLACE INTO toread (book_id) VALUES (?)", (fb.book_id,))
                cur.execute("DELETE FROM likes WHERE book_id=?", (fb.book_id,))
                cur.execute("DELETE FROM dislikes WHERE book_id=?", (fb.book_id,))
            # mark seen for any action so a liked/disliked/toread book won't resurface
            cur.execute("INSERT OR REPLACE INTO seen (book_id) VALUES (?)", (fb.book_id,))
            conn.commit()
        finally:
            conn.close()
    # in-memory caches: update immediately so the next request sees the new state
    if STATE_CACHE.get("loaded"):
        STATE_CACHE["likes"].discard(fb.book_id)
        STATE_CACHE["dislikes"].discard(fb.book_id)
        STATE_CACHE["toread"].discard(fb.book_id)
        if fb.action == "like":
            STATE_CACHE["likes"].add(fb.book_id)
        elif fb.action == "dislike":
            STATE_CACHE["dislikes"].add(fb.book_id)
        elif fb.action == "toread":
            STATE_CACHE["toread"].add(fb.book_id)
    # reasons depend on the user's likes signature — invalidate on any like change
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


def build_index():
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

    # build a fresh state, publish atomically
    new_state = AppState()
    new_state.model = model
    new_state.index = index
    new_state.books = books
    new_state.book_ids = [b["id"] for b in books]
    new_state.id_to_idx = {bid: i for i, bid in enumerate(new_state.book_ids)}
    return new_state


def load_index():
    if not INDEX_PATH.exists() or not (CONFIG_DIR / "books.json").exists():
        return False
    logger.info("Loading existing index...")
    # only instantiate the model once; reuse across reloads
    new_state = AppState()
    if STATE.model is None:
        new_state.model = SentenceTransformer(MODEL_NAME)
    else:
        new_state.model = STATE.model  # reuse the loaded model
    new_state.index = faiss.read_index(str(INDEX_PATH))
    new_state.books = json.loads((CONFIG_DIR / "books.json").read_text(encoding="utf-8"))
    new_state.book_ids = [b["id"] for b in new_state.books]
    new_state.id_to_idx = {bid: i for i, bid in enumerate(new_state.book_ids)}
    logger.info("Index loaded: %d books", len(new_state.books))
    return new_state


def ensure_index():
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
            new_state = build_index()
        else:
            new_state = load_index()
            if new_state is False:  # index exists but books.json missing
                new_state = build_index()
        publish_state(new_state)
    except Exception as e:
        logger.exception("Failed to build/load index: %s", e)
        raise


def publish_state(new_state):
    """Atomically swap STATE for all readers."""
    global STATE
    STATE = new_state
    # book list changed: any in-memory state referencing old ids is stale
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
    """Max cosine similarity between a book and the user's disliked books.

    Used to actively push candidates away from disliked content (not just
    exclude the disliked books themselves).
    """
    if not dislike_embs:
        return 0.0
    emb = state.index.reconstruct(int(state.id_to_idx[book["id"]]))
    return max(float(np.dot(emb, de)) for de in dislike_embs)


def _relevance(book):
    """Combined relevance: similarity to taste + rating boost − dislike penalty."""
    return (
        book.get("score", 0.0)
        + _rating_stars(book) * 0.03
        - DISLIKE_PENALTY * book.get("dislike_penalty", 0.0)
    )


def _taste_centroids(state, signal, max_clusters=5):
    """Cluster the user's positive signals into distinct taste centroids.

    `signal` is a list of (index, weight) pairs. Likes carry full weight
    (LIKE_WEIGHT); "to read" books carry half weight (TOREAD_WEIGHT) as a soft
    positive signal.

    Averaging all signals into a single vector collapses distinct tastes
    (e.g. sci-fi + history) into a meaningless midpoint. Instead we cluster
    the signals and return one centroid per taste, so each taste gets its own
    nearest-neighbour query.

    faiss.Kmeans has no native per-point weights, so we approximate them by
    duplicating vectors: weight 1.0 -> 2 copies, weight 0.5 -> 1 copy. This
    gives to-read books half the influence of a like in the clustering.
    """
    vecs = []
    for idx, w in signal:
        copies = max(1, int(round(w * 2)))
        v = state.index.reconstruct(int(idx))
        for _ in range(copies):
            vecs.append(v)
    vecs = np.vstack(vecs).astype("float32")
    n = vecs.shape[0]
    k = min(n, max_clusters)
    if k <= 1:
        centroid = vecs.mean(axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        return centroid
    kmeans = faiss.Kmeans(state.index.d, k, niter=20, verbose=False, seed=42)
    kmeans.train(vecs)
    centroids = kmeans.centroids
    faiss.normalize_L2(centroids)
    return centroids


def candidate_pool(state, likes, dislikes, seen, toread, limit=30, shuffle=True):
    """Return a candidate pool of up to `limit` books.

    When `shuffle` is True the pool is randomized so repeated calls don't walk
    the library in a fixed order. The personalized path otherwise returns
    nearest neighbours in descending similarity, which reads as "in order".
    """
    excluded = dislikes | seen | toread

    # Precompute disliked embeddings once for the negative-signal penalty.
    dislike_embs = [
        state.index.reconstruct(int(state.id_to_idx[bid]))
        for bid in dislikes
        if bid in state.id_to_idx
    ]

    # Build the positive signal: likes (full weight) + to-read (soft weight).
    signal = []
    for bid in likes:
        idx = state.id_to_idx.get(bid)
        if idx is not None:
            signal.append((idx, LIKE_WEIGHT))
    for bid in toread:
        idx = state.id_to_idx.get(bid)
        if idx is not None:
            signal.append((idx, TOREAD_WEIGHT))

    if not signal:
        # cold start: no positive signal at all
        pool = []
        for b in state.books:
            if b["id"] in excluded or not b.get("description"):
                continue
            if "omnibus" in b["title"].lower() or "complete" in b["title"].lower() or "collection" in b["title"].lower():
                continue
            # defensive copy so request handlers can mutate the candidate without
            # touching the canonical STATE.books entry
            d = dict(b)
            d["dislike_penalty"] = _dislike_penalty(state, d, dislike_embs)
            pool.append(d)
        # richness + rating − dislike penalty: surface well-described, well-rated,
        # not-disliked-similar books first
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

    centroids = _taste_centroids(state, signal)

    # search all taste centroids in one batched call: D/I are (k, n)
    D, I = state.index.search(centroids, 200)

    # interleave results across clusters so a single dominant taste can't
    # crowd out the others, then dedupe by book id
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
            pool.append(book)
            found_ids.add(bid)
            if len(pool) >= limit:
                break
        if len(pool) >= limit:
            break

    if shuffle:
        random.shuffle(pool)
    return pool


def exploration_pool(state, likes, dislikes, seen, toread, count):
    """Draw `count` random books from the full library for serendipity.

    Exploration is random but never hostile: it still excludes disliked books
    (and applies the dislike penalty), seen, to-read, and liked books. It does
    NOT constrain to the user's taste clusters — that's the whole point.
    """
    if count <= 0:
        return []
    excluded = likes | dislikes | seen | toread
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


def diversify(state, candidates, count, lambda_=0.7):
    """Select up to `count` diverse candidates via Maximal Marginal Relevance.

    Greedily picks candidates that are both relevant to the user's taste
    (`_relevance`, which now includes a dislike penalty) and dissimilar to
    what's already been selected (cosine distance between embeddings). This
    prevents a batch from being dominated by near-duplicate books (e.g. 8
    books by the same author).
    """
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


async def _mark_seen(book, likes, dislikes, seen, toread):
    bid = book["id"]
    if bid in seen or bid in likes or bid in dislikes or bid in toread:
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
    """Return up to `count` randomized, diverse recommendations with reasons.

    Mixes exploitation (nearest neighbours of taste) with exploration (random
    picks from the full library) at EXPLORATION_RATE.
    """
    likes, dislikes, seen, toread = await load_state_from_db()

    # floor (not round) so EXPLORATION_RATE is a ceiling, not a banker's-round
    # surprise: 15% of 10 = 1 exploration pick, not 2.
    n_explore = int(count * EXPLORATION_RATE)
    n_exploit = count - n_explore

    if likes or toread:
        # exploitation: fetch a larger ranked pool, then select a diverse subset
        pool = candidate_pool(state, likes, dislikes, seen, toread, limit=max(n_exploit * 5, 50), shuffle=False)
        pool = diversify(state, pool, n_exploit)
    else:
        # cold start: no taste signal, so a random draw is already diverse
        pool = candidate_pool(state, likes, dislikes, seen, toread, limit=n_exploit, shuffle=True)

    # exploration: random picks from the full library (respecting dislikes)
    explore = exploration_pool(state, likes, dislikes, seen, toread, n_explore)

    combined = pool + explore
    random.shuffle(combined)
    # dedupe by book id — exploitation and exploration pools can overlap
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

    # generate reasons concurrently (LLM calls are the slow part)
    books = await asyncio.gather(
        *[_decorate_book(state, b, likes, liked_titles, like_sig) for b in combined]
    )

    # mark all shown books as seen so the next batch doesn't repeat them
    for b in books:
        await _mark_seen(b, likes, dislikes, seen, toread)
    return books


async def next_recommendation(state):
    likes, dislikes, seen, toread = await load_state_from_db()
    pool = candidate_pool(state, likes, dislikes, seen, toread, limit=30, shuffle=True)
    if not pool:
        return None
    book = pool[0]
    liked_titles = _liked_titles(state, likes)
    like_sig = tuple(sorted(likes))

    book = await _decorate_book(state, book, likes, liked_titles, like_sig)
    await _mark_seen(book, likes, dislikes, seen, toread)
    return book


async def more_like_books(state, book_id, count=10):
    """One-shot "more like this": nearest neighbours of a single book.

    This is a pure query — it does NOT persist anything and does NOT seed the
    recommender. The source book's embedding is used directly as the query.
    """
    idx = state.id_to_idx.get(book_id)
    if idx is None:
        return None
    likes, dislikes, seen, toread = await load_state_from_db()
    source = state.books[idx]

    query = state.index.reconstruct(int(idx)).reshape(1, -1).astype("float32")
    faiss.normalize_L2(query)
    D, I = state.index.search(query, 200)

    excluded = likes | dislikes | seen | toread | {book_id}
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
    ensure_index()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/api/recommend")
async def recommend():
    state = STATE  # snapshot: keep a consistent view across the whole request
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


@app.post("/api/feedback")
async def feedback(fb: Feedback):
    # reject feedback for books not in the index (LAN-only deployment, but no reason to write garbage)
    if fb.book_id not in STATE.id_to_idx:
        return {"ok": False, "error": "unknown book_id"}
    await record_feedback(fb)
    return {"ok": True}


@app.post("/api/reset-seen")
async def reset_seen():
    """Clear the seen history so all books become eligible again."""
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
    likes, dislikes, seen, toread = await load_state_from_db()
    return {
        "total": len(state.books),
        "liked": len(likes),
        "disliked": len(dislikes),
        "seen": len(seen),
        "toread": len(toread),
    }


@app.get("/api/rebuild")
async def rebuild_index():
    async with REBUILD_LOCK:
        await asyncio.to_thread(ensure_index)
    return {"ok": True, "count": len(STATE.books)}


# ── Cover serving ────────────────────────────────────────────────────────────

LIBRARY_PATH = Path("/calibre").resolve()


@app.get("/cover/{book_id}/{filename}")
async def cover(book_id: int, filename: str):
    state = STATE  # snapshot: avoid torn reads if a rebuild swaps STATE mid-request
    # O(1) book lookup via id_to_idx
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
    # ensure the final resolved path is still under the book dir
    try:
        cover_path.resolve().relative_to(book_dir)
    except ValueError:
        return Response(status_code=400)
    if cover_path.is_file():
        return FileResponse(str(cover_path))
    return Response(status_code=404)
