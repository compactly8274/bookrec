from pathlib import Path
import sqlite3
import json
import logging
import os
import asyncio
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

DATA_DIR = Path("/data")
CONFIG_DIR = Path("/config")
DB_PATH = CONFIG_DIR / "bookrec.db"
CALIBRE_DB = Path("/calibre/metadata.db")
INDEX_PATH = CONFIG_DIR / "index.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = ""
OLLAMA_MODEL = "gemma3:4b"


class Feedback(BaseModel):
    book_id: int
    action: str = Field(..., pattern="^(like|dislike|skip|more)$")
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


def embed_text(book):
    parts = []
    if book["title"]:
        parts.append(book["title"])
    if book["authors"]:
        parts.append(", ".join(book["authors"]))
    if book["tags"]:
        parts.append(", ".join(book["tags"]))
    if book["description"]:
        parts.append(book["description"])
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


# ── Caches ────────────────────────────────────────────────────────────────────

# reason cache: (book_id, like_signature) -> reason string.
# invalidate on every write to likes/dislikes.
REASON_CACHE: dict = {}
LIKE_SIG: tuple = ()
STATE_CACHE: dict = {}  # {"likes":..., "dislikes":..., "seen":..., "loaded":False}
INDEX_LOCK = asyncio.Lock()
REBUILD_LOCK = asyncio.Lock()
DB_LOCK = asyncio.Lock()


def init_db():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
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
    conn.commit()
    conn.close()


async def load_state_from_db():
    if STATE_CACHE.get("loaded"):
        return STATE_CACHE["likes"], STATE_CACHE["dislikes"], STATE_CACHE["seen"]
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT book_id FROM likes")
        likes = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT book_id FROM dislikes")
        dislikes = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT book_id FROM seen")
        seen = {r[0] for r in cur.fetchall()}
        conn.close()
    STATE_CACHE["likes"] = likes
    STATE_CACHE["dislikes"] = dislikes
    STATE_CACHE["seen"] = seen
    STATE_CACHE["loaded"] = True
    return likes, dislikes, seen


async def record_feedback(fb: Feedback):
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO feedback (book_id, action, reason_shown) VALUES (?, ?, ?)",
            (fb.book_id, fb.action, fb.reason_shown),
        )
        if fb.action == "like":
            cur.execute("INSERT OR REPLACE INTO likes (book_id) VALUES (?)", (fb.book_id,))
            cur.execute("DELETE FROM dislikes WHERE book_id=?", (fb.book_id,))
        elif fb.action == "dislike":
            cur.execute("INSERT OR REPLACE INTO dislikes (book_id) VALUES (?)", (fb.book_id,))
            cur.execute("DELETE FROM likes WHERE book_id=?", (fb.book_id,))
        # mark seen for any action so a liked/disliked book won't resurface
        cur.execute("INSERT OR REPLACE INTO seen (book_id) VALUES (?)", (fb.book_id,))
        conn.commit()
        conn.close()
    # in-memory caches: update immediately so the next request sees the new state
    if STATE_CACHE.get("loaded"):
        STATE_CACHE["likes"].discard(fb.book_id)
        STATE_CACHE["dislikes"].discard(fb.book_id)
        STATE_CACHE["seen"].add(fb.book_id)
        if fb.action == "like":
            STATE_CACHE["likes"].add(fb.book_id)
        elif fb.action == "dislike":
            STATE_CACHE["dislikes"].add(fb.book_id)
    # reasons depend on the user's likes signature — invalidate on any like change
    REASON_CACHE.clear()


# ── Indexing ─────────────────────────────────────────────────────────────────

def build_index():
    logger.info("Loading Calibre library...")
    books = get_books()
    if not books:
        raise RuntimeError("No books found in Calibre DB")

    logger.info("Loading embedding model %s...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = [embed_text(b) for b in books]
    logger.info("Embedding %d books...", len(texts))
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(INDEX_PATH))
    (CONFIG_DIR / "books.json").write_text(json.dumps(books, default=str), encoding="utf-8")

    STATE.model = model
    STATE.index = index
    STATE.books = books
    STATE.book_ids = [b["id"] for b in books]
    STATE.id_to_idx = {bid: i for i, bid in enumerate(STATE.book_ids)}
    # book list changed: any in-memory state referencing old ids is stale
    STATE_CACHE.clear()
    REASON_CACHE.clear()
    logger.info("Index built: %d books", len(books))


def load_index():
    if not INDEX_PATH.exists() or not (CONFIG_DIR / "books.json").exists():
        return False
    logger.info("Loading existing index...")
    # only instantiate the model once; reuse across reloads
    if STATE.model is None:
        STATE.model = SentenceTransformer(MODEL_NAME)
    STATE.index = faiss.read_index(str(INDEX_PATH))
    STATE.books = json.loads((CONFIG_DIR / "books.json").read_text(encoding="utf-8"))
    STATE.book_ids = [b["id"] for b in STATE.books]
    STATE.id_to_idx = {bid: i for i, bid in enumerate(STATE.book_ids)}
    logger.info("Index loaded: %d books", len(STATE.books))
    return True


def ensure_index():
    try:
        mtime = CALIBRE_DB.stat().st_mtime
        index_mtime = INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0
        if not load_index() or mtime > index_mtime:
            build_index()
    except Exception as e:
        logger.exception("Failed to build/load index: %s", e)
        raise


# ── Recommendation ─────────────────────────────────────────────────────────

def cover_url(book):
    if not book.get("has_cover"):
        return ""
    return f"/cover/{book['id']}/cover.jpg"


def candidate_pool(likes, dislikes, seen, limit=30):
    if not likes:
        # cold start: prefer single books with good metadata, not omnibuses
        pool = []
        for b in STATE.books:
            if b["id"] in dislikes or b["id"] in seen or not b.get("description"):
                continue
            if "omnibus" in b["title"].lower() or "complete" in b["title"].lower() or "collection" in b["title"].lower():
                continue
            pool.append(b)
        pool.sort(key=lambda x: len(x.get("description", "")) + len(x.get("tags", [])) * 50, reverse=True)
        pool = pool[:limit * 2]
        np.random.shuffle(pool)
        return pool[:limit]

    # personalised: nearest neighbour over the user's likes
    liked_indices = []
    for bid in likes:
        idx = STATE.id_to_idx.get(bid)
        if idx is not None:
            liked_indices.append(idx)
    if not liked_indices:
        return candidate_pool(set(), dislikes, seen, limit)

    query = np.zeros((1, STATE.index.d), dtype="float32")
    for idx in liked_indices:
        query += STATE.index.reconstruct(int(idx))
    query /= len(liked_indices)
    faiss.normalize_L2(query)

    D, I = STATE.index.search(query, 200)
    pool = []
    found_ids = set()
    for score, idx in zip(D[0], I[0]):
        book = STATE.books[idx]
        bid = book["id"]
        if bid in likes or bid in dislikes or bid in seen or bid in found_ids:
            continue
        book = dict(book)
        book["score"] = float(score)
        book["reason"] = "similar to books you liked"
        pool.append(book)
        found_ids.add(bid)
        if len(pool) >= limit:
            break
    return pool


async def llm_reason(book, liked_titles, like_sig):
    if not OLLAMA_URL:
        return ""
    cache_key = (book["id"], like_sig)
    if cache_key in REASON_CACHE:
        return REASON_CACHE[cache_key]
    prompt = f"""The user likes these books: {liked_titles}.
Recommend the book "{book['title']}" by {', '.join(book.get('authors', [])) or 'unknown'} in one short sentence (under 25 words). Mention a specific theme, style, or mood that connects it to what they like. Keep it casual."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
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


def deterministic_reason(book, liked_titles):
    tags = ", ".join(book.get("tags", [])[:5])
    authors = ", ".join(book.get("authors", [])[:2])
    if liked_titles:
        return f"Because you liked {liked_titles[0]}: {tags} by {authors}." if tags else f"Because you liked {liked_titles[0]}."
    return f"{tags} by {authors}" if tags else f"by {authors}"


async def next_recommendation():
    likes, dislikes, seen = await load_state_from_db()
    pool = candidate_pool(likes, dislikes, seen, limit=30)
    if not pool:
        return None
    book = pool[0]
    liked_titles = []
    for bid in list(likes)[:3]:
        idx = STATE.id_to_idx.get(bid)
        if idx is not None:
            liked_titles.append(STATE.books[idx]["title"])

    # signature of current like set, used to memoize LLM reasons
    like_sig = tuple(sorted(likes))

    reason = await llm_reason(book, liked_titles, like_sig)
    if not reason:
        reason = deterministic_reason(book, liked_titles)

    book["reason"] = reason
    book["cover_url"] = cover_url(book)

    # mark as seen so a card that the user just looks at but doesn't click
    # won't resurface on the next /api/recommend
    bid = book["id"]
    if bid not in seen and bid not in likes and bid not in dislikes:
        async with DB_LOCK:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO seen (book_id) VALUES (?)", (bid,))
            conn.commit()
            conn.close()
        seen.add(bid)
        if STATE_CACHE.get("loaded"):
            STATE_CACHE["seen"].add(bid)

    return book


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
    book = await next_recommendation()
    if not book:
        return {"done": True}
    return {"done": False, "book": book}


@app.post("/api/feedback")
async def feedback(fb: Feedback):
    if fb.action == "more":
        # "more like this" both seeds the recommender (as a like) and records the "more" event
        fb_like = Feedback(book_id=fb.book_id, action="like", reason_shown=fb.reason_shown)
        await record_feedback(fb_like)
        async with DB_LOCK:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO feedback (book_id, action, reason_shown) VALUES (?, ?, ?)",
                (fb.book_id, "more", fb.reason_shown),
            )
            conn.commit()
            conn.close()
    else:
        await record_feedback(fb)
    return {"ok": True}


@app.get("/api/stats")
async def stats():
    likes, dislikes, seen = await load_state_from_db()
    return {"total": len(STATE.books), "liked": len(likes), "disliked": len(dislikes), "seen": len(seen)}


@app.get("/api/rebuild")
async def rebuild_index():
    async with REBUILD_LOCK:
        await asyncio.to_thread(ensure_index)
    return {"ok": True, "count": len(STATE.books)}


# ── Cover serving ────────────────────────────────────────────────────────────

LIBRARY_PATH = Path("/calibre").resolve()


@app.get("/cover/{book_id}/{filename}")
async def cover(book_id: int, filename: str):
    # Calibre stores cover.jpg inside the book's path folder.
    # Reject any book whose resolved path escapes the library root.
    for b in STATE.books:
        if b["id"] == book_id:
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
            break
    return Response(status_code=404)
