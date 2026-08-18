# BookRec

Self-hosted book recommender that reads your existing Calibre library and surfaces personalised recommendations in a table view.

- Reads Calibre `metadata.db` read-only.
- Embeds books locally with `all-MiniLM-L6-v2`.
- Learns from your 👍 / 👎 / skip / 📚 To Read / "More like this" feedback.
- Multi-taste clustering, MMR diversity, dislike penalties, exploration, and series awareness.
- Optionally generates reasons via any Ollama-compatible endpoint.

## Run

```bash
docker compose up -d
```

Open `http://your-host:8484`.

## UI

The main view shows a **table of 10 recommendations** at a time, each with cover, title, author, a short reason, description, and action buttons:

| Action | What it does |
|---|---|
| 👍 Like | Marks the book as a strong positive taste signal. Excluded from future recs. |
| 👎 Dislike | Actively pushes recommendations *away* from similar books (not just exclusion). |
| Skip | Marks the book as seen. It'll resurface after 30 days. |
| 📚 To Read | "Already own it, haven't read it." A soft positive signal (half the weight of a like). Excluded from future recs. |
| More | One-shot "more like this" — shows books similar to that specific book without persisting anything. |
| 🔄 New batch | Fetches a fresh randomized set of 10 recommendations. |
| ↺ Reset seen | Clears the seen history so all books become eligible again. |

## Recommendation algorithm

### How it works

1. **Embedding** — Each book is embedded using `all-MiniLM-L6-v2` with field weighting: title and tags are repeated (weighted up) and placed first; the description is truncated to 120 words and placed last, so it can't drown out the more discriminative fields.

2. **Multi-centroid taste clustering** — Your liked and to-read books are clustered into up to 5 distinct taste centroids using `faiss.Kmeans`. Each taste gets its own nearest-neighbour query, so your sci-fi likes and history likes don't collapse into a meaningless average.

3. **MMR diversity** — Candidates are selected via Maximal Marginal Relevance: each pick is both relevant to your taste *and* dissimilar to what's already in the batch. Prevents a batch of 10 from being 8 books by the same author.

4. **Dislike penalty** — Disliked books aren't just excluded; they actively push candidates away. A candidate's max cosine similarity to any disliked book is subtracted from its relevance score.

5. **Series awareness** — If you've liked an earlier book in a series, later entries in that series get a relevance boost (even if embedding similarity is weak — sequels often have different descriptions).

6. **Exploration** — 15% of each batch is random picks from the full library (excluding dislikes), so you get serendipity and discovery of new genres.

7. **Seen TTL** — Books marked "seen" become eligible again after 30 days. Use "↺ Reset seen" to clear immediately.

### Tunable constants

All algorithm parameters are constants at the top of `src/main.py`:

| Constant | Default | Description |
|---|---|---|
| `LIKE_WEIGHT` | `1.0` | Weight of a like in taste clustering. |
| `TOREAD_WEIGHT` | `0.5` | Weight of a to-read book (half a like). |
| `DISLIKE_PENALTY` | `0.5` | How strongly dislikes repel candidates. |
| `SEEN_TTL_DAYS` | `30` | Days before a seen book becomes eligible again. |
| `EXPLORATION_RATE` | `0.15` | Fraction of each batch that is random exploration. |
| `SERIES_BOOST` | `0.15` | Relevance boost for "next in series." |
| `DESCRIPTION_MAX_WORDS` | `120` | Description truncation for embeddings. |
| `TITLE_REPEAT` | `2` | Title repetitions in embedding input. |
| `TAGS_REPEAT` | `2` | Tag repetitions in embedding input. |
| `EMBED_VERSION` | `3` | Bump to force an index rebuild after embedding changes. |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | *(empty)* | Ollama API base URL. Empty disables LLM reasons. |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name for reason generation. |

## Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `/mnt/user/Books` | `/calibre` | Calibre library (read-only) |
| `/mnt/user/appdata/bookrec` | `/config` | State, feedback, generated index |

## Permissions

The container runs as an unprivileged user (`UID 99`, `GID 1000`). The `/config`
volume must be writable by that user, otherwise the app cannot create
`bookrec.db` and will fail to start. On Unraid this usually works out of the box
(`nobody:users`), but if you see permission errors, fix the ownership of the
host directory:

```bash
chown -R 99:1000 /mnt/user/appdata/bookrec
```

## First boot

The first start downloads the embedding model (~90 MB) and indexes your entire
Calibre library, which can take several minutes on a large collection. The
healthcheck has an extended `start_period` to allow for this; the app is ready
once `http://your-host:8484/api/stats` returns `200`.

The index is automatically rebuilt when:
- The Calibre `metadata.db` is modified (detected via mtime).
- `EMBED_VERSION` or `MODEL_NAME` changes (detected via a sidecar `index_meta.json`).

This means embedding or model changes take effect on next boot — no manual
`index.faiss` deletion needed.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI (table view). |
| `/api/recommendations?count=N` | GET | Batch of up to N recommendations (default 10, max 50). |
| `/api/recommend` | GET | Single recommendation (legacy endpoint). |
| `/api/more-like/{book_id}?count=N` | GET | One-shot "more like this" for a specific book. |
| `/api/feedback` | POST | Record feedback (`like`, `dislike`, `skip`, `toread`). |
| `/api/reset-seen` | POST | Clear seen history. |
| `/api/stats` | GET | Library stats (total, liked, disliked, seen, toread). |
| `/api/rebuild` | GET | Force an index rebuild. |
| `/cover/{book_id}/{filename}` | GET | Serve a book cover image. |

## Development

### Tests

```bash
pip install -r requirements-dev.txt
pip install numpy
python -m pytest tests/ -v
```

Tests use a `FakeState`/`FakeIndex` that mimics FAISS with numpy — no model,
no DB, no Ollama needed. CI runs tests automatically via `.github/workflows/tests.yml`.

### Project structure

```
src/
  main.py              # FastAPI app + all recommendation logic
  templates/
    index.html         # Single-page web UI
tests/
  test_recommend.py    # Unit tests for pure functions
Dockerfile
docker-compose.yml
requirements.txt       # Runtime dependencies
requirements-dev.txt   # Test dependencies (pytest)
.github/workflows/
  docker.yml           # Build & push image to GHCR
  tests.yml            # Run pytest on push/PR
```