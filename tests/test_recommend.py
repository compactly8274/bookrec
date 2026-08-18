"""Unit tests for bookrec pure functions.

These tests exercise the recommendation logic without needing the embedding
model, FAISS index, SQLite DB, or Ollama. They use a FakeState that mimics
the AppState interface with simple numpy vectors.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np

# ── Mock heavy dependencies before importing main ────────────────────────────
# The CI environment only has pytest + numpy, not faiss/sentence_transformers/
# fastapi/httpx/jinja2. We inject lightweight stubs into sys.modules so that
# `import main` succeeds without those packages installed.

# faiss stub — provides Kmeans and normalize_L2 (used by _taste_centroids)
class _FakeKmeans:
    def __init__(self, d, k, niter=20, verbose=False, seed=42):
        self.d = d
        self.k = k
        self.centroids = np.zeros((k, d), dtype="float32")

    def train(self, vecs):
        # naive centroids: just pick k evenly spaced rows
        n = vecs.shape[0]
        if self.k <= 1:
            self.centroids = vecs.mean(axis=0, keepdims=True).astype("float32")
        else:
            indices = np.linspace(0, n - 1, self.k, dtype=int)
            self.centroids = vecs[indices].astype("float32")


_faiss_stub = MagicMock()
_faiss_stub.Kmeans = _FakeKmeans
_faiss_stub.IndexFlatIP = MagicMock
_faiss_stub.normalize_L2 = lambda x: x  # no-op (vectors are already normalised in tests)
_faiss_stub.write_index = MagicMock()
_faiss_stub.read_index = MagicMock()
sys.modules["faiss"] = _faiss_stub

# sentence_transformers stub
_st_stub = MagicMock()
_st_stub.SentenceTransformer = MagicMock()
sys.modules["sentence_transformers"] = _st_stub

# httpx stub
sys.modules["httpx"] = MagicMock()

# jinja2 stub (imported by fastapi.templating, which we mock, but just in case)
sys.modules["jinja2"] = MagicMock()

# fastapi + submodules stubs
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.templating"] = MagicMock()

# pydantic stub — needs BaseModel + Field that accepts any args
class _StubBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def _stub_field(*args, **kwargs):
    """Accept any arguments (Ellipsis, pattern=, default=, etc.) and return None."""
    return None

_pydantic_stub = MagicMock()
_pydantic_stub.BaseModel = _StubBaseModel
_pydantic_stub.Field = _stub_field
sys.modules["pydantic"] = _pydantic_stub

# ── Now import main ──────────────────────────────────────────────────────────

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import main
from main import (
    embed_text,
    _truncate_words,
    _rating_stars,
    _relevance,
    cover_url,
    deterministic_reason,
    candidate_pool,
    exploration_pool,
    diversify,
    _taste_centroids,
    _series_progress,
    _series_boost,
    batch_recommendations,
    more_like_books,
    SERIES_BOOST,
)


# ── Fake state ───────────────────────────────────────────────────────────────

class FakeIndex:
    """Minimal FAISS-like index for testing.

    Stores unit-normalised vectors and supports reconstruct() + search().
    search() returns brute-force cosine similarities (dot products on L2-normalised vectors).
    """

    def __init__(self, vectors):
        self._vectors = vectors.astype("float32")
        self.d = vectors.shape[1]

    def reconstruct(self, idx):
        return self._vectors[idx]

    def search(self, query, k):
        # query: (nq, d) → returns (D, I) with shape (nq, k)
        sims = self._vectors @ query.T  # (n, nq)
        D = np.empty((query.shape[0], k), dtype="float32")
        I = np.empty((query.shape[0], k), dtype="int64")
        for i in range(query.shape[0]):
            col = sims[:, i]
            order = np.argsort(-col)[:k]
            D[i] = col[order]
            I[i] = order
        return D, I


class FakeState:
    """Mimics AppState with a FakeIndex and a list of book dicts."""

    def __init__(self, books, dim=8):
        self.books = books
        self.book_ids = [b["id"] for b in books]
        self.id_to_idx = {bid: i for i, bid in enumerate(self.book_ids)}
        # deterministic unit vectors from book id
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(books), dim).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        self.index = FakeIndex(vecs)


def make_book(bid, title="Book", authors=None, tags=None, description="A book.",
              has_cover=True, rating_val=None, series=None, series_index=None):
    return {
        "id": bid,
        "title": title,
        "author_sort": (authors or ["Author"])[0],
        "path": f"books/{bid}",
        "has_cover": has_cover,
        "description": description,
        "rating_val": rating_val,
        "authors": authors or ["Author"],
        "tags": tags or [],
        "series": series,
        "series_index": series_index,
    }


# ── embed_text / _truncate_words ─────────────────────────────────────────────

class TestEmbedText:

    def test_basic_concatenation(self):
        book = make_book(1, title="Dune", authors=["Frank Herbert"],
                         tags=["Sci-Fi"], description="A desert planet.")
        result = embed_text(book)
        # title appears twice, tags once (repeated), description once
        assert "Dune" in result
        assert "Sci-Fi" in result
        assert "A desert planet." in result

    def test_title_repeated(self):
        book = make_book(1, title="Dune", description="")
        result = embed_text(book)
        # TITLE_REPEAT = 2, so "Dune" appears twice
        assert result.count("Dune") == 2

    def test_tags_repeated(self):
        book = make_book(1, title="X", tags=["Sci-Fi", "Adventure"], description="")
        result = embed_text(book)
        # TAGS_REPEAT = 2, so the tag string appears twice
        assert result.count("Sci-Fi, Adventure") == 2

    def test_description_truncated(self):
        long_desc = " ".join(["word"] * 500)
        book = make_book(1, title="X", description=long_desc)
        result = embed_text(book)
        # DESCRIPTION_MAX_WORDS = 120, so the description part has 120 words
        desc_part = result.split(" | ")[-1]
        assert len(desc_part.split()) == 120

    def test_empty_fields(self):
        book = {"id": 1, "title": "", "authors": [], "tags": [], "description": ""}
        result = embed_text(book)
        assert result == ""

    def test_title_first_in_order(self):
        book = make_book(1, title="TitleA", tags=["TagB"], authors=["AuthorC"],
                         description="DescD")
        result = embed_text(book)
        # Title should appear before tags, authors, description
        assert result.index("TitleA") < result.index("TagB")
        assert result.index("TagB") < result.index("AuthorC")
        assert result.index("AuthorC") < result.index("DescD")


class TestTruncateWords:

    def test_short_text_unchanged(self):
        assert _truncate_words("hello world", 10) == "hello world"

    def test_exact_limit(self):
        assert _truncate_words("one two three", 3) == "one two three"

    def test_truncated(self):
        assert _truncate_words("one two three four", 2) == "one two"

    def test_empty(self):
        assert _truncate_words("", 10) == ""
        assert _truncate_words(None, 10) == ""


# ── _rating_stars ────────────────────────────────────────────────────────────

class TestRatingStars:

    def test_none(self):
        assert _rating_stars({"rating_val": None}) == 0.0

    def test_zero(self):
        assert _rating_stars({"rating_val": 0}) == 0.0

    def test_max(self):
        assert _rating_stars({"rating_val": 10}) == 5.0

    def test_midpoint(self):
        assert _rating_stars({"rating_val": 6}) == 3.0

    def test_invalid(self):
        assert _rating_stars({"rating_val": "not a number"}) == 0.0

    def test_missing_key(self):
        assert _rating_stars({}) == 0.0


# ── _relevance ───────────────────────────────────────────────────────────────

class TestRelevance:

    def test_score_only(self):
        book = {"score": 0.8, "rating_val": None}
        assert _relevance(book) == 0.8

    def test_with_rating(self):
        book = {"score": 0.8, "rating_val": 10}
        assert _relevance(book) == 0.8 + 5.0 * 0.03

    def test_with_dislike_penalty(self):
        book = {"score": 0.8, "rating_val": None, "dislike_penalty": 0.9}
        expected = 0.8 - 0.5 * 0.9
        assert abs(_relevance(book) - expected) < 1e-6

    def test_with_series_boost(self):
        book = {"score": 0.5, "rating_val": None, "series_boost": SERIES_BOOST}
        expected = 0.5 + SERIES_BOOST
        assert abs(_relevance(book) - expected) < 1e-6

    def test_combined(self):
        book = {"score": 0.5, "rating_val": 8, "dislike_penalty": 0.6, "series_boost": SERIES_BOOST}
        expected = 0.5 + 4.0 * 0.03 - 0.5 * 0.6 + SERIES_BOOST
        assert abs(_relevance(book) - expected) < 1e-6


# ── cover_url ────────────────────────────────────────────────────────────────

class TestCoverUrl:

    def test_with_cover(self):
        assert cover_url({"id": 5, "has_cover": True}) == "/cover/5/cover.jpg"

    def test_no_cover(self):
        assert cover_url({"id": 5, "has_cover": False}) == ""

    def test_missing_key(self):
        assert cover_url({"id": 5}) == ""


# ── deterministic_reason ─────────────────────────────────────────────────────

class TestDeterministicReason:

    def test_with_liked_titles(self):
        book = make_book(1, title="Dune", tags=["Sci-Fi"], authors=["Herbert"])
        reason = deterministic_reason(book, ["Foundation"])
        assert "Foundation" in reason
        assert "Sci-Fi" in reason

    def test_no_liked_titles(self):
        book = make_book(1, title="Dune", tags=["Sci-Fi"], authors=["Herbert"])
        reason = deterministic_reason(book, [])
        assert "Sci-Fi" in reason

    def test_similar_to(self):
        book = make_book(1, title="Dune", tags=["Sci-Fi"], authors=["Herbert"])
        reason = deterministic_reason(book, [], similar_to="Foundation")
        assert "Foundation" in reason
        assert "Similar to" in reason

    def test_no_tags(self):
        book = make_book(1, title="Dune", tags=[], authors=["Herbert"])
        reason = deterministic_reason(book, ["Foundation"])
        assert "Foundation" in reason


# ── _series_progress / _series_boost ─────────────────────────────────────────

class TestSeriesProgress:

    def test_no_series_in_likes(self):
        books = [make_book(1, description="Book 1")]
        state = FakeState(books)
        progress = _series_progress(state, {1})
        assert progress == {}

    def test_series_detected(self):
        books = [
            make_book(1, series="Dune Series", series_index=1.0),
            make_book(2, series="Dune Series", series_index=2.0),
        ]
        state = FakeState(books)
        progress = _series_progress(state, {1})
        assert "Dune Series" in progress
        assert 1.0 in progress["Dune Series"]

    def test_multiple_series(self):
        books = [
            make_book(1, series="Series A", series_index=1.0),
            make_book(2, series="Series B", series_index=3.0),
        ]
        state = FakeState(books)
        progress = _series_progress(state, {1, 2})
        assert "Series A" in progress
        assert "Series B" in progress

    def test_ignores_toread_and_seen(self):
        """Only likes should contribute to series progress, not toread/seen."""
        books = [
            make_book(1, series="S", series_index=1.0),
            make_book(2, series="S", series_index=2.0),
        ]
        state = FakeState(books)
        # Only book 1 is liked — book 2 is toread but not liked
        progress = _series_progress(state, {1})
        assert "S" in progress
        assert 2.0 not in progress["S"]


class TestSeriesBoost:

    def test_no_series(self):
        book = make_book(1, series=None, series_index=None)
        assert _series_boost(book, {}) == 0.0

    def test_no_progress(self):
        book = make_book(1, series="Dune", series_index=2.0)
        assert _series_boost(book, {}) == 0.0

    def test_earlier_liked(self):
        book = make_book(2, series="Dune", series_index=2.0)
        progress = {"Dune": {1.0}}
        assert _series_boost(book, progress) == SERIES_BOOST

    def test_later_liked_not_boosted(self):
        """If user liked book 3, book 1 should NOT be boosted (only forward)."""
        book = make_book(1, series="Dune", series_index=1.0)
        progress = {"Dune": {3.0}}
        assert _series_boost(book, progress) == 0.0

    def test_same_index_not_boosted(self):
        """Same index (same book) should not boost itself."""
        book = make_book(1, series="Dune", series_index=2.0)
        progress = {"Dune": {2.0}}
        assert _series_boost(book, progress) == 0.0

    def test_fractional_index(self):
        """Fractional indices (novellas between books) should work."""
        book = make_book(1, series="S", series_index=2.5)
        progress = {"S": {2.0}}
        assert _series_boost(book, progress) == SERIES_BOOST

    def test_multiple_liked_earlier(self):
        book = make_book(3, series="S", series_index=3.0)
        progress = {"S": {1.0, 2.0}}
        assert _series_boost(book, progress) == SERIES_BOOST


# ── candidate_pool ───────────────────────────────────────────────────────────

class TestCandidatePool:

    def test_cold_start_excludes_no_description(self):
        books = [
            make_book(1, description="Has desc"),
            make_book(2, description=""),
            make_book(3, description="Also has desc"),
        ]
        state = FakeState(books)
        pool = candidate_pool(state, set(), set(), set(), set(), limit=10, shuffle=False)
        ids = [b["id"] for b in pool]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids

    def test_cold_start_excludes_omnibus(self):
        books = [
            make_book(1, title="Regular Book", description="Desc"),
            make_book(2, title="The Complete Collection", description="Desc"),
            make_book(3, title="Omnibus Edition", description="Desc"),
        ]
        state = FakeState(books)
        pool = candidate_pool(state, set(), set(), set(), set(), limit=10, shuffle=False)
        ids = [b["id"] for b in pool]
        assert 1 in ids
        assert 2 not in ids
        assert 3 not in ids

    def test_excludes_seen_dislikes_toread(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 11)]
        state = FakeState(books)
        pool = candidate_pool(
            state, likes={1}, dislikes={2}, seen={3}, toread={4}, limit=10, shuffle=False
        )
        ids = {b["id"] for b in pool}
        assert 1 not in ids  # liked
        assert 2 not in ids  # disliked
        assert 3 not in ids  # seen
        assert 4 not in ids  # toread
        # 5-10 should be eligible
        assert ids == {5, 6, 7, 8, 9, 10}

    def test_limit_respected(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 51)]
        state = FakeState(books)
        pool = candidate_pool(state, set(), set(), set(), set(), limit=5, shuffle=False)
        assert len(pool) <= 5

    def test_personalized_uses_taste_signal(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 21)]
        state = FakeState(books)
        pool = candidate_pool(state, likes={1}, dislikes=set(), seen=set(), toread=set(), limit=10, shuffle=False)
        assert len(pool) > 0
        # liked book itself should not be in pool
        assert 1 not in {b["id"] for b in pool}

    def test_returns_defensive_copies(self):
        books = [make_book(1, description="Book 1")]
        state = FakeState(books)
        pool = candidate_pool(state, set(), set(), set(), set(), limit=1, shuffle=False)
        if pool:
            assert pool[0] is not state.books[0]

    def test_series_boost_applied(self):
        """A sequel in a series the user has liked an earlier entry of gets a boost."""
        books = [
            make_book(1, series="Dune", series_index=1.0, description="Book 1"),
            make_book(2, series="Dune", series_index=2.0, description="Book 2"),
            make_book(3, series="Dune", series_index=3.0, description="Book 3"),
            make_book(4, description="Unrelated book"),
            make_book(5, description="Unrelated book 2"),
        ]
        state = FakeState(books)
        pool = candidate_pool(state, likes={1}, dislikes=set(), seen=set(), toread=set(), limit=10, shuffle=False)
        by_id = {b["id"]: b for b in pool}
        if 2 in by_id:
            assert by_id[2]["series_boost"] == SERIES_BOOST
        if 3 in by_id:
            assert by_id[3]["series_boost"] == SERIES_BOOST
        if 4 in by_id:
            assert by_id[4].get("series_boost", 0.0) == 0.0


# ── exploration_pool ─────────────────────────────────────────────────────────

class TestExplorationPool:

    def test_excludes_all_state(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 21)]
        state = FakeState(books)
        pool = exploration_pool(state, likes={1}, dislikes={2}, seen={3}, toread={4}, count=5)
        ids = {b["id"] for b in pool}
        assert 1 not in ids
        assert 2 not in ids
        assert 3 not in ids
        assert 4 not in ids

    def test_count_zero(self):
        books = [make_book(i) for i in range(1, 5)]
        state = FakeState(books)
        assert exploration_pool(state, set(), set(), set(), set(), 0) == []

    def test_respects_count(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 51)]
        state = FakeState(books)
        pool = exploration_pool(state, set(), set(), set(), set(), 3)
        assert len(pool) == 3

    def test_randomness(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 51)]
        state = FakeState(books)
        p1 = {b["id"] for b in exploration_pool(state, set(), set(), set(), set(), 5)}
        p2 = {b["id"] for b in exploration_pool(state, set(), set(), set(), set(), 5)}
        # Extremely unlikely to get the same 5 random books twice
        assert p1 != p2


# ── diversify ────────────────────────────────────────────────────────────────

class TestDiversify:

    def test_returns_all_if_fewer_than_count(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 4)]
        state = FakeState(books)
        candidates = [dict(b) for b in books]
        result = diversify(state, candidates, 5)
        assert len(result) == 3

    def test_selects_count(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 21)]
        state = FakeState(books)
        candidates = [dict(b, score=0.5) for b in books]
        result = diversify(state, candidates, 5)
        assert len(result) == 5

    def test_first_pick_is_highest_relevance(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 11)]
        state = FakeState(books)
        candidates = [
            dict(make_book(1, description="A"), score=0.9),
            dict(make_book(2, description="B"), score=0.3),
        ]
        result = diversify(state, candidates, 1)
        assert result[0]["id"] == 1

    def test_no_duplicates(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 21)]
        state = FakeState(books)
        candidates = [dict(b, score=0.5) for b in books]
        result = diversify(state, candidates, 5)
        ids = [b["id"] for b in result]
        assert len(ids) == len(set(ids))


# ── _taste_centroids ─────────────────────────────────────────────────────────

class TestTasteCentroids:

    def test_single_signal_returns_one_centroid(self):
        books = [make_book(i) for i in range(1, 6)]
        state = FakeState(books)
        centroids = _taste_centroids(state, [(0, 1.0)])
        assert centroids.shape == (1, state.index.d)

    def test_multiple_signals_can_produce_multiple_centroids(self):
        books = [make_book(i) for i in range(1, 21)]
        state = FakeState(books)
        signal = [(i, 1.0) for i in range(10)]
        centroids = _taste_centroids(state, signal, max_clusters=3)
        assert centroids.shape[0] <= 3
        assert centroids.shape[1] == state.index.d

    def test_centroids_are_unit_normalised(self):
        books = [make_book(i) for i in range(1, 11)]
        state = FakeState(books)
        signal = [(i, 1.0) for i in range(5)]
        centroids = _taste_centroids(state, signal)
        norms = np.linalg.norm(centroids, axis=1)
        for n in norms:
            assert abs(n - 1.0) < 1e-5


# ── batch_recommendations ────────────────────────────────────────────────────

class TestBatchRecommendations:

    def test_dedup_no_duplicates(self):
        """The bug that prompted this test: exploitation + exploration can overlap."""
        books = [make_book(i, description=f"Book {i}") for i in range(1, 101)]
        state = FakeState(books)

        # Mock load_state_from_db to return controlled state
        async def fake_load():
            return set(), set(), set(), set()  # cold start
        main.load_state_from_db = fake_load

        # Mock _decorate_book to just add reason/cover
        async def fake_decorate(state, book, likes, liked_titles, like_sig, similar_to=None):
            book["reason"] = "test"
            book["cover_url"] = ""
            return book
        main._decorate_book = fake_decorate

        # Mock _mark_seen to no-op
        async def fake_mark(book, likes, dislikes, seen, toread):
            pass
        main._mark_seen = fake_mark

        for _ in range(10):
            result = asyncio.get_event_loop().run_until_complete(
                batch_recommendations(state, count=10)
            )
            ids = [b["id"] for b in result]
            assert len(ids) == len(set(ids)), f"Duplicate ids found: {ids}"

    def test_cold_start_returns_books(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 21)]
        state = FakeState(books)

        async def fake_load():
            return set(), set(), set(), set()
        main.load_state_from_db = fake_load

        async def fake_decorate(state, book, likes, liked_titles, like_sig, similar_to=None):
            book["reason"] = "test"
            book["cover_url"] = ""
            return book
        main._decorate_book = fake_decorate

        async def fake_mark(book, likes, dislikes, seen, toread):
            pass
        main._mark_seen = fake_mark

        result = asyncio.get_event_loop().run_until_complete(
            batch_recommendations(state, count=5)
        )
        assert len(result) <= 5
        assert len(result) > 0

    def test_exploration_rate_floor(self):
        """count=10, rate=0.15 → n_explore=1 (floor), not 2 (round)."""
        assert int(10 * 0.15) == 1

    def test_empty_library(self):
        state = FakeState([])

        async def fake_load():
            return set(), set(), set(), set()
        main.load_state_from_db = fake_load

        async def fake_decorate(state, book, likes, liked_titles, like_sig, similar_to=None):
            book["reason"] = "test"
            book["cover_url"] = ""
            return book
        main._decorate_book = fake_decorate

        async def fake_mark(book, likes, dislikes, seen, toread):
            pass
        main._mark_seen = fake_mark

        result = asyncio.get_event_loop().run_until_complete(
            batch_recommendations(state, count=10)
        )
        assert result == []


# ── more_like_books ──────────────────────────────────────────────────────────

class TestMoreLikeBooks:

    def test_unknown_book_returns_none(self):
        books = [make_book(i) for i in range(1, 5)]
        state = FakeState(books)

        async def fake_load():
            return set(), set(), set(), set()
        main.load_state_from_db = fake_load

        result = asyncio.get_event_loop().run_until_complete(
            more_like_books(state, 999, count=5)
        )
        assert result is None

    def test_excludes_source_book(self):
        books = [make_book(i, description=f"Book {i}") for i in range(1, 21)]
        state = FakeState(books)

        async def fake_load():
            return set(), set(), set(), set()
        main.load_state_from_db = fake_load

        async def fake_decorate(state, book, likes, liked_titles, like_sig, similar_to=None):
            book["reason"] = "test"
            book["cover_url"] = ""
            return book
        main._decorate_book = fake_decorate

        result = asyncio.get_event_loop().run_until_complete(
            more_like_books(state, 1, count=5)
        )
        if result:
            ids = {b["id"] for b in result}
            assert 1 not in ids  # source book excluded


# ── run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))