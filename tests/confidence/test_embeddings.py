"""Tests for mara.confidence.embeddings — model cache and embed().

These tests load the real sentence-transformers model. They are intentionally
kept fast by using short texts and a small batch size.
"""

import numpy as np
import pytest

from mara.confidence.embeddings import _cache, embed, get_embedding_model


MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # known dimension for all-MiniLM-L6-v2


# ---------------------------------------------------------------------------
# get_embedding_model — caching behaviour
# ---------------------------------------------------------------------------

class TestGetEmbeddingModel:
    def test_returns_model_instance(self):
        model = get_embedding_model(MODEL)
        assert model is not None

    def test_same_name_returns_same_object(self):
        """Cache must return the identical object on repeated calls."""
        m1 = get_embedding_model(MODEL)
        m2 = get_embedding_model(MODEL)
        assert m1 is m2

    def test_model_is_stored_in_cache(self):
        get_embedding_model(MODEL)
        assert MODEL in _cache
        assert _cache[MODEL] is get_embedding_model(MODEL)


# ---------------------------------------------------------------------------
# embed — return shape and dtype
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_returns_ndarray(self):
        result = embed(["hello world"], MODEL)
        assert isinstance(result, np.ndarray)

    def test_shape_single_text(self):
        result = embed(["hello world"], MODEL)
        assert result.shape == (1, EMBEDDING_DIM)

    def test_shape_multiple_texts(self):
        texts = ["hello", "world", "foo"]
        result = embed(texts, MODEL)
        assert result.shape == (3, EMBEDDING_DIM)

    def test_dtype_is_float32(self):
        """sentence-transformers encodes to float32 by default."""
        result = embed(["test"], MODEL)
        assert result.dtype == np.float32

    def test_embeddings_are_unit_norm(self):
        """normalize_embeddings=True must produce L2-unit vectors."""
        texts = ["The quick brown fox", "jumps over the lazy dog", "hello world"]
        embeddings = embed(texts, MODEL)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-5)

    def test_different_texts_produce_different_embeddings(self):
        e1 = embed(["The sky is blue"], MODEL)
        e2 = embed(["Quantum mechanics is complex"], MODEL)
        assert not np.allclose(e1, e2)

    def test_identical_texts_produce_identical_embeddings(self):
        text = "Identical input text"
        e1 = embed([text], MODEL)
        e2 = embed([text], MODEL)
        np.testing.assert_array_equal(e1, e2)

    def test_semantically_similar_texts_have_high_cosine_similarity(self):
        """Normalized embeddings: dot product == cosine similarity."""
        e = embed(["The cat sat on the mat", "A cat was sitting on the mat"], MODEL)
        similarity = float(np.dot(e[0], e[1]))
        assert similarity > 0.85  # expected for paraphrases with this model

    def test_semantically_dissimilar_texts_have_low_cosine_similarity(self):
        e = embed(["The cat sat on the mat", "Quantum field theory in curved spacetime"], MODEL)
        similarity = float(np.dot(e[0], e[1]))
        assert similarity < 0.5

    def test_batch_result_matches_individual_results(self):
        """Batch embedding must equal encoding each text individually."""
        texts = ["alpha", "beta", "gamma"]
        batch = embed(texts, MODEL)
        for i, text in enumerate(texts):
            individual = embed([text], MODEL)
            np.testing.assert_allclose(batch[i], individual[0], atol=1e-5)
