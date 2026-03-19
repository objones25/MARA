"""SentenceTransformer model loading with a module-level cache.

The model is heavy (~90 MB). Loading it once per process and caching it avoids
repeated disk I/O and GPU/CPU initialisation across scoring calls.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for model_name.

    Thread-safe for read access; the first call for a given model_name is
    not protected by a lock, which is acceptable for single-threaded and
    async-concurrent use (worst case: the model is loaded twice and one
    result is discarded).
    """
    if model_name not in _cache:
        _cache[model_name] = SentenceTransformer(model_name)
    return _cache[model_name]


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Return normalized embeddings as a numpy array of shape (len(texts), dim).

    Keeping the output as an ndarray (rather than converting to Python lists)
    allows callers to use np.dot() for cosine similarity, which is significantly
    faster than a Python-loop dot product for high-dimensional vectors.

    Args:
        texts:      Strings to embed.
        model_name: SentenceTransformer model identifier.

    Returns:
        float32 ndarray of shape (len(texts), embedding_dim), L2-normalised.
    """
    model = get_embedding_model(model_name)
    return model.encode(texts, normalize_embeddings=True)
