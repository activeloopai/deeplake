import os

import pytest

from deeplake.integrations.twelvelabs import (
    MarengoEmbedder,
    MARENGO_EMBEDDING_DIM,
)

requires_api_key = pytest.mark.skipif(
    not os.environ.get("TWELVELABS_API_KEY"),
    reason="TWELVELABS_API_KEY not set",
)


def test_missing_api_key_raises(monkeypatch):
    # No-network: constructor must fail clearly when no key is available.
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TwelveLabs API key"):
        MarengoEmbedder()


def test_first_segment_vector_empty_raises():
    # No-network: empty / errored embedding responses surface a clear error.
    class _Empty:
        segments = []
        error_message = "boom"

    with pytest.raises(ValueError, match="no embedding segments"):
        MarengoEmbedder._first_segment_vector(_Empty())

    with pytest.raises(ValueError, match="no embedding segments"):
        MarengoEmbedder._first_segment_vector(None)


@requires_api_key
def test_embed_text_returns_512_dim_vectors():
    embedder = MarengoEmbedder()
    assert embedder.dim == MARENGO_EMBEDDING_DIM == 512

    vectors = embedder.embed_text("a chef plating a dish in a busy kitchen")
    assert len(vectors) == 1
    assert len(vectors[0]) == 512
    assert all(isinstance(x, float) for x in vectors[0])

    # Accepts a list and preserves order / count.
    batch = embedder.embed_text(["a dog running", "city traffic at night"])
    assert len(batch) == 2
    assert all(len(v) == 512 for v in batch)
