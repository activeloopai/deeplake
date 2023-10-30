import pytest

from deeplake.core.vectorstore.managed_vectorstore import ManagedVectorStore


def test_unsupported_parameters():
    with pytest.raises(NotImplementedError):
        ManagedVectorStore(
            path="hub://test/test",
            embedding_function=lambda x: x,
        )
