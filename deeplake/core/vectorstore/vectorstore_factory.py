from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deepmemory_vectorstore import DeepMemoryVectorStore


def vectorstore_factory(
    *args,
    **kwargs,
):
    if kwargs.get("runtime") == {"tensor_db": True}:
        return DeepMemoryVectorStore(*args, **kwargs)
    return VectorStore(*args, **kwargs)
