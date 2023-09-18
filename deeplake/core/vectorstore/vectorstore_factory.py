from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deep_memory import deep_memory_available
from deeplake.core.vectorstore.deepmemory_vectorstore import DeepMemoryVectorStore


def vectorstore_factory(
    *args,
    **kwargs,
):
    if kwargs["runtime"] == {"tensor_db": True} and deep_memory_available():
        return DeepMemoryVectorStore(*args, **kwargs)
    return VectorStore(*args, **kwargs)
