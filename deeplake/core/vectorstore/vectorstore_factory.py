from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deepmemory_vectorstore import DeepMemoryVectorStore
from deeplake.client.client import DeepMemoryBackendClient


def vectorstore_factory(
    *args,
    **kwargs,
):
    # dm_client = DeepMemoryBackendClient(token=kwargs.get("token"))
    # deepmemory_is_available = dm_client.deepmemory_is_available(kwargs.get("path")) # TODO: Discuss with Davit B
    deepmemory_is_available = True
    if kwargs.get("runtime") == {"tensor_db": True} and deepmemory_is_available:
        return DeepMemoryVectorStore(*args, **kwargs)
    return VectorStore(*args, **kwargs)
