from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deepmemory_vectorstore import DeepMemoryVectorStore
from deeplake.client.client import DeepMemoryBackendClient


def vectorstore_factory(
    *args,
    **kwargs,
):
    dm_client = DeepMemoryBackendClient(token=kwargs.get("token"))
    path = kwargs.get("path")
    dataset_id = path[6:].split("/")[0]
    # deepmemory_is_available = dm_client.deepmemory_is_available(dataset_id)
    deepmemory_is_available = True
    if kwargs.get("runtime") == {"tensor_db": True} and deepmemory_is_available:
        return DeepMemoryVectorStore(client=dm_client, *args, **kwargs)
    return VectorStore(*args, **kwargs)
