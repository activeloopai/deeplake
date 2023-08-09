from deeplake.core.vectorstore.vectorstores.vectorstore import VectorStore
from deeplake.core.vectorstore.vectorstores.managed_db_vectorstore import (
    ManagedDBVectorStore,
)
from deeplake.util.path import is_db_engine, is_hub_cloud_path


def vectorstore_factory(path, *args, **kwargs):
    token = kwargs.get("token")
    runtime = kwargs.get("runtime", {})

    empty, db_engine = is_db_engine(path, token, runtime)
    # create a new vectorstore in tensor_db:
    if db_engine:
        return ManagedDBVectorStore(path, empty=empty, *args, **kwargs)
    return VectorStore(path, *args, **kwargs)
