import deeplake
from typing import Dict, List, Optional, Tuple
from deeplake.core.vectorstore.vector_search.utils import create_data


def create_and_load_vectorstore():
    from deeplake import VectorStore

    db = VectorStore(
        path="local_path",
        overwrite=True,
    )

    texts, embeddings, ids, metadata, _ = create_data(
        number_of_data=100, embedding_dim=1536, metadata_key="abc"
    )
    db.add(
        text=texts,
        embedding=embeddings,
        id=ids,
        metadata=metadata,
    )
    return db


def train_deepmemory_model(
    dataset_name: str = f"hub://activeloop-test/scifact",
    corpus: Optional[Dict] = None,
    relevenace: Optional[List[List[Tuple[str, int]]]] = None,
    queries: Optional[List[str]] = None,
    token: Optional[str] = None,
    overwrite: bool = False,
    enviroment: str = "staging",
):
    from deeplake import VectorStore
    from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore

    if enviroment == "staging":
        deeplake.client.config.USE_STAGING_ENVIRONMENT = True
    elif enviroment == "dev":
        deeplake.client.config.USE_DEV_ENVIRONMENT = True

    embedding_function = OpenAIEmbeddings()
    if corpus is None:
        if (
            not deeplake.exists(dataset_name, token=token, creds={})
            or overwrite == True
        ):
            deeplake.deepcopy(
                f"hub://activeloop-test/deepmemory_test_corpus",
                dataset_name,
                token=token,
                overwrite=True,
                runtime={"tensor_db": True},
            )

        db = VectorStore(
            dataset_name,
            token=token,
            embedding_function=embedding_function,
        )
    else:
        db = VectorStore(
            dataset_name,
            token=token,
            overwrite=True,
            embedding_function=embedding_function,
        )
        db.add(**corpus)

    query_vs = None

    if relevenace is None:
        query_vs = VectorStore(
            path=f"hub://activeloop-test/deepmemory_test_queries",
            runtime={"tensor_db": True},
            token=token,
        )
        relevance = query_vs.dataset.metadata.data()["value"]

    if queries is None:
        if not query_vs:
            query_vs = VectorStore(
                path=f"hub://activeloop-test/deepmemory_test_queries",
                runtime={"tensor_db": True},
                token=token,
            )
        queries = query_vs.dataset.text.data()["value"]

    db.deep_memory.train(
        relevance=relevance,
        queries=queries,
    )
    return db


def set_backend(backend="prod"):
    if backend == "staging":
        deeplake.client.config.USE_STAGING_ENVIRONMENT = True
        deeplake.client.config.USE_DEV_ENVIRONMENT = False
    elif backend == "dev":
        deeplake.client.config.USE_DEV_ENVIRONMENT = True
        deeplake.client.config.USE_STAGING_ENVIRONMENT = False
    else:
        deeplake.client.config.USE_DEV_ENVIRONMENT = False
        deeplake.client.config.USE_STAGING_ENVIRONMENT = False
