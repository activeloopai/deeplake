import numpy as np
import pytest

import deeplake
from deeplake import VectorStore
from deeplake.core.vectorstore import utils


# 1. init tests:
def test_managed_vectorstore_should_not_accept_dataset_during_init(
    hub_cloud_path, hub_cloud_dev_token
):
    with pytest.raises(NotImplementedError):
        VectorStore(
            dataset=deeplake.empty(hub_cloud_path),
            token=hub_cloud_dev_token,
            runtime={"tensor_db": True},
        )


# def test_managed_vectorstore_should_not_accept_embedding_function_during_init(
#     hub_cloud_path, hub_cloud_dev_token
# ):
#     with pytest.raises(NotImplementedError):
#         VectorStore(
#             path=hub_cloud_path,
#             token=hub_cloud_dev_token,
#             runtime={"tensor_db": True},
#             embedding_function=lambda x: x,
#         )


def test_managed_vectorstore_should_not_accept_exec_option_during_init(
    hub_cloud_path, hub_cloud_dev_token
):
    with pytest.raises(NotImplementedError):
        VectorStore(
            path=hub_cloud_path,
            exec_option="conpute_engine",
            token=hub_cloud_dev_token,
            runtime={"tensor_db": True},
        )


def test_managed_vectorstore_should_not_accept_creds_during_init(
    hub_cloud_path, hub_cloud_dev_token
):
    with pytest.raises(NotImplementedError):
        VectorStore(
            path=hub_cloud_path,
            token=hub_cloud_dev_token,
            creds={"random_creds": "random_creds"},
            runtime={"tensor_db": True},
        )


def test_managed_vectorstore_should_not_accept_org_id_during_init(
    hub_cloud_path, hub_cloud_dev_token
):
    with pytest.raises(NotImplementedError):
        VectorStore(
            path=hub_cloud_path,
            token=hub_cloud_dev_token,
            org_id="random_org_id",
            runtime={"tensor_db": True},
        )


def test_managed_vectorstore_should_not_accept_kwargs_during_init(
    hub_cloud_path, hub_cloud_dev_token
):
    with pytest.raises(NotImplementedError):
        VectorStore(
            path=hub_cloud_path,
            token=hub_cloud_dev_token,
            some_argument="some_argument",
            runtime={"tensor_db": True},
        )


def test_managed_vectorstore_should_not_accept_embedding_data_during_add(
    hub_cloud_path, hub_cloud_dev_token
):
    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )

    with pytest.raises(NotImplementedError):
        db.add(
            text=["a", "b", "c"],
            metadata=[{}, {}, {}],
            embedding_function=lambda x: x,
        )


def test_managed_vectorstore_should_not_accept_embedding_tensor_during_add(
    hub_cloud_path, hub_cloud_dev_token
):
    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )

    with pytest.raises(NotImplementedError):
        db.add(
            text=["a", "b", "c"],
            metadata=[{}, {}, {}],
            embedding_tensor="embedding_tensor",
        )


def test_managed_vectorstore_should_not_accept_rate_limiter_during_add(
    hub_cloud_path, hub_cloud_dev_token
):
    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )

    with pytest.raises(NotImplementedError):
        db.add(
            text=["a", "b", "c"],
            metadata=[{}, {}, {}],
            rate_limiter={"enabled": True, "bytes_per_minute": 1000000},
        )


# 3. search tests:
def test_managed_vectorstore_should_not_accept_embedding_function_during_search(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=100,
    )

    with pytest.raises(NotImplementedError):
        db.search(
            embedding=np.zeros(100, dtype=np.float32),
            embedding_function=lambda x: x,
        )


def test_managed_vectorstore_should_not_accept_embedding_data_during_search(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=100,
    )

    with pytest.raises(NotImplementedError):
        db.search(
            embedding=np.zeros(100, dtype=np.float32),
            embedding_data="text",
        )


# def test_managed_vectorstore_should_not_accept_embedding_tensor_during_search(
#     hub_cloud_path, hub_cloud_dev_token
# ):
#     db = utils.create_and_populate_vs(
#         path=hub_cloud_path,
#         token=hub_cloud_dev_token,
#         runtime={"tensor_db": True},
#         embedding_dim=100,
#     )

#     with pytest.raises(NotImplementedError):
#         db.search(
#             embedding=np.zeros(100, dtype=np.float32),
#             embedding_tensor="embedding",
#         )


def test_managed_vectorstore_should_not_accept_return_view_during_search(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=100,
    )

    with pytest.raises(NotImplementedError):
        db.search(
            embedding=np.zeros(100, dtype=np.float32),
            return_view=True,
        )


def test_managed_vectorstore_should_not_accept_exec_option_during_delete(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=100,
    )

    with pytest.raises(NotImplementedError):
        db.search(
            embedding=np.zeros(100, dtype=np.float32),
            exec_option="compute_engine",
        )


def test_managed_vectorstore_should_not_accept_exec_option_during_update_embedding(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=3,
    )

    embedding_dict = {"embedding": [[0]] * 3}

    with pytest.raises(NotImplementedError):
        db.update_embedding(
            embedding_dict=embedding_dict,
            embedding_source_tensor="text",
            embedding_tensor="embedding",
        )


def test_managed_vectorstore_should_not_accept_embedding_function_during_update_embedding(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=3,
    )

    embedding_dict = {"embedding": [[0, 0, 0]] * 3}

    with pytest.raises(NotImplementedError):
        db.update_embedding(
            embedding_dict=embedding_dict,
            embedding_function=lambda x: x,
        )


def test_managed_vectorstore_should_not_accept_force_during_delete_by_path(
    hub_cloud_path, hub_cloud_dev_token
):
    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )

    with pytest.raises(NotImplementedError):
        db.delete_by_path(path=hub_cloud_path, force=True, runtime={"tensor_db": True})


def test_managed_vectorstore_should_not_accept_creds_during_delete_by_path(
    hub_cloud_path, hub_cloud_dev_token
):
    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )

    with pytest.raises(NotImplementedError):
        db.delete_by_path(
            path=hub_cloud_path,
            creds={"creds": "Non existing creds"},
            runtime={"tensor_db": True},
        )


def test_managed_vectorstore_should_not_accept_commit(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=100,
    )

    with pytest.raises(NotImplementedError):
        db.commit()


def test_managed_vectorstore_should_not_accept_checkout(
    hub_cloud_path, hub_cloud_dev_token
):
    db = utils.create_and_populate_vs(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
        embedding_dim=100,
    )

    with pytest.raises(NotImplementedError):
        db.checkout("new_branch")


def test_managed_vectorstore_should_not_return_dataset(
    hub_cloud_path, hub_cloud_dev_token
):
    # this will be supported in the phase 2
    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )

    with pytest.raises(NotImplementedError):
        dataset = db.dataset
