from typing import List, Union, Generic, TypeVar
from deeplake.deeplog.actions import (
    DeepLogAction,
    AddFileAction,
    CreateBranchAction,
    ProtocolAction,
    MetadataAction,
    CreateTensorAction,
    CreateCommitAction,
)
from deeplake.deeplog.adapters import to_commit_id
from functools import wraps
import deeplake


def atomic(func):
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        assert isinstance(self, deeplake.Tensor)
        chunk_engine = self.chunk_engine
        storage = chunk_engine.base_storage
        if storage.deeplog.log_format() >= 4 and storage._staged_transaction is None:
            root_call = True
        else:
            root_call = False
        func(*args, **kwargs)
        if root_call:
            chunk_engine.cache.flush()
            staged_transaction = storage._staged_transaction
            if staged_transaction:
                staged_transaction.commit()
                storage._staged_transaction = None
                branch_id = staged_transaction.snapshot.branch_id
                self.version_state["commit_id"] = to_commit_id(
                    branch_id, storage.deeplog.version(branch_id)
                )

    return inner
