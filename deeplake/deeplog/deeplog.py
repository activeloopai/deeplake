from typing import List, Union, Generic, TypeVar, Type

from deeplake.deeplog.actions import (
    DeepLogAction,
    AddFileAction,
    CreateBranchAction,
    ProtocolAction,
    MetadataAction,
    CreateTensorAction,
    CreateCommitAction,
)

DataType = TypeVar("DataType", bound=Union[DeepLogAction, List[DeepLogAction]])


class DeeplogState(Generic[DataType]):
    def __init__(self, action: DataType):
        self.action = action

    def version(self) -> str:
        return 5

    def data(self) -> DataType:
        return self.action


class DeepLog:
    def __init__(self, path: str, storage: "deeplake.core.StorageProvider"):
        self._path = path
        self._storage = storage

    def log_format(self) -> int:
        return 4

    def version(self, branch_id: str) -> int:
        return 31

    def path(self) -> str:
        return self._path

    def protocol(self, version: int = -1) -> DeeplogState:
        """
        Return the protocol at the given version. If version == -1, use the latest version
        """
        return DeeplogState(ProtocolAction())

    def metadata(self, version: int = -1) -> DeeplogState:
        return DeeplogState(MetadataAction())

    def branches(self, version: int = -1) -> DeeplogState[List[CreateBranchAction]]:
        return DeeplogState(
            [
                CreateBranchAction("", "main"),
                CreateBranchAction("8331-351-3313", "other_branch"),
            ]
        )

    def commits(self, version: int = -1) -> DeeplogState[List[CreateCommitAction]]:
        return DeeplogState(
            [
                CreateCommitAction("ghwdkghasdgasg", "", 3, "first commit", 123123123),
                CreateCommitAction("gjjdiasdg", "", 5, "second commit", 123123125),
                CreateCommitAction(
                    "ghwdkghasdgasg",
                    "8331-351-3313",
                    7,
                    "first other commit",
                    123123126,
                ),
            ]
        )

    def data_files(
        self, branch_id: str, version: int = -1
    ) -> DeeplogState[List[AddFileAction]]:
        return DeeplogState(
            [
                AddFileAction("my/path.txt"),
                AddFileAction("other/path.txt"),
            ]
        )

    def tensors(
        self, branch_id: str, version: int = -1
    ) -> DeeplogState[List[CreateTensorAction]]:
        return DeeplogState(
            [
                CreateTensorAction("858123-889g81-43626", "text"),
                CreateTensorAction("81123-889g81-455626", "_text_id"),
                # CreateTensorAction("88623-gasdg-398gasdg", "other_tensor"),
            ]
        )


class DeepLogV3(DeepLog):
    def __init__(self, path: str, storage: "deeplake.core.StorageProvider"):
        super().__init__(path, storage)

    def log_format(self) -> int:
        return 3

    def version(self) -> int:
        raise NotImplementedError()

    def protocol(self, version: int = -1) -> DeeplogState:
        raise NotImplementedError()

    def metadata(self, version: int = -1) -> DeeplogState:
        raise NotImplementedError()

    def branches(self, version: int = -1) -> DeeplogState[List[CreateBranchAction]]:
        raise NotImplementedError()

    def data_files(self, version: int = -1) -> DeeplogState[List[AddFileAction]]:
        raise NotImplementedError()


def open_deeplog(path: str, storage: "deeplake.core.StorageProvider") -> DeepLog:
    if "_deeplake_log/00000000000000000000.json" in storage:
        return DeepLog(path, storage)
    else:
        return DeepLogV3(path, storage)
