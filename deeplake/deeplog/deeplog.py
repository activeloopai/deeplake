from typing import List, Union

from deeplake.deeplog.actions import (
    DeepLogAction,
    AddFileAction,
    CreateBranchAction,
    ProtocolAction,
    MetadataAction,
    CreateTensorAction,
)


class DeeplogState:

    def __init__(self, action: Union[DeepLogAction, List[DeepLogAction]]):
        self.action = action

    def version(self) -> str:
        return 5

    def data(self) -> DeepLogAction:
        return self.action

class DeepLog:
    def version(self) -> int:
        return 31

    def path(self) -> str:
        return "path/to/deeplog"

    def protocol(self, version: int = -1) -> DeeplogState:
        """
        Return the protocol at the given version. If version == -1, use the latest version
        """
        return DeeplogState(ProtocolAction())

    def metadata(self, version: int = -1) -> DeeplogState:
        return DeeplogState(MetadataAction())

    def branches(self, version: int = -1) -> DeeplogState[List[CreateBranchAction]]:
        return DeeplogState([
            CreateBranchAction("", "main"),
            CreateBranchAction("8331-351-3313", "other_branch"),
        ])

    def data_files(self, version: int = -1) -> DeeplogState[List[AddFileAction]]:
        return DeeplogState([
            AddFileAction("my/path.txt"),
            AddFileAction("other/path.txt"),
        ])
    
    def tensors(self, version: int = -1) -> DeeplogState[List[CreateTensorAction]]:
        return DeeplogState([
            CreateTensorAction("my_tensor", "my_tensor"),
            CreateTensorAction("other_tensor", "other_tensor"),
        ])
