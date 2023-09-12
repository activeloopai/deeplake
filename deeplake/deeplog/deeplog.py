from typing import List

from deeplake.deeplog.actions import (
    AddFileAction,
    CreateBranchAction,
    ProtocolAction,
    MetadataAction,
)


class DeepLog:
    def version(self) -> int:
        return 31

    def path(self) -> str:
        return "path/to/deeplog"

    def protocol(self) -> ProtocolAction:
        return ProtocolAction()

    def metadata(self) -> MetadataAction:
        return MetadataAction()

    def branches(self) -> List[CreateBranchAction]:
        return [
            CreateBranchAction("", "main"),
            CreateBranchAction("8331-351-3313", "other_branch"),
        ]

    def data_files(self) -> List[AddFileAction]:
        return [
            AddFileAction("my/path.txt"),
            AddFileAction("other/path.txt"),
        ]
