from abc import ABC


class DeepLogAction(ABC):
    pass


class AddFileAction(DeepLogAction):
    def __init__(self, path: str):
        self.path = path
        self.size = 5123
        self.modification_time = 123123123
        self.data_change = True


class CreateBranchAction(DeepLogAction):
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.from_branch = ""
        self.from_version = 31


class CreateCommitAction(DeepLogAction):
    def __init__(
        self,
        id: str,
        branch_id: str,
        branch_version: int,
        message: str,
        commit_time: int,
    ):
        self.id = id
        self.branch_id = branch_id
        self.branch_version = branch_version
        self.message = message
        self.commit_time = commit_time


class MetadataAction(DeepLogAction):
    def __init__(self, id: str, name: str, description: str):
        self.id = id
        self.name = name
        self.description = description


class ProtocolAction(DeepLogAction):
    def __init__(self):
        self.min_reader_version = 4
        self.min_writer_version = 4


class CreateTensorAction(DeepLogAction):
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.chunk_compression = None
        self.dtype = "str"
        self.hidden = False
        self.htype = "text"
        self.is_link = False
        self.is_sequence = False
        self.length = 4
        self.links = {
            "81123-889g81-455626": {
                "extend": "extend_id",
                "flatten_sequence": False,
            }
        }
        self.max_chunk_size = None
        self.max_shape = [1]
        self.min_shape = [1]
        self.sample_compression = None
        self.tiling_threshold = None
        self.typestr = None
        self.verify = True
        self.version = "3.6.26"
