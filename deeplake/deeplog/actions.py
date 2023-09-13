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
    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name
