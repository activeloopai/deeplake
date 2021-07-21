from abc import ABC, abstractmethod
from hub.util.path import find_root
from pathlib import Path


class UnstructuredDataset(ABC):
    def __init__(self, source: str):
        self.source = Path(find_root(source))

    @abstractmethod
    def structure(ds, use_progress_bar: bool = True, **kwargs):
        pass
