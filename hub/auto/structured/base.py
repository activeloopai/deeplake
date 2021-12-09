from abc import ABC, abstractmethod
from hub.util.path import find_root
from pathlib import Path
import os


class StructuredDataset(ABC):
    def __init__(self, source: str):
        self.source = Path(source)
        if os.path.isdir(self.source):
            self.source = Path(find_root(self.source))

    """Initializes an structured dataset.
    
    Args:
        source (str): The local path to folder containing an unstructured dataset and of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
    """
