from abc import ABC, abstractmethod
from hub.util.path import find_root
from pathlib import Path
import os


class StructuredDataset(ABC):
    def __init__(self, source: str):
        if os.path.isdir(source):
            self.source = Path(find_root(source))
        else:
            self.source = Path(source)

    """Initializes a structured dataset.
    
    Args:
        source (str): The local path to folder or file containing a structured dataset and of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
    """
