from abc import ABC, abstractmethod
from hub.util.path import find_root
from pathlib import Path


class UnstructuredDataset(ABC):
    def __init__(self, source: str):
        self.source = Path(find_root(source))

    """Initializes an unstructured dataset.
    
    Args:
        source (str): The full path to the dataset.
            Can be a Hub cloud path of the form hub://username/datasetname. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
            Can be a s3 path of the form s3://bucketname/path/to/dataset. Credentials are required in either the environment or passed to the creds argument.
            Can be a local file system path of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
            Can be a memory path of the form mem://path/to/dataset which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
    """

    @abstractmethod
    def structure(ds, use_progress_bar: bool = True, **kwargs):
        pass
