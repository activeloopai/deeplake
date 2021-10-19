import numpy as np

__pdoc__ = {
    "core": False,
    "api": False,
    "cli": False,
    "client": False,
    "constants": False,
    "integrations": False,
    "tests": False,
    "Dataset.clear_cache": False,
    "Dataset.delete": False,
    "Dataset.flush": False,
    "Dataset.read_only": False,
    "Dataset.size_approx": False,
    "Dataset.token": False,
    "Dataset.num_samples": False,
}

from .api.dataset import dataset
from .api.read import read
from .core.dataset import Dataset
from .core.transform import compute, compose
from .core.tensor import Tensor
from .util.bugout_reporter import hub_reporter
from .compression import SUPPORTED_COMPRESSIONS
from .htype import HTYPE_CONFIGURATIONS

compressions = list(SUPPORTED_COMPRESSIONS)
htypes = sorted(list(HTYPE_CONFIGURATIONS))
list = dataset.list
load = dataset.load
empty = dataset.empty
like = dataset.like
list = dataset.list
dataset_cl = Dataset
ingest = dataset.ingest
ingest_kaggle = dataset.ingest_kaggle
tensor = Tensor

__all__ = [
    "dataset",
    "tensor",
    "read",
    "__version__",
    "load",
    "empty",
    "compute",
    "compose",
    "like",
    "list",
    "dataset_cl",
    "ingest",
    "ingest_kaggle",
    "compressions",
    "htypes",
]

__version__ = "2.0.13"
__encoded_version__ = np.array(__version__)

hub_reporter.tags.append(f"version:{__version__}")
hub_reporter.system_report(publish=True)
hub_reporter.setup_excepthook(publish=True)
