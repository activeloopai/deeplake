import numpy as np

__pdoc__ = {
    "core": False,
    "api": False,
    "cli": False,
    "client": False,
    "constants": False,
    "integrations": False,
    "tests": False,
}

from .api.dataset_handler import dataset
from .api.tensor import Tensor
from .api.read import read
from .util.bugout_reporter import hub_reporter

load = dataset.load
empty = dataset.empty
__all__ = ["dataset", "Tensor", "read", "__version__", "load", "empty"]

__version__ = "2.0.2"
__encoded_version__ = np.array(__version__)

hub_reporter.tags.append(f"version:{__version__}")
hub_reporter.system_report(publish=True)
hub_reporter.setup_excepthook(publish=True)
