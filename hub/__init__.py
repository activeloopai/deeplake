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

from .api.dataset import Dataset
from .api.tensor import Tensor
from .api.load import load
from .util.bugout_reporter import hub_reporter
from . import auto

__all__ = ["Dataset", "Tensor", "load", "__version__", "auto"]

__version__ = "2.0.0"
__encoded_version__ = np.array(__version__)

hub_reporter.tags.append(f"version:{__version__}")
hub_reporter.system_report(publish=True)
hub_reporter.setup_excepthook(publish=True)
