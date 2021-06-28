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
from .core.transform.transform import transform  # type: ignore
from .util.bugout_reporter import hub_reporter

__all__ = ["Dataset", "Tensor", "load", "transform", "__version__"]

__version__ = "2.0a7"

hub_reporter.tags.append(f"version:{__version__}")
hub_reporter.system_report(publish=True)
hub_reporter.setup_excepthook(publish=True)
