import threading
from queue import Queue
from botocore.config import Config
import numpy as np
import multiprocessing
import sys
from hub.util.check_latest_version import warn_if_update_required

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

__pdoc__ = {
    "api": False,
    "auto": False,
    "cli": False,
    "client": False,
    "constants": False,
    "config": False,
    "integrations": False,
    "tests": False,
    "util": False,
    "Dataset.clear_cache": False,
    "Dataset.flush": False,
    "Dataset.read_only": False,
    "Dataset.size_approx": False,
    "Dataset.token": False,
    "Dataset.num_samples": False,
}
from .api.dataset import dataset as api_dataset
from .api.read import read
from .api.link import link
from .api.tiled import tiled
from .core.dataset import Dataset
from .core.transform import compute, compose
from .core.tensor import Tensor
from .util.bugout_reporter import hub_reporter
from .compression import SUPPORTED_COMPRESSIONS
from .htype import HTYPE_CONFIGURATIONS
from .integrations import huggingface

compressions = list(SUPPORTED_COMPRESSIONS)
htypes = sorted(list(HTYPE_CONFIGURATIONS))
list = api_dataset.list
exists = api_dataset.exists
load = api_dataset.load
empty = api_dataset.empty
like = api_dataset.like
delete = api_dataset.delete
rename = api_dataset.rename
copy = api_dataset.copy
deepcopy = api_dataset.deepcopy
ingest = api_dataset.ingest
ingest_kaggle = api_dataset.ingest_kaggle
ingest_dataframe = api_dataset.ingest_dataframe
ingest_huggingface = huggingface.ingest_huggingface
dataset = api_dataset.init
tensor = Tensor

__all__ = [
    "tensor",
    "read",
    "__version__",
    "load",
    "empty",
    "exists",
    "compute",
    "compose",
    "copy",
    "dataset",
    "Dataset",
    "deepcopy",
    "like",
    "list",
    "ingest",
    "ingest_kaggle",
    "ingest_huggingface",
    "compressions",
    "htypes",
    "config",
    "delete",
    "copy",
    "rename",
]

__version__ = "2.4.1"
warn_if_update_required(__version__)
__encoded_version__ = np.array(__version__)
config = {"s3": Config(max_pool_connections=50, connect_timeout=300, read_timeout=300)}


hub_reporter.tags.append(f"version:{__version__}")
hub_reporter.system_report(publish=True)
hub_reporter.setup_excepthook(publish=True)

event_queue: Queue = Queue()


def send_event():
    while True:
        try:
            event = event_queue.get()
            client, event_dict = event
            client.send_event(event_dict)
        except Exception:
            pass


threading.Thread(target=send_event, daemon=True).start()
