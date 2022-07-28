r"""
The hub package provides a database which stores data as compressed chunked arrays that can be stored anywhere and 
later streamed to deep learning models.
"""

import threading
from queue import Queue
from botocore.config import Config
import numpy as np
import multiprocessing
import sys
from hub.util.check_latest_version import warn_if_update_required

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)


from .api.dataset import dataset
from .api.read import read
from .api.link import link
from .api.tiled import tiled
from .core.dataset import Dataset
from .core.transform import compute, compose
from .core.tensor import Tensor
from .util.bugout_reporter import hub_reporter
from .compression import SUPPORTED_COMPRESSIONS
from .htype import HTYPE_CONFIGURATIONS
from .htype import htype
from .integrations import huggingface

compressions = list(SUPPORTED_COMPRESSIONS)
htypes = sorted(list(HTYPE_CONFIGURATIONS))
list = dataset.list
exists = dataset.exists
load = dataset.load
empty = dataset.empty
like = dataset.like
delete = dataset.delete
rename = dataset.rename
copy = dataset.copy
deepcopy = dataset.deepcopy
ingest = dataset.ingest
ingest_kaggle = dataset.ingest_kaggle
ingest_dataframe = dataset.ingest_dataframe
ingest_huggingface = huggingface.ingest_huggingface
dataset = dataset.init
tensor = Tensor

__all__ = [
    "tensor",
    "read",
    "link",
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

__version__ = "2.7.3"
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
