from .collections import dataset
from .collections.dataset.core import Transform
from .collections import tensor
from .collections.dataset import load
from .collections.client_manager import init
import hub.config


def local_mode():
    hub.config.HUB_REST_ENDPOINT = hub.config.HUB_LOCAL_REST_ENDPOINT
