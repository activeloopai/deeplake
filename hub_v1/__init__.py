"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np

import hub_v1.config

from hub_v1.api.dataset import Dataset

from hub_v1.compute import transform
from hub_v1.log import logger
import traceback
from hub_v1.exceptions import (
    DaskModuleNotInstalledException,
    HubDatasetNotFoundException,
)




from hub_v1.report import hub_reporter, hub_tags
from hub_v1.version import __version__


def local_mode():
    hub_v1.config.HUB_REST_ENDPOINT = hub_v1.config.HUB_LOCAL_REST_ENDPOINT


def dev_mode():
    hub_v1.config.HUB_REST_ENDPOINT = hub_v1.config.HUB_DEV_REST_ENDPOINT


def dtype(*args, **kwargs):
    return np.dtype(*args, **kwargs)
