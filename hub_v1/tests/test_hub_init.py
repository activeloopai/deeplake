"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.exceptions import HubDatasetNotFoundException
import pytest

import hub_v1
import hub_v1.config
from hub_v1.utils import dask_loaded


def test_local_mode():
    hub_v1.local_mode()
    assert hub_v1.config.HUB_REST_ENDPOINT == "http://localhost:5000"


def test_dev_mode():
    hub_v1.dev_mode()
    assert hub_v1.config.HUB_REST_ENDPOINT == "https://app.dev.activeloop.ai"


if __name__ == "__main__":
    test_local_mode()
    test_dev_mode()
    # test_load()
