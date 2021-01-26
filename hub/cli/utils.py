"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import hub
from outdated import check_outdated
import subprocess
import pkg_resources

from hub.exceptions import HubException
from hub.log import logger


def get_cli_version():
    return "1.0.0"
