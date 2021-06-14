"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import logging
import sys

logger = logging.getLogger("hub")


def configure_logger(debug=0):
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logging.basicConfig(format="%(message)s", level=log_level, stream=sys.stdout)


configure_logger(0)
