"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import json
import uuid

from humbug.consent import HumbugConsent
from humbug.report import Reporter

from hub_v1.config import (
    HUMBUG_TOKEN,
    HUMBUG_KB_ID,
    REPORT_CONSENT_FILE_PATH,
)
from hub_v1.version import __version__ as hub_version


def hub_consent_from_file() -> bool:
    consent = False
    try:
        if not os.path.isfile(REPORT_CONSENT_FILE_PATH):
            return False
        consent_config = {}
        with open(REPORT_CONSENT_FILE_PATH, "r") as ifp:
            consent_config = json.load(ifp)
        consent = consent_config.get("consent", False)
    except Exception:
        pass
    return consent


def configure_reporting(consent, client_id=None, username=None):
    """
    Allow or disallow Hub reporting.
    """
    reporting_config = {}
    if os.path.isfile(REPORT_CONSENT_FILE_PATH):
        try:
            with open(REPORT_CONSENT_FILE_PATH, "r") as ifp:
                reporting_config = json.load(ifp)
        except Exception:
            pass

    if client_id is not None and reporting_config.get("client_id") is None:
        reporting_config["client_id"] = client_id

    if reporting_config.get("client_id") is None:
        reporting_config["client_id"] = str(uuid.uuid4())

    if username is not None:
        reporting_config["username"] = username

    reporting_config["consent"] = consent

    try:
        config_dir = os.path.dirname(REPORT_CONSENT_FILE_PATH)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        with open(REPORT_CONSENT_FILE_PATH, "w") as ofp:
            json.dump(reporting_config, ofp)
    except Exception:
        pass


def get_reporting_config():
    reporting_config = {}
    try:
        if not os.path.exists(REPORT_CONSENT_FILE_PATH):
            client_id = str(uuid.uuid4())
            reporting_config["client_id"] = client_id
            configure_reporting(True, client_id)
        with open(REPORT_CONSENT_FILE_PATH, "r") as ifp:
            reporting_config = json.load(ifp)
    except Exception:
        pass
    return reporting_config


session_id = str(uuid.uuid4())

hub_consent = HumbugConsent(hub_consent_from_file)
hub_reporter = Reporter(
    "activeloopai/Hub",
    hub_consent,
    client_id=get_reporting_config().get("client_id"),
    session_id=session_id,
    bugout_token=HUMBUG_TOKEN,
    bugout_journal_id=HUMBUG_KB_ID,
    timeout_seconds=5,
)

hub_version_tag = "version:{}".format(hub_version)
hub_tags = [hub_version_tag]

hub_user = get_reporting_config().get("username")
if hub_user is not None:
    hub_tags.append("username:{}".format(hub_user))
