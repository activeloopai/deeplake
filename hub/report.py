"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import json
import uuid

import click
from humbug.consent import HumbugConsent, environment_variable_opt_out
from humbug.report import Reporter

from hub.config import (
    HUMBUG_TOKEN,
    HUMBUG_KB_ID,
    REPORT_CONSENT_FILE_PATH,
    REPORT_CONSENT_ENV_VAR,
)


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


hub_consent_override_from_env = environment_variable_opt_out(
    REPORT_CONSENT_ENV_VAR,
    ["0", "f", "F", "false", "False", "FALSE", "n", "N", "no", "No", "NO"],
)


@click.command()
@click.option("--allow/--disallow", default=True)
def consent(allow: bool) -> None:
    """
    Allow or disallow Hub reporting.
    """
    consent_config = {}
    if os.path.isfile(REPORT_CONSENT_FILE_PATH):
        try:
            with open(REPORT_CONSENT_FILE_PATH, "r") as ifp:
                consent_config = json.load(ifp)
        except Exception:
            pass

    if consent_config.get("client_id") is None:
        consent_config["client_id"] = str(uuid.uuid4())

    consent_config["consent"] = allow

    config_dir = os.path.dirname(REPORT_CONSENT_FILE_PATH)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    with open(REPORT_CONSENT_FILE_PATH, "w") as ofp:
        json.dump(consent_config, ofp)


hub_consent = HumbugConsent(hub_consent_from_file, hub_consent_override_from_env)
hub_reporter = Reporter(
    "activeloopai/Hub",
    hub_consent,
    bugout_token=HUMBUG_TOKEN,
    bugout_journal_id=HUMBUG_KB_ID,
    timeout_seconds=5,
)

client_id = None
try:
    with open(REPORT_CONSENT_FILE_PATH, "r") as ifp:
        consent_config = json.load(ifp)
    client_id = consent_config.get("client_id")
except Exception:
    pass
client_id_tag = "client_id:{}".format(client_id)

session_id = str(uuid.uuid4())
session_id_tag = "session_id:{}".format(session_id)

hub_version_tag = "version:1.2.2-dev-reporting"


class ExceptionWithReporting(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hub_reporter.error_report(
            self, tags=[client_id_tag, session_id_tag, hub_version_tag], publish=True
        )
