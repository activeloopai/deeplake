import json
import jwt
import os
from pathlib import Path
from platform import machine
from typing import Any, Dict, Optional, Union
import uuid

from deeplake.client.config import REPORTING_CONFIG_FILE_PATH, DEEPLAKE_AUTH_TOKEN
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.bugout_token import BUGOUT_TOKEN
from humbug.consent import HumbugConsent
from humbug.report import HumbugReporter

import pathlib

from deeplake.util.path import (
    convert_pathlib_to_string_if_needed,
)


def save_reporting_config(
    consent: bool, client_id: Optional[str] = None, username: Optional[str] = None
) -> Dict[str, Any]:
    """Modify reporting config.

    Args:
        consent (bool): Enabling and disabling sending crashes and system report to Activeloop Hub.
        client_id (str, optional): Unique client id.
        username (str, optional): Activeloop username.

    Returns:
        The configuration that it just saved.
    """
    reporting_config = {}

    if os.path.isfile(REPORTING_CONFIG_FILE_PATH):
        try:
            with open(REPORTING_CONFIG_FILE_PATH, "r") as ifp:
                reporting_config = json.load(ifp)
        except Exception:
            pass
    else:
        # We should not expect that the parent directory for the reporting configuration will exist.
        # If it doesn't exist, we create the directory, if possible.
        # This mirrors the code for the `write_token` method in /deeplake/client/utils.py.
        path = Path(REPORTING_CONFIG_FILE_PATH)
        os.makedirs(path.parent, exist_ok=True)

    if client_id is not None and reporting_config.get("client_id") is None:
        reporting_config["client_id"] = client_id

    if reporting_config.get("client_id") is None:
        reporting_config["client_id"] = str(uuid.uuid4())

    if username is not None:
        reporting_config["username"] = username

    reporting_config["consent"] = consent

    try:
        with open(REPORTING_CONFIG_FILE_PATH, "w") as ofp:
            json.dump(reporting_config, ofp)
    except Exception:
        pass

    return reporting_config


def get_reporting_config() -> Dict[str, Any]:
    """Get an existing reporting config"""
    reporting_config: Dict[str, Any] = {"consent": False}
    try:
        if not os.path.exists(REPORTING_CONFIG_FILE_PATH):
            client_id = str(uuid.uuid4())
            reporting_config["client_id"] = client_id
            reporting_config = save_reporting_config(True, client_id)
        else:
            with open(REPORTING_CONFIG_FILE_PATH, "r") as ifp:
                reporting_config = json.load(ifp)

        # The following changes do NOT mutate the reporting_config.json file on the file system, but
        # they provide a means to report the username as the client_id (if the username is available)
        # while tracking the existing client_id as a machine_id.
        reporting_config["machine_id"] = reporting_config["client_id"]

        if (
            reporting_config.get("username") is not None
            and reporting_config["client_id"] != reporting_config["username"]
        ):
            reporting_config["client_id"] = reporting_config["username"]

    except Exception:
        # Not being able to load reporting consent should not get in the user's way. We will just
        # return the default reporting_config object in which consent is set to False.
        pass
    return reporting_config


def consent_from_reporting_config_file() -> bool:
    """Get consent settings from the existing reporting config"""
    reporting_config = get_reporting_config()
    return reporting_config.get("consent", False)


consent = HumbugConsent(consent_from_reporting_config_file)

session_id = str(uuid.uuid4())
bugout_reporting_config = get_reporting_config()
client_id = bugout_reporting_config.get("client_id")


def blacklist_token_parameters_fn(params: Dict[str, Any]) -> Dict[str, Any]:
    admissible_params = {k: v for k, v in params.items() if "token" not in k.lower()}
    return admissible_params


deeplake_reporter = HumbugReporter(
    name="activeloopai/Hub",
    consent=consent,
    client_id=client_id,
    session_id=session_id,
    bugout_token=BUGOUT_TOKEN,
    blacklist_fn=blacklist_token_parameters_fn,
    tags=[],
)


def set_username(username: str) -> None:
    index, current_username = find_current_username()

    if current_username is None:
        deeplake_reporter.tags.append(f"username:{username}")
    else:
        if f"username:{username}" != current_username:
            deeplake_reporter.tags[index] = f"username:{username}"


hub_user = bugout_reporting_config.get("username")
if hub_user is not None:
    deeplake_reporter.tags.append(f"username:{hub_user}")

machine_id = bugout_reporting_config.get("machine_id")
if machine_id is not None:
    deeplake_reporter.tags.append(f"machine_id:{machine_id}")


def feature_report_path(
    path: Union[str, pathlib.Path],
    feature_name: str,
    parameters: dict,
    starts_with: str = "hub://",
    token: Optional[str] = None,
    username: str = "public",
):
    """Helper function for generating humbug feature reports depending on the path"""

    if not deeplake_reporter.consent.check():
        return

    path = convert_pathlib_to_string_if_needed(path)

    if path.startswith(starts_with):
        parameters["Path"] = path

    token = token or os.environ.get(DEEPLAKE_AUTH_TOKEN)
    if token is not None:
        try:
            username = jwt.decode(token, options={"verify_signature": False})["id"]
        except Exception:
            username = "public"

    set_username(username)

    deeplake_reporter.feature_report(
        feature_name=feature_name,
        parameters=parameters,
    )


def find_current_username():
    for index, tag in enumerate(deeplake_reporter.tags):
        if "username" in tag:
            return index, tag
    return None, None
