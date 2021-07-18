import json
import os
from typing import Any, Dict, Optional
import uuid

from hub.client.config import REPORTING_CONFIG_FILE_PATH
from humbug.consent import HumbugConsent
from humbug.report import HumbugReporter


def save_reporting_config(
    consent: bool, client_id: Optional[str] = None, username: Optional[str] = None
) -> None:
    """Modify reporting config.

    Args:
        consent (bool): Enabling and disabling sending crashes and system report to Activeloop Hub.
        client_id (str, optional): Unique client id.
        username (str, optional): Activeloop username.
    """
    reporting_config = {}

    repo_dir = os.getcwd()

    config_report_path = os.path.join(repo_dir, REPORTING_CONFIG_FILE_PATH)
    if os.path.isfile(config_report_path):
        try:
            with open(config_report_path, "r") as ifp:
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
        with open(config_report_path, "w") as ofp:
            json.dump(reporting_config, ofp)
    except Exception:
        pass


def get_reporting_config() -> Dict[str, Any]:
    """Get an existing reporting config"""
    reporting_config = {}
    repo_dir = os.getcwd()
    if repo_dir is not None:
        config_report_path = os.path.join(repo_dir, REPORTING_CONFIG_FILE_PATH)
        try:
            if not os.path.exists(config_report_path):
                client_id = str(uuid.uuid4())
                reporting_config["client_id"] = client_id
                save_reporting_config(True, client_id)
            else:
                with open(config_report_path, "r") as ifp:
                    reporting_config = json.load(ifp)
        except Exception:
            pass
    return reporting_config


def consent_from_reporting_config_file() -> bool:
    """Get consent settings from the existing reporting config"""
    reporting_config = get_reporting_config()
    return reporting_config.get("consent", False)


session_id = str(uuid.uuid4())
client_id = get_reporting_config().get("client_id")

consent = HumbugConsent(consent_from_reporting_config_file)

hub_reporter = HumbugReporter(
    name="activeloopai/Hub",
    consent=consent,
    client_id=client_id,
    session_id=session_id,
    bugout_token="f7176d62-73fa-4ecc-b24d-624364bddcb0",
    tags=[],
)

hub_user = get_reporting_config().get("username")
if hub_user is not None:
    hub_reporter.tags.append(f"username:{hub_user}")
