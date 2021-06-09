import json
import os
from typing import Any, Dict, Optional
import uuid

from humbug.consent import HumbugConsent, environment_variable_opt_in, yes
from humbug.report import HumbugReporter

REPORT_CONFIG_FILE_NAME = "reporting_config.json"


def save_reporting_config(client_id: Optional[str] = None) -> None:
    """
    Allow or disallow reporting.
    """
    reporting_config = {}

    repo_dir = os.getcwd()

    config_report_path = os.path.join(repo_dir, REPORT_CONFIG_FILE_NAME)
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

    try:
        with open(config_report_path, "w") as ofp:
            json.dump(reporting_config, ofp)
    except Exception:
        pass


def get_reporting_config() -> Dict[str, Any]:
    reporting_config = {}
    repo_dir = os.getcwd()
    if repo_dir is not None:
        config_report_path = os.path.join(repo_dir, REPORT_CONFIG_FILE_NAME)
        try:
            if not os.path.exists(config_report_path):
                client_id = str(uuid.uuid4())
                reporting_config["client_id"] = client_id
                save_reporting_config(client_id)
            else:
                with open(config_report_path, "r") as ifp:
                    reporting_config = json.load(ifp)
        except Exception:
            pass
    return reporting_config


session_id = str(uuid.uuid4())
client_id = get_reporting_config().get("client_id")

consent = HumbugConsent(environment_variable_opt_in("REPORTING_ENABLED", yes))

reporter = HumbugReporter(
    name="activeloopai/Hub",
    consent=consent,
    client_id=client_id,
    session_id=session_id,
    bugout_token="f7176d62-73fa-4ecc-b24d-624364bddcb0",
)
