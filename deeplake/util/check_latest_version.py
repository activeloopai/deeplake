import json
import os
import warnings
import requests  # type: ignore
from deeplake.client.config import HUB_PYPI_VERSION_PATH
from deeplake.core.fast_forwarding import version_compare
import time


def get_latest_version():
    if os.path.exists(HUB_PYPI_VERSION_PATH):
        with open(HUB_PYPI_VERSION_PATH) as f:
            latest_version, saved_time = json.load(f)
            seconds_in_a_day = 60 * 60 * 24
            time_elapsed = time.time() - saved_time
            if time_elapsed < seconds_in_a_day:
                return latest_version

    response = requests.get("https://pypi.org/pypi/deeplake/json", timeout=2)
    latest_version = response.json()["info"]["version"]
    with open(HUB_PYPI_VERSION_PATH, "w") as f:
        json.dump((latest_version, time.time()), f)
    return latest_version


def warn_if_update_required(current_version):
    try:
        latest_version = get_latest_version()
    except Exception:
        return
    if version_compare(current_version, latest_version) < 0:
        warnings.warn(
            f"A newer version of deeplake ({latest_version}) is available. It's recommended that you update to the latest version using `pip install -U deeplake`."
        )
