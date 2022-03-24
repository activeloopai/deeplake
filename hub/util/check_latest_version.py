import warnings
import requests
from hub.core.fast_forwarding import version_compare


def get_latest_version():
    response = requests.get("https://pypi.org/pypi/hub/json")
    return response.json()["info"]["version"]


def warn_if_update_required(current_version):
    try:
        latest_version = get_latest_version()
    except Exception:
        return
    if version_compare(current_version, latest_version) < 0:
        warnings.warn(
            f"A newer version of hub ({latest_version}) is available. It's recommended that you update to the latest version using `pip install -U hub=={latest_version}`."
        )
