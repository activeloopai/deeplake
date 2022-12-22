from deeplake.constants import ALL_CLOUD_PREFIXES
from typing import Optional


def convert_creds_key(creds_key: Optional[str], path: str):
    if creds_key is None and path.startswith(ALL_CLOUD_PREFIXES):
        creds_key = "ENV"
    elif creds_key == "ENV" and not path.startswith(ALL_CLOUD_PREFIXES):
        creds_key = None
    return creds_key


def get_path_type(path: str):
    if path.startswith("s3://"):
        return "s3"
    elif path.startswith(("http://", "https://")):
        return "http"
    elif path.startswith(("gcs://", "gcp://", "gs://")):
        return "gcs"
    else:
        return "local"
