import time


def expires_in_to_expires_at(creds: dict) -> None:
    if "expires_in" not in creds:
        return
    creds["expires_at"] = time.time() + creds.pop("expires_in")


def is_expired_token(creds: dict) -> bool:
    if "Authorization" not in creds:
        return False
    if "expires_at" not in creds:
        return False
    return creds["expires_at"] < time.time()
