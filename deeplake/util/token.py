import time
from warnings import warn


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


def src_token_and_dest_token_deprecation_warning(src_token, dest_token, token):
    if src_token is None and dest_token is None and token is None:
        return src_token, dest_token

    if src_token or dest_token:
        warn(
            "`src_token` and `dest_token` are deprecated, better to use just single `token`",
            FutureWarning,
            stacklevel=2,
        )

    if dest_token and src_token and token:
        warn(
            "You are using `dest_token` and `src_token` with `token`. Only `token` will be executed",
            UserWarning,
            stacklevel=2,
        )
        dest_token = token
        src_token = token

    elif dest_token and token:
        warn(
            "You are using `dest_token` with `token`. Only `token` will be executed",
            UserWarning,
            stacklevel=2,
        )
        dest_token = token

    elif src_token and token:
        warn(
            "You are using `src_token` with `token`. Only `token` will be executed",
            UserWarning,
            stacklevel=2,
        )
        src_token = token

    if dest_token is None:
        dest_token = token

    if src_token is None:
        src_token = token

    return src_token, dest_token
