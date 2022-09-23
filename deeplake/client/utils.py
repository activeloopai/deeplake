import os
import json
import requests
from pathlib import Path


from hub.client.config import (
    REPORTING_CONFIG_FILE_PATH,
    TOKEN_FILE_PATH,
    HUB_AUTH_TOKEN,
)
from hub.util.exceptions import (
    AuthenticationException,
    AuthorizationException,
    BadGatewayException,
    BadRequestException,
    GatewayTimeoutException,
    LockedException,
    OverLimitException,
    ResourceNotFoundException,
    ServerException,
    UnexpectedStatusCodeException,
    EmptyTokenException,
)


def write_token(token: str):
    """Writes the auth token to the token file."""
    if not token:
        raise EmptyTokenException
    path = Path(TOKEN_FILE_PATH)
    os.makedirs(path.parent, exist_ok=True)
    with open(TOKEN_FILE_PATH, "w") as f:
        f.write(token)


def read_token():
    """Returns the token. Searches for the token first in token file and then in enviroment variables."""
    token = None
    if os.path.exists(TOKEN_FILE_PATH):
        with open(TOKEN_FILE_PATH) as f:
            token = f.read()
    else:
        token = os.environ.get(HUB_AUTH_TOKEN)

    return token


def remove_token():
    """Deletes the token file"""
    if os.path.isfile(TOKEN_FILE_PATH):
        os.remove(TOKEN_FILE_PATH)


def remove_username_from_config():
    try:
        config = {}
        with open(REPORTING_CONFIG_FILE_PATH, "r") as f:
            config = json.load(f)
            config["username"] = "public"
        with open(REPORTING_CONFIG_FILE_PATH, "w") as f:
            json.dump(config, f)
    except (FileNotFoundError, KeyError):
        return


def check_response_status(response: requests.Response):
    """Check response status and throw corresponding exception on failure."""
    code = response.status_code
    if code >= 200 and code < 300:
        return

    try:
        message = response.json()["description"]
    except Exception:
        message = " "

    if code == 400:
        raise BadRequestException(message)
    elif response.status_code == 401:
        raise AuthenticationException
    elif response.status_code == 403:
        raise AuthorizationException(message, response=response)
    elif response.status_code == 404:
        if message != " ":
            raise ResourceNotFoundException(message)
        raise ResourceNotFoundException
    elif response.status_code == 423:
        raise LockedException
    elif response.status_code == 429:
        raise OverLimitException
    elif response.status_code == 502:
        raise BadGatewayException
    elif response.status_code == 504:
        raise GatewayTimeoutException
    elif 500 <= response.status_code < 600:
        raise ServerException("Server under maintenance, try again later.")
    else:
        message = f"An error occurred. Server response: {response.status_code}"
        raise UnexpectedStatusCodeException(message)


def get_user_name() -> str:
    """Returns the name of the user currently logged into Hub."""
    path = REPORTING_CONFIG_FILE_PATH
    try:
        with open(path, "r") as f:
            d = json.load(f)
            return d["username"]
    except (FileNotFoundError, KeyError):
        return "public"
