import requests
import sys

from hub import config
from hub.log import logger
from hub.client.token_manager import TokenManager
from hub.cli.utils import get_cli_version

from hub.exceptions import (
    AuthenticationException,
    AuthorizationException,
    BadGatewayException,
    BadRequestException,
    GatewayTimeoutException,
    NotFoundException,
    OverLimitException,
    ServerException,
    LockedException,
    HubException,
)


def urljoin(*args):
    """
    Joins given arguments into a url. Trailing but not leading slashes are
    stripped for each argument.
    """
    return "/".join(map(lambda x: str(x).strip("/"), args))


class HubHttpClient(object):
    """
    Basic communication with Hub AI Controller rest API
    """

    def __init__(self):
        self.auth_header = TokenManager.get_auth_header()

    def request(
        self,
        method,
        relative_url,
        endpoint=None,
        params={},
        data={},
        files={},
        json={},
        timeout=config.DEFAULT_TIMEOUT,
        headers={},
    ):
        if not endpoint:
            endpoint = config.HUB_REST_ENDPOINT

        request_url = urljoin(endpoint, relative_url)
        headers["hub-cli-version"] = get_cli_version()
        if (
            "Authorization" not in headers
            or headers["Authorization"] != self.auth_header
        ):
            headers["Authorization"] = self.auth_header

        try:
            logger.debug("Sending: Headers {}, Json: {}".format(headers, json))
            response = requests.request(
                method,
                request_url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                files=files,
                timeout=timeout,
            )

        except requests.exceptions.ConnectionError as e:
            logger.debug("Exception: {}".format(e, exc_info=True))
            sys.exit("Connection error. Please retry or check your internet connection")
        except requests.exceptions.Timeout as e:
            logger.debug("Exception: {}".format(e, exc_info=True))
            sys.exit(
                "Connection timeout. Please retry or check your internet connection"
            )
        logger.debug(
            "Response Content: {}, Headers: {}".format(
                response.content, response.headers
            )
        )
        self.check_response_status(response)
        return response

    def check_response_status(self, response):
        """
        Check response status and throw corresponding exception on failure
        """
        code = response.status_code
        if code < 200 or code >= 300:
            try:
                message = response.json()["error"]
            except Exception:
                message = " "

            logger.debug(
                'Error received: status code: {}, message: "{}"'.format(code, message)
            )
            if code == 400:
                raise BadRequestException(response)
            elif response.status_code == 401:
                raise AuthenticationException()
            elif response.status_code == 403:
                raise AuthorizationException()
            elif response.status_code == 404:
                raise NotFoundException()
            elif response.status_code == 429:
                raise OverLimitException(message)
            elif response.status_code == 502:
                raise BadGatewayException()
            elif response.status_code == 504:
                raise GatewayTimeoutException(message)
            elif response.status_code == 423:
                raise LockedException(message)
            elif 500 <= response.status_code < 600:
                if "Server under maintenance" in response.content.decode():
                    raise ServerException(
                        "Server under maintenance, please try again later."
                    )
                else:
                    raise ServerException()
            else:
                msg = "An error occurred. Server response: {}".format(
                    response.status_code
                )
                raise HubException(message=msg)
