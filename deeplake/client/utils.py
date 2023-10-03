import os
import json
import requests
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Union, Optional


from deeplake.client.config import (
    REPORTING_CONFIG_FILE_PATH,
    TOKEN_FILE_PATH,
    DEEPLAKE_AUTH_TOKEN,
)
from deeplake.util.exceptions import (
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
    UnprocessableEntityException,
)


PENDING_STATUS = "not available yet"
BEST_RECALL = "best_recall@10"


def write_token(token: str):
    """Writes the auth token to the token file."""
    if not token:
        raise EmptyTokenException
    path = Path(TOKEN_FILE_PATH)
    os.makedirs(path.parent, exist_ok=True)
    with open(TOKEN_FILE_PATH, "w") as f:
        f.write(token)


def read_token(from_env=True):
    """Returns the token. Searches for the token first in token file and then in enviroment variables."""
    token = None
    if os.path.exists(TOKEN_FILE_PATH):
        with open(TOKEN_FILE_PATH) as f:
            token = f.read()
    elif from_env:
        token = os.environ.get(DEEPLAKE_AUTH_TOKEN)
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
    elif response.status_code == 422:
        raise UnprocessableEntityException(message)
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


# TODO: Refactor JobResponseStatusSchema code as it have a lot of hard coded edge cases and duplications
# Ideally, need to create a class called Table for status and Docker table class for job list, then
# create subclasses for each entry and move edge cases to specific classes. Also move out all hard coded
# constants into deeplake.constans class.
class JobResponseStatusSchema:
    def __init__(self, response: Dict[str, Any]):
        if not isinstance(response, List):
            responses = [response]
        else:
            responses = response

        self.responses: List[Dict[str, Any]] = responses
        self.validate_status_response()

    def validate_status_response(self):
        for response in self.responses:
            if "dataset_id" not in response:
                raise ValueError("Invalid response. Missing 'dataset_id' key.")

            if "id" not in response:
                raise ValueError("Invalid response. Missing 'id' key.")

    def print_status(
        self,
        job_id: Union[str, List[str]],
        recall: Optional[str] = None,
        importvement: Optional[str] = None,
    ):
        if not isinstance(job_id, List):
            job_id = [job_id]

        line = "-" * 62
        for response in self.responses:
            if response["id"] not in job_id:
                continue

            if response["status"] == "completed":
                response["results"] = get_results(
                    response,
                    " " * 30,
                    add_vertical_bars=True,
                    recall=recall,
                    improvement=importvement,
                )

            print(line)
            print("|{:^60}|".format(response["id"]))
            print(line)
            print("| {:<27}| {:<30}|".format("status", response["status"]))
            print(line)
            progress = preprocess_progress(
                response,
                " " * 30,
                add_vertical_bars=True,
                recall=recall,
                improvement=importvement,
            )
            progress_string = "| {:<27}| {:<30}"
            if progress == "None":
                progress_string += "|"
            progress_string = progress_string.format("progress", progress)
            print(progress_string)
            print(line)
            results_str = "| {:<27}| {:<30}|"
            if not response.get("results"):
                response["results"] = PENDING_STATUS

            print(results_str.format("results", response["results"]))
            print(line)
            print("\n")

    def print_jobs(
        self,
        debug: bool,
        recalls: str,
        improvements: str,
    ):
        (
            id_size,
            status_size,
            results_size,
            progress_size,
        ) = get_table_size(responses=self.responses)

        header_format = f"{{:<{id_size}}}  {{:<{status_size}}}  {{:<{results_size}}}  {{:<{progress_size}}}"
        data_format = header_format  # as they are the same

        output_str = header_format.format("ID", "STATUS", "RESULTS", "PROGRESS")

        for response in self.responses:
            response_id = response["id"]
            response_status = response["status"]

            response_results = (
                response["results"] if response.get("results") else PENDING_STATUS
            )
            if response_status == "completed":
                response_results = get_results(
                    response,
                    "",
                    add_vertical_bars=False,
                    width=15,
                    recall=recalls[response_id],
                    improvement=improvements[response_id],
                )

            response_progress = preprocess_progress(
                response,
                "",
                add_vertical_bars=False,
                recall=recalls[response_id],
                improvement=improvements[response_id],
            )
            response_progress_items = response_progress.split("\n")

            not_allowed_response_progress_items = ["dataset: query"]
            first_time = True
            for idx, response_progress_item in enumerate(response_progress_items):
                if response_progress_item in not_allowed_response_progress_items:
                    continue

                if first_time:
                    first_time = False
                    output_str += "\n" + data_format.format(
                        response_id,
                        response_status,
                        response_results,
                        response_progress_item,
                    )
                else:
                    output_str += "\n" + data_format.format(
                        "",
                        "",
                        "",
                        response_progress_item,
                    )

            # else:
            #     output_str += "\n" + data_format.format(
            #         response_id,
            #         response_status,
            #         response_results,
            #         str(response_progress),
            #     )

        print(output_str)
        if debug:
            return output_str


def get_results(
    response: Dict[str, Any],
    indent: str,
    add_vertical_bars: bool,
    width: int = 21,
    improvement: Optional[str] = None,
    recall: Optional[str] = None,
):
    progress = response["progress"]
    for progress_key, progress_value in progress.items():
        if progress_key == BEST_RECALL:
            curr_recall, curr_improvement = progress_value.split("%")[:2]

            recall = recall or curr_recall
            improvement = improvement or curr_improvement

            if "(" not in improvement:
                improvement = f"(+{improvement}%)"

            output = f"recall@10: {str(recall)}% {improvement}"
            return output


def preprocess_progress(
    response: Dict[str, Any],
    progress_indent: str,
    add_vertical_bars: bool = False,
    recall: Optional[str] = None,
    improvement: Optional[str] = None,
):
    allowed_progress_items = ["eta", BEST_RECALL, "error"]
    progress_indent = (
        "|" + progress_indent[:-1] if add_vertical_bars else progress_indent
    )
    response_progress = response["progress"] if response.get("progress") else "None"

    if response_progress != "None":
        response_progress_str = ""
        first_entry = True
        first_error_line = True
        for key, value in response_progress.items():
            if key not in allowed_progress_items:
                continue

            if key == "error" and value is None:
                continue

            elif key == "error" and value is not None:
                value = textwrap.fill(value, width=20)
                values = value.split("\n")
                value = ""
                for value_i in values:
                    if first_error_line:
                        first_error_line = False
                        if add_vertical_bars:
                            value_i += (23 - len(value_i)) * " " + "|"
                    elif add_vertical_bars:
                        value_i = (
                            "\n"
                            + progress_indent[:-1]
                            + "| "
                            + " " * 7
                            + value_i
                            + (23 - len(value_i)) * " "
                            + "|"
                        )
                    value += value_i
                if value[-1] == "|":
                    value = value[:-1]

            if isinstance(value, float):
                value = f"{value:.1f}"

            # Add indentation for every line after the first
            if first_entry:
                first_entry = False
            else:
                response_progress_str += (
                    progress_indent[:-1] + "| "
                    if add_vertical_bars
                    else progress_indent
                )

            key_value_pair = f"{key}: {value}"

            if key == "eta":
                key_value_pair = f"{key}: {value} seconds"
            elif key == BEST_RECALL:
                key = "recall@10"
                recall = recall or value.split("%")[0]
                improvement = improvement or value.split("%")[1]
                if "(" not in improvement:
                    improvement = f"(+{improvement}%)"
                key_value_pair = f"{key}: {recall}% {improvement}"

            vertical_bar_if_needed = (
                (30 - len(key_value_pair)) * " " + "|\n" if add_vertical_bars else "\n"
            )
            key_value_pair += vertical_bar_if_needed
            response_progress_str += key_value_pair

        response_progress = response_progress_str.rstrip()  # remove trailing newline
    return response_progress


def get_table_size(responses: List[Dict[str, Any]]):
    id_size, status_size, results_size, progress_size = (
        2,  # Minimum size to fit "ID"
        6,  # Minimum size to fit "STATUS"
        29,  # Minimum size to fit "RESULTS"
        15,  # Minimum size to fit "PROGRESS"
    )
    for response in responses:
        id_size = max(id_size, len(response["id"]))
        status_size = max(status_size, len(response["status"]))
        results_size = max(results_size, len(response.get("results", PENDING_STATUS)))

    return id_size, status_size, results_size, progress_size
