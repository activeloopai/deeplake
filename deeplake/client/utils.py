import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Union


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
            response = [response]

        self.responses = response
        self.validate_status_response()

    def validate_status_response(self):
        for response in self.responses:
            if "dataset_id" not in response:
                raise ValueError("Invalid response. Missing 'dataset_id' key.")

            if "id" not in response:
                raise ValueError("Invalid response. Missing 'id' key.")

    def print_status(self, job_id: Union[str, List[str]]):
        if not isinstance(job_id, List):
            job_id = [job_id]

        line = "-" * 62
        for response in self.responses:
            if response["id"] not in job_id:
                continue

            if response["status"] == "completed":
                response["results"] = get_results(
                    response, " " * 30, add_vertical_bars=True
                )

            print(line)
            print("|{:^60}|".format(response["id"]))
            print(line)
            print("| {:<27}| {:<30}|".format("status", response["status"]))
            print(line)
            progress = preprocess_progress(response, " " * 30, add_vertical_bars=True)
            progress_string = "| {:<27}| {:<30}"
            if progress == "None":
                progress_string += "|"
            progress_string = progress_string.format("progress", progress)
            print(progress_string)
            print(line)
            results_str = "| {:<27}| {:<30}"
            if not response.get("results"):
                results_str += "|"
                response["results"] = "not available yet"

            print(results_str.format("results", response["results"]))
            print(line)
            print("\n")

    def print_jobs(self, debug: bool = False):
        (
            id_size,
            dataset_id_size,
            organization_id_size,
            status_size,
            results_size,
        ) = get_table_size(responses=self.responses)

        progress_size = (
            15  # Keep it fixed for simplicity or compute dynamically if needed
        )

        header_format = f"{{:<{id_size}}}  {{:<{dataset_id_size}}}  {{:<{organization_id_size}}}  {{:<{status_size}}}  {{:<{results_size}}}  {{:<{progress_size}}}"
        data_format = header_format  # as they are the same

        output_str = header_format.format(
            "ID", "DATASET ID", "ORGANIZATION ID", "STATUS", "RESULTS", "PROGRESS"
        )

        for response in self.responses:
            response_id = response["id"]
            response_dataset_id = response["dataset_id"]
            response_organization_id = response["organization_id"]
            response_status = response["status"]

            response_results = (
                response["results"] if response.get("results") else "not available yet"
            )
            if response_status == "completed":
                progress_indent = " " * (
                    id_size
                    + dataset_id_size
                    + organization_id_size
                    + status_size
                    + 5 * 2
                )
                response_results = get_results(
                    response,
                    "",
                    add_vertical_bars=False,
                    width=15,
                )

            progress_indent = " " * (
                id_size
                + dataset_id_size
                + organization_id_size
                + status_size
                + results_size
                + 5 * 2
            )
            response_progress = preprocess_progress(
                response, progress_indent, add_vertical_bars=False
            )

            if response_status == "completed":
                response_progress = preprocess_progress(
                    response, "", add_vertical_bars=False
                )
                response_results_items = response_results.split("\n")[1:]
                response_progress_items = response_progress.split("\n")

                first_time = True
                for idx, response_results_item in enumerate(response_results_items):
                    if first_time:
                        first_time = False
                        output_str += data_format.format(
                            response_id,
                            response_dataset_id,
                            response_organization_id,
                            response_status,
                            response_results_item,
                            response_progress_items[idx],
                        )
                    else:
                        response_progress_item = ""
                        if idx < len(response_progress_items):
                            response_progress_item = response_progress_items[idx]

                        output_str += data_format.format(
                            "",
                            "",
                            "",
                            "",
                            response_results_item,
                            response_progress_item,
                        )

            else:
                output_str += data_format.format(
                    response_id,
                    response_dataset_id,
                    response_organization_id,
                    response_status,
                    response_results,
                    str(response_progress),
                )

        print(output_str)
        if debug:
            return output_str


def get_results(
    response: Dict[str, Any],
    indent: str,
    add_vertical_bars: bool,
    width: int = 21,
):
    progress = response["progress"]
    for progress_key, progress_value in progress.items():
        if progress_key == "best_recall@10":
            recall, improvement = progress_value.split("%")[:2]

            output = (
                "Congratulations! Your model has achieved a recall@10 of "
                + str(recall)
                + " which is an improvement of "
                + str(improvement)
                + " on the validation set compared to naive vector search."
            )
            return format_to_fixed_width(output, width, indent, add_vertical_bars)


def format_to_fixed_width(
    s: str,
    width: int,
    indent: str,
    add_vertical_bars: bool,
):
    words = s.split()
    lines = ""
    line = ""
    first_entry = True
    for word in words:
        # If adding the new word to the current line would make it too long
        if len(line) + len(word) + 1 > width:  # +1 for space
            current_indent = ""
            if first_entry:
                first_entry = False
            else:
                current_indent = (
                    "|" + indent[:-2] + "| " if add_vertical_bars else indent
                )
            lines += (
                current_indent + line.rstrip()
            )  # Add the current line to lines and remove trailing spaces
            lines += (30 - len(line)) * " " + " |\n" if add_vertical_bars else "\n"
            line = ""  # Start a new line
        line += word + " "

    # Add the last line if it's not empty
    if line:
        current_indent = "|" + indent[:-2] + "| " if add_vertical_bars else indent
        lines += current_indent
        lines += (
            line.rstrip() + (30 - len(line)) * " " + " |"
            if add_vertical_bars
            else line.rstrip() + "\n"
        )

    return lines


def preprocess_progress(
    response: Dict[str, Any],
    progress_indent: str,
    add_vertical_bars: bool = False,
):
    allowed_progress_items = ["eta", "best_recall@10", "dataset", "error"]
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
            elif key == "best_recall@10":
                key = "recall@10"
                key_value_pair = f"{key}: {value}"

            vertical_bar_if_needed = (
                (30 - len(key_value_pair)) * " " + "|\n" if add_vertical_bars else "\n"
            )
            key_value_pair += vertical_bar_if_needed
            response_progress_str += key_value_pair

        response_progress = response_progress_str.rstrip()  # remove trailing newline
    return response_progress


def get_table_size(responses: Dict[str, Any]):
    id_size, dataset_id_size, organization_id_size, status_size, results_size = (
        2,  # Minimum size to fit "ID"
        10,  # Minimum size to fit "DATASET ID"
        15,  # Minimum size to fit "ORGANIZATION ID"
        6,  # Minimum size to fit "STATUS"
        7,  # Minimum size to fit "RESULTS"
    )
    for response in responses:
        id_size = max(id_size, len(response["id"]))
        dataset_id_size = max(dataset_id_size, len(response["dataset_id"]))
        organization_id_size = max(
            organization_id_size, len(response["organization_id"])
        )
        status_size = max(status_size, len(response["status"]))
        results_size = max(
            results_size, len(response.get("results", "not available yet"))
        )

    return id_size, dataset_id_size, organization_id_size, status_size, results_size
