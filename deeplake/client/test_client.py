import pytest
from deeplake.cli.commands import login, logout
from click.testing import CliRunner
from deeplake.client.client import (
    DeepLakeBackendClient,
    DeepMemoryBackendClient,
    JobResponseStatusSchema,
)
from deeplake.client.utils import (
    write_token,
    read_token,
    remove_token,
)

from time import sleep


@pytest.mark.slow
def test_client_requests(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    deeplake_client = DeepLakeBackendClient()
    deeplake_client.request_auth_token(username, password)
    with pytest.raises(Exception):
        # request will fail as username already exists
        deeplake_client.send_register_request(
            "activeloop", "abc@d.com", "notactualpassword"
        )


def test_client_utils():
    write_token("abcdefgh")
    assert read_token() == "abcdefgh"
    remove_token()
    assert read_token() is None


@pytest.mark.slow
@pytest.mark.parametrize("method", ["creds", "token"])
def test_client_workspace_organizations(
    method, hub_cloud_dev_credentials, hub_cloud_dev_token
):
    username, password = hub_cloud_dev_credentials
    deeplake_client = DeepLakeBackendClient()

    runner = CliRunner()
    result = runner.invoke(logout)
    assert result.exit_code == 0

    assert deeplake_client.get_user_organizations() == ["public"]

    if method == "creds":
        runner.invoke(login, f"-u {username} -p {password}")
    elif method == "token":
        runner.invoke(login, f"-t {hub_cloud_dev_token}")

    deeplake_client = DeepLakeBackendClient()
    assert username in deeplake_client.get_user_organizations()

    runner.invoke(logout)


def create_response(
    job_id="6508464cd80cab681bfcfff3",
    dataset_id="some_dataset_id",
    organization_id="some_organization_id",
    status="training",
    progress={
        "eta": 100.34,
        "last_update_at": "2021-08-31T15:00:00.000000",
        "error": None,
        "train_recall@10": "87.8%",
        "best_recall@10": "85.5% (+2.6)%",
        "epoch": 0,
        "base_val_recall@10": 0.8292181491851807,
        "val_recall@10": "85.5%",
        "dataset": "query",
        "split": 0,
        "loss": -0.05437087118625641,
        "delta": 2.572011947631836,
    },
):
    return {
        "id": job_id,
        "dataset_id": dataset_id,
        "organization_id": organization_id,
        "status": status,
        "progress": progress,
    }


class Status:
    pending = (
        "--------------------------------------------------------------\n"
        "|                  6508464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | pending                       |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | None                          |\n"
        "--------------------------------------------------------------\n"
        "| results                    | not available yet             |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    training = (
        "--------------------------------------------------------------\n"
        "|                  6508464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | training                      |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: 100.3 seconds            |\n"
        "|                            | recall@10: 85.5% (+2.6)%      |\n"
        "|                            | dataset: query                |\n"
        "--------------------------------------------------------------\n"
        "| results                    | not available yet             |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    completed = (
        "--------------------------------------------------------------\n"
        "|                  6508464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | completed                     |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: 100.3 seconds            |\n"
        "|                            | recall@10: 85.5% (+2.6)%      |\n"
        "|                            | dataset: query                |\n"
        "--------------------------------------------------------------\n"
        "| results                    | Congratulations!              |\n"
        "|                            | Your model has                |\n"
        "|                            | achieved a recall@10          |\n"
        "|                            | of 85.5 which is an           |\n"
        "|                            | improvement of                |\n"
        "|                            | (+2.6) on the                 |\n"
        "|                            | validation set                |\n"
        "|                            | compared to naive             |\n"
        "|                            | vector search.                |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    failed = (
        "--------------------------------------------------------------\n"
        "|                  6508464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | failed                        |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: None seconds             |\n"
        "|                            | error: list indices must be   |\n"
        "|                            |        integers or slices,    |\n"
        "|                            |        not str                |\n"
        "|                            | dataset: query                |\n"
        "--------------------------------------------------------------\n"
        "| results                    | not available yet             |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    list_jobs = (
        "ID                        DATASET ID       ORGANIZATION ID       STATUS     RESULTS          "
        + 520 * " "
        + "PROGRESS       \n"
        "6508464cd80cab681bfcfff3  some_dataset_id  some_organization_id  pending    not available yet"
        + 520 * " "
        + "None           \n"
        "6508464cd80cab681bfcfff3  some_dataset_id  some_organization_id  training   not available yet"
        + 520 * " "
        + "eta: 100.3 seconds\n"
        "                                                                                             "
        + 520 * " "
        + "recall@10: 85.5% (+2.6)%\n"
        "                                                                                             "
        + 520 * " "
        + "dataset: query\n"
        "6508464cd80cab681bfcfff3  some_dataset_id  some_organization_id  completed  Congratulations! "
        + 520 * " "
        + "eta: 100.3 seconds\n"
        "                                                                            Your model has   "
        + 520 * " "
        + "recall@10: 85.5% (+2.6)%\n"
        "                                                                            achieved a       "
        + 520 * " "
        + "dataset: query \n"
        "                                                                            recall@10 of     "
        + 520 * " "
        + "\n"
        "                                                                            85.5 which is    "
        + 520 * " "
        + "\n"
        "                                                                            an improvement   "
        + 520 * " "
        + "\n"
        "                                                                            of (+2.6) on     "
        + 520 * " "
        + "\n"
        "                                                                            the validation   "
        + 520 * " "
        + "\n"
        "                                                                            set compared     "
        + 520 * " "
        + "\n"
        "                                                                            to naive         "
        + 520 * " "
        + "\n"
        "                                                                                             "
        + 520 * " "
        + "\n"
        "                                                                                             "
        + 520 * " "
        + "\n"
        "6508464cd80cab681bfcfff3  some_dataset_id  some_organization_id  failed     not available yet"
        + 520 * " "
        + "eta: None seconds\n"
        "                                                                                             "
        + 520 * " "
        + "error: list indices must beintegers or slices,not st\n"
        "                                                                                             "
        + 520 * " "
        + "dataset: query\n"
    )

    @classmethod
    def get_cancelled_str(job_id):
        return """
        --------------------------------------------------------------
        |                  {}                  |
        --------------------------------------------------------------
        | status                     | canceled                      |
        --------------------------------------------------------------
        | progress                   | None                          |
        --------------------------------------------------------------
        | results                    | not available yet             |
        --------------------------------------------------------------
        """.format(
            job_id
        )


def test_deepmemory_print_status_and_list_jobs(capsys):
    # for training that is just started
    pending_response = create_response(
        status="pending",
        progress=None,
    )
    response_schema = JobResponseStatusSchema(response=pending_response)
    response_schema.print_status("6508464cd80cab681bfcfff3")
    captured = capsys.readouterr()
    assert captured.out == Status.pending

    # for training that is in progress
    training_response = create_response()
    response_schema = JobResponseStatusSchema(response=training_response)
    response_schema.print_status("6508464cd80cab681bfcfff3")
    captured = capsys.readouterr()
    assert captured.out == Status.training

    # for training jobs that are finished
    completed_response = create_response(
        status="completed",
    )
    response_schema = JobResponseStatusSchema(response=completed_response)
    response_schema.print_status("6508464cd80cab681bfcfff3")
    captured = capsys.readouterr()
    assert captured.out == Status.completed

    # for jobs that failed
    failed_response = create_response(
        status="failed",
        progress={
            "eta": None,
            "last_update_at": "2021-08-31T15:00:00.000000",
            "error": "list indices must be integers or slices, not str",
            "dataset": "query",
        },
    )
    response_schema = JobResponseStatusSchema(response=failed_response)
    response_schema.print_status("6508464cd80cab681bfcfff3")
    captured = capsys.readouterr()
    assert captured.out == Status.failed

    responses = [
        pending_response,
        training_response,
        completed_response,
        failed_response,
    ]
    response_schema = JobResponseStatusSchema(response=responses)
    response_schema.print_jobs()
    captured = capsys.readouterr()
    assert captured.out == Status.list_jobs[1:]


@pytest.mark.slow
def test_deepmemory_train_and_cancel(
    corpus_query_pair_path, job_id, capsys, hub_cloud_dev_token
):
    # Note: this test could be flaky as it depends heavilly on deepmemory managed service
    corpus_path, query_path = corpus_query_pair_path

    client = DeepMemoryBackendClient(hub_cloud_dev_token)
    job = client.start_taining(
        corpus_path=corpus_path,
        query_path=query_path,
    )
    captured = capsys.readouterr()
    canceled = client.cancel(job_id=job["id"])
    client.check_status(job_id=job["id"])
    assert canceled == True
    assert captured.out == Status.get_cancelled_str(job["job_id"])[1:]

    captured = capsys.readouterr()
    canceled = client.cancel(job_id="non-existent-job-id")
    expected = "Job with job_id='random id' was not cancelled!\n Error: Entity random id does not exist."
    assert canceled == False
    assert captured.out == expected

    captured = capsys.readouterr()
    canceled = client.cancel(job_id=job_id)
    expected = f"Job with job_id={job_id} was not cancelled!\n "
    f"Error: Job {job_id} is not in pending state, skipping cancellation."
    assert canceled == False
    assert captured.out == expected


@pytest.mark.slow
def test_deepmemory_list_jobs(corpus_with_failed_runs, capsys, hub_cloud_dev_token):
    # Note: this test could be flaky as it depends heavilly on deepmemory managed service
    responses = []
    response = create_response(
        status="pending",
        progress=None,
    )

    response = create_response(
        status="pending",
        progress=None,
    )

    response = create_response(
        status="pending",
        progress=None,
    )
    response_schema = JobResponseStatusSchema(response=response)
    captured = capsys.readouterr()
    response_schema.check_status("6508464cd80cab681bfcfff3")
    assert captured.out == Status.list_jobs[1:]


def test_deepmemory_delete(corpus_path, query_path, hub_cloud_dev_token):
    client = DeepMemoryBackendClient(hub_cloud_dev_token)
    job = client.start_taining(
        corpus_path=corpus_path,
        query_path=query_path,
    )
    client.cancel(job_id=job["job_id"])
    client.delete(job_id=job["job_id"])
