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
        "|                  1238464cd80cab681bfcfff3                  |\n"
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
        "|                  3218464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | training                      |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: 100.3 seconds            |\n"
        "|                            | recall@10: 85.5% (+2.6%)      |\n"
        "--------------------------------------------------------------\n"
        "| results                    | not available yet             |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    completed = (
        "--------------------------------------------------------------\n"
        "|                  2138464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | completed                     |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: 100.3 seconds            |\n"
        "|                            | recall@10: 85.5% (+2.6%)      |\n"
        "--------------------------------------------------------------\n"
        "| results                    | recall@10: 85.5% (+2.6%)      |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    failed = (
        "--------------------------------------------------------------\n"
        "|                  1338464cd80cab681bfcfff3                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | failed                        |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: None seconds             |\n"
        "|                            | error: list indices must be   |\n"
        "|                            |        integers or slices,    |\n"
        "|                            |        not str                |\n"
        "--------------------------------------------------------------\n"
        "| results                    | not available yet             |\n"
        "--------------------------------------------------------------\n\n\n"
    )


def test_deepmemory_response_without_job_id():
    response = create_response()

    del response["dataset_id"]
    del response["id"]

    with pytest.raises(ValueError):
        response_schema = JobResponseStatusSchema(response=response)

    response["dataset_id"] = "some id"

    with pytest.raises(ValueError):
        response_schema = JobResponseStatusSchema(response=response)


def test_deepmemory_print_status_and_list_jobs(capsys, precomputed_jobs_list):
    # for training that is just started
    job_id = "1238464cd80cab681bfcfff3"
    pending_response = create_response(
        job_id=job_id,
        status="pending",
        progress=None,
    )
    response_schema = JobResponseStatusSchema(response=pending_response)
    response_schema.print_status(job_id)
    captured = capsys.readouterr()
    assert captured.out == Status.pending

    # for training that is in progress
    job_id = "3218464cd80cab681bfcfff3"
    training_response = create_response(job_id=job_id)
    response_schema = JobResponseStatusSchema(response=training_response)
    response_schema.print_status(job_id, recall="85.5", importvement="2.6")
    captured = capsys.readouterr()
    assert captured.out == Status.training

    # for training jobs that are finished
    job_id = "2138464cd80cab681bfcfff3"
    completed_response = create_response(
        job_id=job_id,
        status="completed",
    )
    response_schema = JobResponseStatusSchema(response=completed_response)
    response_schema.print_status(job_id, recall="85.5", importvement="2.6")
    captured = capsys.readouterr()
    assert captured.out == Status.completed

    # for jobs that failed
    job_id = "1338464cd80cab681bfcfff3"
    failed_response = create_response(
        job_id=job_id,
        status="failed",
        progress={
            "eta": None,
            "last_update_at": "2021-08-31T15:00:00.000000",
            "error": "list indices must be integers or slices, not str",
            "dataset": "query",
        },
    )
    response_schema = JobResponseStatusSchema(response=failed_response)
    response_schema.print_status(job_id)
    captured = capsys.readouterr()
    assert captured.out == Status.failed

    responses = [
        pending_response,
        training_response,
        completed_response,
        failed_response,
    ]
    recalls = {
        "1238464cd80cab681bfcfff3": None,
        "3218464cd80cab681bfcfff3": "85.5",
        "2138464cd80cab681bfcfff3": "85.5",
        "1338464cd80cab681bfcfff3": None,
    }
    improvements = {
        "1238464cd80cab681bfcfff3": None,
        "3218464cd80cab681bfcfff3": "2.6",
        "2138464cd80cab681bfcfff3": "2.6",
        "1338464cd80cab681bfcfff3": None,
    }
    response_schema = JobResponseStatusSchema(response=responses)
    output_str = response_schema.print_jobs(
        debug=True,
        recalls=recalls,
        improvements=improvements,
    )
    assert output_str == precomputed_jobs_list


@pytest.mark.slow
def test_deepmemory_train_and_cancel(job_id, capsys, hub_cloud_dev_token):
    client = DeepMemoryBackendClient(hub_cloud_dev_token)

    canceled = client.cancel_job(job_id="non-existent-job-id")
    captured = capsys.readouterr()
    expected = "Job with job_id='non-existent-job-id' was not cancelled!\n Error: Entity non-existent-job-id does not exist.\n"
    assert canceled == False
    assert captured.out == expected

    canceled = client.cancel_job(job_id=job_id)
    captured = capsys.readouterr()
    expected = (
        f"Job with job_id='{job_id}' was not cancelled!\n"
        f" Error: Job {job_id} is not in pending state, skipping cancellation.\n"
    )
    assert canceled == False
    assert expected in captured.out == expected


@pytest.mark.slow
def test_deepmemory_delete(
    capsys,
    hub_cloud_dev_credentials,
    corpus_query_relevances_copy,
    hub_cloud_dev_token,
):
    (
        corpus_path,
        _,
        _,
        _,
    ) = corpus_query_relevances_copy

    username, _ = hub_cloud_dev_credentials
    query_path = f"hub://{username}/deepmemory_test_queries_managed"
    client = DeepMemoryBackendClient(hub_cloud_dev_token)
    job = client.start_taining(
        corpus_path=corpus_path,
        queries_path=query_path,
    )
    client.cancel_job(job_id=job["job_id"])
    client.delete_job(job_id=job["job_id"])

    deleted = client.delete_job(job_id="non-existent-job-id")
    output_str = capsys.readouterr()
    expected = "Job with job_id='non-existent-job-id' was not deleted!\n Error: Entity non-existent-job-id does not exist.\n"
    assert deleted == False
    assert expected in output_str.out
