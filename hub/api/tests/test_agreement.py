import sys
import hub
import pytest
from io import StringIO
from contextlib import contextmanager
from click.testing import CliRunner
from hub.cli.auth import login, logout
from hub.client.client import HubBackendClient
from hub.util.exceptions import AgreementNotAcceptedError, NotLoggedInAgreementError


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def dont_agree(path):
    """Load the hub cloud dataset at path and simulate disagreeing to the terms of access."""

    with pytest.raises(AgreementNotAcceptedError):
        # this text can be anything except expected
        with replace_stdin(StringIO("no, i don't agree!")):
            hub.load(path)


def agree(path):
    """Load the hub cloud dataset at path and simulate agreeing to the terms of access."""
    dataset_name = path.split("/")[-1]
    with replace_stdin(StringIO(dataset_name)):
        ds = hub.load(path)
    ds.images[0].numpy()

    # next load should work without agreeing
    ds = hub.load(path)
    ds.images[0].numpy()


def reject(path):
    client = HubBackendClient()
    org_id, ds_name = path.split("/")[-2:]
    client.reject_agreements(org_id, ds_name)


def test_agreement_logged_out(hub_cloud_dev_credentials):
    runner = CliRunner()
    runner.invoke(logout)
    path = "hub://activeloop/imagenet-test"
    with pytest.raises(NotLoggedInAgreementError):
        agree(path)


def test_agreement_logged_in(hub_cloud_dev_credentials):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials
    runner.invoke(login, f"-u {username} -p {password}")
    path = "hub://activeloop/imagenet-test"
    agree(path)
    reject(path)
    runner.invoke(logout)


def test_not_agreement_logged_in(hub_cloud_dev_credentials):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials
    runner.invoke(login, f"-u {username} -p {password}")
    path = "hub://activeloop/imagenet-test"
    dont_agree(path)
    runner.invoke(logout)
