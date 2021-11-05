import sys
import hub
import pytest
from io import StringIO
from contextlib import contextmanager
from click.testing import CliRunner
from hub.cli.auth import login, logout
from hub.util.agreement import get_all_local_agreements, update_local_agreements
from hub.util.exceptions import AgreementNotAcceptedError, NotLoggedInError


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
    ds.labels[0].numpy()


def remove_agreement(username, path):
    """Removes the agreement for path from the locally stored info."""
    all_local_agreements = get_all_local_agreements()
    agreement_set = all_local_agreements.get(username) or set()
    agreement_set.discard(path)
    all_local_agreements[username] = agreement_set
    update_local_agreements(all_local_agreements)


def test_agreement_logged_out(hub_cloud_dev_credentials):
    runner = CliRunner()
    runner.invoke(logout)
    path = "hub://activeloop/imagenet-train"
    with pytest.raises(NotLoggedInError):
        agree(path)


def test_agreement_logged_in(hub_cloud_dev_credentials):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials
    runner.invoke(login, f"-u {username} -p {password}")
    path = "hub://activeloop/imagenet-train"
    agree(path)
    runner.invoke(logout)
    remove_agreement(username, path)


def test_not_agreement_logged_in(hub_cloud_dev_credentials):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials
    runner.invoke(login, f"-u {username} -p {password}")
    path = "hub://activeloop/imagenet-train"
    dont_agree(path)
    runner.invoke(logout)
