import sys
import deeplake
import pytest
from io import StringIO
from contextlib import contextmanager
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.exceptions import (
    AgreementNotAcceptedError,
    NotLoggedInAgreementError,
)


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def dont_agree(path, token: str):
    """Load the Deep Lake cloud dataset at path and simulate disagreeing to the terms of access."""

    with pytest.raises(AgreementNotAcceptedError):
        # this text can be anything except expected
        with replace_stdin(StringIO("no, i don't agree!")):
            deeplake.load(path, token=token)


def agree(path, token: str):
    """Load the Deep Lake cloud dataset at path and simulate agreeing to the terms of access."""
    dataset_name = path.split("/")[-1]
    with replace_stdin(StringIO(dataset_name)):
        ds = deeplake.load(path, token=token)
    ds.images[0].numpy()

    # next load should work without agreeing
    ds = deeplake.load(path, token=token)
    ds.images[0].numpy()


def reject(path, token: str):
    client = DeepLakeBackendClient(token=token)
    org_id, ds_name = path.split("/")[-2:]
    client.reject_agreements(org_id, ds_name)


def test_agreement_logged_out():
    path = "hub://activeloop/imagenet-test"
    with pytest.raises(NotLoggedInAgreementError):
        agree(path, token=None)


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
def test_agreement_logged_in(hub_cloud_dev_token):
    path = "hub://activeloop/imagenet-test"
    agree(path, hub_cloud_dev_token)
    reject(path, hub_cloud_dev_token)


@pytest.mark.flaky(reruns=3)
@pytest.mark.slow
def test_not_agreement_logged_in(hub_cloud_dev_token):
    path = "hub://activeloop/imagenet-test"
    dont_agree(path, hub_cloud_dev_token)
