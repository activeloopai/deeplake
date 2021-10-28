import pytest
import hub
from hub.tests.common import assert_array_lists_equal
import sys
from contextlib import contextmanager
from io import StringIO
from hub.util.exceptions import UnagreedTermsOfAccessError


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def dont_agree(path):
    """Load the hub cloud dataset at path and simulate disagreeing to the terms of access."""

    with pytest.raises(UnagreedTermsOfAccessError):
        # this text can be anything except expected
        with replace_stdin(StringIO("no, i don't agree!")):
            hub.load(path)


def agree(path):
    """Load the hub cloud dataset at path and simulate agreeing to the terms of access."""

    dataset_name = path.split("/")[-1]
    with replace_stdin(StringIO(dataset_name)):
        ds = hub.load(path)
    assert_array_lists_equal(ds.test.numpy(), [[1, 2, 3]])


def test_creator_has_access(hub_cloud_ds_generator):
    gen = hub_cloud_ds_generator
    ds = gen()
    path = ds.path

    ds.add_terms_of_access("only nerdz allowed")
    ds.create_tensor("test")
    ds.test.append([1, 2, 3])

    # access granted by default since this is the creator of the dataset
    ds = gen()
    assert_array_lists_equal(ds.test.numpy(), [[1, 2, 3]])

    # revoke access
    ds.client._respond_to_terms_of_access(ds.org_id, ds.ds_name, "disagree")
    del ds

    # access NOT granted
    dont_agree(path)

    # ds = gen()
    agree(path)
