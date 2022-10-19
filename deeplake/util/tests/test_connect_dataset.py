import pytest
from deeplake.util.connect_dataset import _get_org_id_and_ds_name
from deeplake.util.exceptions import InvalidDestinationPathError


def test_destination_path():
    org_id, ds_name = _get_org_id_and_ds_name(dest_path="hub://org_id/ds_name")
    assert org_id == "org_id"
    assert ds_name == "ds_name"

    org_id, ds_name = _get_org_id_and_ds_name(org_id="another_org")
    assert org_id == "another_org"
    assert ds_name is None

    org_id, ds_name = _get_org_id_and_ds_name(
        org_id="yet_another_org", ds_name="some_name"
    )
    assert org_id == "yet_another_org"
    assert ds_name == "some_name"

    org_id, ds_name = _get_org_id_and_ds_name(
        org_id="org_id", ds_name="ds_name", dest_path="does_not_matter"
    )
    assert org_id == "org_id"
    assert ds_name == "ds_name"

    org_id, ds_name = _get_org_id_and_ds_name(
        dest_path="hub://another_org/some_dataset", ds_name="does_not_matter"
    )
    assert org_id == "another_org"
    assert ds_name == "some_dataset"

    with pytest.raises(InvalidDestinationPathError):
        _get_org_id_and_ds_name()  # Make sure at least org_id is required

    with pytest.raises(InvalidDestinationPathError):
        _get_org_id_and_ds_name(
            dest_path="s3://bucket/dataset"
        )  # Make sure that path can only be a Deep Lake cloud path
