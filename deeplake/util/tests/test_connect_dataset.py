import pytest
from deeplake.util.connect_dataset import (
    DsInfo,
    is_path_connectable,
)
from deeplake.util.exceptions import InvalidDestinationPathError


def test_source_and_destination_paths():
    assert not is_path_connectable("hub://org_id/ds_name", allow_local=False)
    assert is_path_connectable("s3://bucket/path/to/dataset", allow_local=False)

    ds_info = DsInfo(dest_path="hub://org_id/ds_name")
    ds_info.validate()
    org_id, ds_name = ds_info.get_org_id_and_ds_name()
    assert org_id == "org_id"
    assert ds_name == "ds_name"

    ds_info = DsInfo(org_id="another_org")
    ds_info.validate()
    org_id, ds_name = ds_info.get_org_id_and_ds_name()
    assert org_id == "another_org"
    assert ds_name is None

    ds_info = DsInfo(org_id="yet_another_org", ds_name="some_name")
    ds_info.validate()
    org_id, ds_name = ds_info.get_org_id_and_ds_name()
    assert org_id == "yet_another_org"
    assert ds_name == "some_name"

    ds_info = DsInfo(org_id="org_id", ds_name="ds_name", dest_path="does_not_matter")
    ds_info.validate()
    org_id, ds_name = ds_info.get_org_id_and_ds_name()
    assert org_id == "org_id"
    assert ds_name == "ds_name"

    ds_info = DsInfo(
        dest_path="hub://another_org/some_dataset", ds_name="does_not_matter"
    )
    ds_info.validate()
    org_id, ds_name = ds_info.get_org_id_and_ds_name()
    assert org_id == "another_org"
    assert ds_name == "some_dataset"

    with pytest.raises(InvalidDestinationPathError):
        ds_info = DsInfo()
        ds_info.validate()  # Make sure at least org_id is required

    with pytest.raises(InvalidDestinationPathError):
        ds_info = DsInfo(dest_path="s3://bucket/dataset")
        ds_info.validate()  # Make sure that path can only be a Deep Lake cloud path.
