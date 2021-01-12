import pytest
import cloudpickle

from hub.store.s3_file_system_replacement import S3FileSystemReplacement
from hub.utils import s3_creds_exist

DATA = bytes("YES YES YES!", "utf-8")
KEY = "pickable"


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_file_system_replacement_pickability(
    url: str = "snark-test/test_s3_file_system_replacement_pickability",
):
    storage = S3FileSystemReplacement()
    fsmap = storage.get_mapper(url)
    fsmap[KEY] = DATA
    dumped_storage = cloudpickle.dumps(storage)
    loaded_storage = cloudpickle.loads(dumped_storage)
    loaded_fsmap = loaded_storage.get_mapper(url)
    assert loaded_fsmap[KEY] == DATA