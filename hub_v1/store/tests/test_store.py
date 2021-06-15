"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.store.store import get_cache_path


def test_get_cache_path():
    cache_folder = "./cache/"
    assert "./cache/test/testdb" == get_cache_path("s3://test/testdb", cache_folder)
    assert "./cache/test/testdb" == get_cache_path("gcs://test/testdb", cache_folder)
    assert "./cache/test/testdb" == get_cache_path("test/testdb", cache_folder)
    assert "./cache/test/testdb" == get_cache_path("https://test/testdb", cache_folder)
    assert "./cache/test/testdb" == get_cache_path("~/test/testdb", cache_folder)
    assert "./cache/test/testdb" == get_cache_path("/test/testdb", cache_folder)
    assert "./cache/test\\testdb" == get_cache_path("C:\\test\\testdb", cache_folder)


if __name__ == "__main__":
    test_get_cache_path()
