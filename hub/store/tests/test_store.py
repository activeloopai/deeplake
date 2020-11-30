from hub.store.store import get_cache_path


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
