from hub.core.storage import MemoryProvider, LocalProvider, S3Provider
from hub.util.s3 import has_s3_creds
from hub.util.cache_chain import get_cache_chain
import pytest

NUM_FILES = 20
MB = 1024 * 1024

local_provider = LocalProvider("./test/benchmark")
memory_provider = MemoryProvider("test/benchmark")
s3_provider = S3Provider("snark-test/hub2/benchmark")


def check_storage_provider(provider):
    FILE_1 = "abc.txt"
    FILE_2 = "def.txt"

    provider[FILE_1] = b"hello world"
    assert provider[FILE_1] == b"hello world"
    assert provider.get_bytes(FILE_1, 2, 5) == b"llo"

    provider.set_bytes(FILE_1, b"abcde", 6)
    assert provider[FILE_1] == b"hello abcde"

    provider.set_bytes(FILE_1, b"tuvwxyz", 6)
    assert provider[FILE_1] == b"hello tuvwxyz"

    provider.set_bytes(FILE_2, b"hello world", 3)
    assert provider[FILE_2] == b"\x00\x00\x00hello world"
    provider.set_bytes(FILE_2, b"new_text", overwrite=True)
    assert provider[FILE_2] == b"new_text"

    assert len(provider) >= 1

    for _ in provider:
        pass

    del provider[FILE_1]
    del provider[FILE_2]

    with pytest.raises(KeyError):
        provider[FILE_1]

    provider.flush()


def test_memory_provider():
    check_storage_provider(memory_provider)


def test_local_provider():
    check_storage_provider(local_provider)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_s3_provider():
    check_storage_provider(s3_provider)


def test_lru_mem_local():
    lru = get_cache_chain([memory_provider, local_provider], [32 * MB])
    check_storage_provider(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_lru_mem_s3():
    lru = get_cache_chain([memory_provider, s3_provider], [32 * MB])
    check_storage_provider(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_lru_local_s3():
    lru = get_cache_chain([local_provider, s3_provider], [160 * MB])
    check_storage_provider(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_lru_mem_local_s3():
    lru = get_cache_chain(
        [memory_provider, local_provider, s3_provider],
        [32 * MB, 160 * MB],
    )
    check_storage_provider(lru)


def write_to_files(provider):
    chunk = b"0123456789123456" * MB
    for i in range(NUM_FILES):
        provider[f"file_{i}"] = chunk
    provider.flush()


def read_from_files(provider):
    for i in range(NUM_FILES):
        provider[f"file_{i}"]


def delete_files(provider):
    for i in range(NUM_FILES):
        del provider[f"file_{i}"]


def test_write_memory(benchmark):
    benchmark(write_to_files, memory_provider)
    delete_files(memory_provider)


def test_write_local(benchmark):
    benchmark(write_to_files, local_provider)
    delete_files(local_provider)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_write_s3(benchmark):
    benchmark(write_to_files, s3_provider)
    delete_files(s3_provider)


def test_write_lru_mem_local(benchmark):
    lru = get_cache_chain([memory_provider, local_provider], [32 * MB])
    benchmark(write_to_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_write_lru_mem_s3(benchmark):
    lru = get_cache_chain([memory_provider, s3_provider], [32 * MB])
    benchmark(write_to_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_write_lru_local_s3(benchmark):
    lru = get_cache_chain([local_provider, s3_provider], [160 * MB])
    benchmark(write_to_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_write_lru_mem_local_s3(benchmark):
    lru = get_cache_chain(
        [memory_provider, local_provider, s3_provider],
        [32 * MB, 160 * MB],
    )
    benchmark(write_to_files, lru)
    delete_files(lru)


def test_read_memory(benchmark):
    write_to_files(memory_provider)
    benchmark(read_from_files, memory_provider)
    delete_files(memory_provider)


def test_read_local(benchmark):
    write_to_files(local_provider)
    benchmark(read_from_files, local_provider)
    delete_files(local_provider)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_read_s3(benchmark):
    write_to_files(s3_provider)
    benchmark(read_from_files, s3_provider)
    delete_files(s3_provider)


def test_read_lru_mem_local(benchmark):
    write_to_files(local_provider)
    lru = get_cache_chain([memory_provider, local_provider], [32 * MB])
    benchmark(read_from_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_read_lru_mem_s3(benchmark):
    write_to_files(s3_provider)
    lru = get_cache_chain([memory_provider, s3_provider], [32 * MB])
    benchmark(read_from_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_read_lru_local_s3(benchmark):
    write_to_files(s3_provider)
    lru = get_cache_chain([local_provider, s3_provider], [160 * MB])
    benchmark(read_from_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_read_lru_mem_local_s3(benchmark):
    write_to_files(s3_provider)
    lru = get_cache_chain(
        [memory_provider, local_provider, s3_provider],
        [32 * MB, 160 * MB],
    )
    benchmark(read_from_files, lru)
    delete_files(lru)


def test_full_cache_read_lru_mem_local(benchmark):
    write_to_files(local_provider)
    lru = get_cache_chain([memory_provider, local_provider], [320 * MB])
    read_from_files(lru)
    benchmark(read_from_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_full_cache_read_lru_mem_s3(benchmark):
    write_to_files(s3_provider)
    lru = get_cache_chain([memory_provider, s3_provider], [320 * MB])
    read_from_files(lru)
    benchmark(read_from_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_full_cache_read_lru_local_s3(benchmark):
    write_to_files(s3_provider)
    lru = get_cache_chain([local_provider, s3_provider], [320 * MB])
    read_from_files(lru)
    benchmark(read_from_files, lru)
    delete_files(lru)


@pytest.mark.skipif(not has_s3_creds(), reason="requires s3 credentials")
def test_full_cache_read_lru_mem_local_s3(benchmark):
    write_to_files(s3_provider)
    lru = get_cache_chain(
        [memory_provider, local_provider, s3_provider],
        [32 * MB, 320 * MB],
    )
    read_from_files(lru)
    benchmark(read_from_files, lru)
    delete_files(lru)
