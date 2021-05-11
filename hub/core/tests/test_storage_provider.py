from hub.core.storage import MemoryProvider, LocalProvider, S3Provider
from hub.core.storage.mapped_provider import MappedProvider
import pytest
from hub.util.check_s3_creds import s3_creds_exist
from hub.util.cache_chain import get_cache_chain

MB = 1024 * 1024


def test_storage_provider(provider=MappedProvider()):
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


def test_memory_provider():
    test_storage_provider(MemoryProvider("abc/def"))


def test_local_provider():
    test_storage_provider(LocalProvider("./hub_storage_local_test"))


@pytest.mark.skipif(not s3_creds_exist(), reason="requires s3 credentials")
def test_s3_provider():
    test_storage_provider(S3Provider("snark-test/hub_storage_s3_test"))


def speed_check(provider=MappedProvider()):
    chunk = b"0123456789123456" * MB
    for i in range(20):
        provider[f"file_{i}"] = chunk
    for i in range(20):
        provider[f"file_{i}"]
    for i in range(20):
        del provider[f"file_{i}"]


def test_speed_memory(benchmark):
    benchmark(speed_check, MemoryProvider("benchmark"))


def test_speed_local(benchmark):
    benchmark(speed_check, LocalProvider("./benchmark"))


def test_speed_s3(benchmark):
    benchmark(speed_check, S3Provider("snark-test/hub2/benchmark"))


def test_speed_lru_mem_local(benchmark):
    local_provider = LocalProvider("./benchmark")
    memory_provider = MemoryProvider("benchmark")
    s3_provider = S3Provider("snark-test/hub2/benchmark")
    lru_0 = get_cache_chain([memory_provider, local_provider], [32 * MB])
    benchmark(speed_check, lru_0)


def test_speed_lru_mem_s3(benchmark):
    local_provider = LocalProvider("./benchmark")
    memory_provider = MemoryProvider("benchmark")
    lru = get_cache_chain([memory_provider, s3_provider], [32 * MB])
    benchmark(speed_check, lru)


def test_speed_lru_local_s3(benchmark):
    local_provider = LocalProvider("./benchmark")
    s3_provider = S3Provider("snark-test/hub2/benchmark")
    lru = get_cache_chain([local_provider, s3_provider], [160 * MB])
    benchmark(speed_check, lru)


def test_speed_lru_local_mem_s3(benchmark):
    local_provider = LocalProvider("./benchmark")
    memory_provider = MemoryProvider("benchmark")
    s3_provider = S3Provider("snark-test/hub2/benchmark")
    lru = get_cache_chain(
        [memory_provider, local_provider, s3_provider],
        [32 * MB, 160 * MB],
    )
    benchmark(speed_check, lru)

    # local_provider = LocalProvider("./benchmark")
    # memory_provider = MemoryProvider("benchmark")
    # s3_provider = S3Provider("snark-test/hub2/benchmark")

    # benchmark(write_speed, memory_provider)
    # benchmark(read_speed, memory_provider)
    # benchmark(delete_speed, memory_provider)

    # benchmark(write_speed, local_provider)
    # benchmark(read_speed, local_provider)
    # benchmark(delete_speed, local_provider)

    # benchmark(write_speed, s3_provider)
    # benchmark(read_speed, s3_provider)
    # benchmark(delete_speed, s3_provider)

    # lru_0 = get_cache_chain([memory_provider, local_provider], [32 * MB])
    # benchmark(write_speed, lru_0)
    # benchmark(read_speed, lru_0)
    # benchmark(delete_speed, lru_0)

    # lru_1 = get_cache_chain([memory_provider, s3_provider], [32 * MB])
    # benchmark(write_speed, lru_1)
    # benchmark(read_speed, lru_1)
    # benchmark(delete_speed, lru_1)

    # lru_2 = get_cache_chain([local_provider, s3_provider], [160 * MB])
    # benchmark(write_speed, lru_2)
    # benchmark(read_speed, lru_2)
    # benchmark(delete_speed, lru_2)

    # lru_3 = get_cache_chain(
    #     [memory_provider, local_provider, s3_provider],
    #     [32 * MB, 160 * MB],
    # )
    # benchmark(write_speed, lru_3)
    # benchmark(read_speed, lru_3)
    # benchmark(delete_speed, lru_3)


# TODO add pytest benchmarks
