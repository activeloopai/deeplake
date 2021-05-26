# Contributing Standards

## Linting
We use the [black](https://pypi.org/project/black/) python linter. You can have your code auto-formatted by running `pip install black`, then `black .` inside the directory you want to format.

## Docstrings
We use Google Docstrings. Please refer to [this example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Typing
We also use static typing for our function arguments/variables for better code readability. We have a github action that runs `mypy .`, which runs similar to `pytest .` to check for valid static typing. You can refer to [mypy documentation](https://mypy.readthedocs.io/en/stable/) for more information.

## Testing
We use [pytest](https://docs.pytest.org/en/6.2.x/) for our tests. In order to make it easier, we also have a set of custom options defined in [conftest.py](conftest.py).


### Running Tests
- To run memory only tests, use: `python -m pytest .`.
- To run local only tests, use: `python -m pytest . --memory-skip --local`.
- To run s3 only tests, use: `python -m pytest . --memory-skip --s3`.
- To run ALL tests, use: `python -m pytest . --local --s3 --cache-chains`.


### Prerequisites
- Understand how to write [pytest](https://docs.pytest.org/en/6.2.x/) tests.
- Understand what a [pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html) is.
- Understand what [pytest parametrizations](https://docs.pytest.org/en/6.2.x/parametrize.html) are.


### Options
To see a list of our custom pytest options, run this command: `pytest -h | sed -En '/custom options:/,/\[pytest\] ini\-options/p'`.

### Fixtures
You can find more information on pytest fixtures [here](https://docs.pytest.org/en/6.2.x/fixture.html).

- `memory_storage`: If `--memory-skip` is provided, tests with this fixture will be skipped. Otherwise, the test will run with only a `MemoryProvider`.
- `local_storage`: If `--local` is **not** provided, tests with this fixture will be skipped. Otherwise, the test will run with only a `LocalProvider`.
- `s3_storage`: If `--s3` is **not** provided, tests with this fixture will be skipped. Otherwise, the test will run with only an `S3Provider`.
- `storage`: All tests that use the `storage` fixture will be parametrized with the enabled `StorageProvider`s (enabled via options defined below). If `--cache-chains` is provided, `storage` may also be a cache chain. Cache chains have the same interface as `StorageProvider`, but instead of just a single provider, it is multiple chained in a sequence, where the last provider in the chain is considered the actual storage.
- `ds`: The same as the `storage` fixture, but the storages that are parametrized are wrapped with a `Dataset`.

Each `StorageProvider`/`Dataset` that is created for a test via a fixture will automatically have a root created before and destroyed after the test. If you want to keep this data after the test run, you can use the `--keep-storage` option. 


#### Fixture Examples


Single storage provider fixture
```python
def test_memory(memory_storage):
    # test will skip if `--memory-skip` is provided
    memory_storage["key"] = b"1234"  # this data will only be stored in memory

def test_local(local_storage):
    # test will skip if `--local` is not provided
    memory_storage["key"] = b"1234"  # this data will only be stored locally

def test_local(s3_storage):
    # test will skip if `--s3` is not provided
    # test will fail if credentials are not provided
    memory_storage["key"] = b"1234"  # this data will only be stored in s3
```

Multiple storage providers/cache chains
```python
from hub.core.tests.common import parametrize_all_storages, parametrize_all_caches, parametrize_all_storages_and_caches

@parametrize_all_storages
def test_storage(storage):
    # storage will be parametrized with all enabled `StorageProvider`s
    pass

@parametrize_all_caches
def test_caches(storage):
    # storage will be parametrized with all common caches containing enabled `StorageProvider`s
    pass

@parametrize_all_storages_and_caches
def test_storages_and_caches(storage):
    # storage will be parametrized with all enabled `StorageProvider`s and common caches containing enabled `StorageProvider`s
    pass
```


Dataset storage providers/cache chains
```python
from hub.core.tests.common import parametrize_all_dataset_storages, parametrize_all_dataset_storages_and_caches

@parametrize_all_dataset_storages
def test_dataset(ds):
    # `ds` will be parametrized with 1 `Dataset` object per enabled `StorageProvider`
    pass

@parametrize_all_dataset_storages_and_caches
def test_dataset(ds):
    # `ds` will be parametrized with 1 `Dataset` object per enabled `StorageProvider` and all cache chains containing enabled `StorageProvider`s
    pass
```


## Benchmarks
We use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/usage.html) for our benchmark code which is a plugin for [pytest](https://docs.pytest.org/en/6.2.x/).

### Running Benchmarks
- To run benchmarks for memory only, use: `python -m pytest . --benchmark-only`.
- To run ALL **fast** benchmarks, use: `python -m pytest . --local --s3 --cache-chains --benchmark-only`. Note: this only runs the subset of benchmarks that finish quickly.
- To run ALL **fast AND slow** benchmarks, use: `python -m pytest . --local --s3 --full-benchmarks --benchmark-only`. Note: this will take a while... (also cache chains are implicitly enabled from `--full-benchmarks`.)
- You can opt out of `--local` and `--s3` for all commands, or add `--memory-skip`.
- Optionally, you can remove the `--benchmark-only` flag in any of these commands to run normal tests alongside the benchmarks.


TODO: benchmarking is subject to change. will update this section once it is better defined.
