# Contributing Standards

## Linting

We use the [black](https://pypi.org/project/black/) python linter. You can have your code auto-formatted by
running `pip install black`, then `black .` inside the directory you want to format.

## Docstrings

We use Google Docstrings. Please refer
to [this example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Typing
We also use static typing for our function arguments/variables for better code readability. We have a github action that runs `mypy .`, which runs similar to `pytest .` to check for valid static typing. You can refer to [mypy documentation](https://mypy.readthedocs.io/en/stable/) for more information.

## Testing
We use [pytest](https://docs.pytest.org/en/6.2.x/) for our tests. In order to make it easier, we also have a set of custom options defined in [conftest.py](conftest.py).

### To install all dependencies run:

```
pip3 install -r deeplake/requirements/common.txt
pip3 install -r deeplake/requirements/plugins.txt
pip3 install -r deeplake/requirements/tests.txt
```


### Running Tests

#### Standard:
- `pytest .`: Run all tests with memory only.
- `pytest . --local`: Run all tests with memory and local.
- `pytest . --s3`: Run all tests with memory and s3.
- `pytest . --gcs`: Run all tests with memory and GCS 
- `pytest . --kaggle`: Run all tests that use the kaggle API.
- `pytest . --memory-skip --hub-cloud`: Run all tests with hub cloud only.
#### Backwards Compatibility Tests
We use another github repository ([buH](https://github.com/activeloopai/buH)) for our backwards compatibility tests. Check out the README for instructions.

### Options
Combine any of the following options to suit your test cases.

- `--local`: Enable local tests.
- `--s3`: Enable S3 tests.
- `--gcs`: Enable GCS tests.
- `--hub-cloud`: Enable hub cloud tests.
- `--memory-skip`: Disable memory tests.
- `--s3-path`: Specify an s3 path if you don't have access to our internal testing bucket.
- `--keep-storage`: By default all storages are cleaned up after tests run. Enable this option if you need to check the storage contents. Note: `--keep-storage` does not keep memory tests storage.


### Extra Resources
If you feel lost with any of these sections, try reading up on some of these topics.

- Understand how to write [pytest](https://docs.pytest.org/en/6.2.x/) tests.
- Understand what a [pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html) is.
- Understand what [pytest parametrizations](https://docs.pytest.org/en/6.2.x/parametrize.html) are.


### Fixture Usage Examples
These are not all of the available fixtures. You can see all of them [here](/deeplake/tests/).

Datasets
```python
@enabled_datasets
def test_dataset(ds: Dataset):
  # this test will run once per enabled storage provider. if no providers are explicitly enabled,
  # only memory will be used.
  pass


def test_local_dataset(local_ds: Dataset):
  # this test will run only once with a local dataset. if the `--local` option is not provided,
  # this test will be skipped.
  pass
```

Storages
```python
@enabled_storages
def test_storage(storage: StorageProvider):
  # this test will run once per enabled storage provider. if no providers are explicitly enabled,
  # only memory will be used.
  pass


def test_memory_storage(memory_storage: StorageProvider):
  # this test will run only once with a memory storage provider. if the `--memory-skip` option is provided,
  # this test will be skipped.
  pass
```

Caches
```python
@enabled_cache_chains
def test_cache(cache_chain: StorageProvider):  # note: caches are provided as `StorageProvider`s
  # this test runs for every cache chain that contains all enabled storage providers.
  # if only memory is enabled (no providers are explicitly enabled), this test will be skipped.
  pass
```

## Generating API Docs

Deep Lake used pdocs3 to generate docs: https://pdoc3.github.io/pdoc/
API docs are hosted at: https://api-docs.activeloop.ai/

Run the below command to generate API documentation:
```
  pdoc3 --html --output-dir api_docs --template-dir pdoc/templates hub
```