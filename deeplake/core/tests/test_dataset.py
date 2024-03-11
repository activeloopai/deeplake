import os
import traceback

import numpy as np
import pytest

import deeplake
from deeplake.client.config import DEEPLAKE_AUTH_TOKEN
from deeplake.core import LRUCache
from deeplake.core.storage.memory import MemoryProvider

from deeplake.core.dataset import Dataset
from deeplake.util.exceptions import (
    TensorMetaInvalidHtypeOverwriteValue,
    SampleAppendError,
)


def test_token_and_username(hub_cloud_dev_token):
    assert DEEPLAKE_AUTH_TOKEN not in os.environ

    ds = Dataset(
        storage=LRUCache(
            cache_storage=MemoryProvider(), cache_size=0, next_storage=MemoryProvider()
        )
    )
    assert ds.token is None
    assert ds.username == "public"

    # invalid tokens come through as "public"
    ds = Dataset(
        token="invalid_value",
        storage=LRUCache(
            cache_storage=MemoryProvider(), cache_size=0, next_storage=MemoryProvider()
        ),
    )
    assert ds.token == "invalid_value"
    assert ds.username == "public"

    # valid tokens come through correctly
    ds = Dataset(
        token=hub_cloud_dev_token,
        storage=LRUCache(
            cache_storage=MemoryProvider(), cache_size=0, next_storage=MemoryProvider()
        ),
    )
    assert ds.token == hub_cloud_dev_token
    assert ds.username == "testingacc2"

    # When env is set, it takes precedence over None for the token but not over a set token
    try:
        os.environ[DEEPLAKE_AUTH_TOKEN] = hub_cloud_dev_token
        ds = Dataset(
            storage=LRUCache(
                cache_storage=MemoryProvider(),
                cache_size=0,
                next_storage=MemoryProvider(),
            )
        )
        assert ds.token == hub_cloud_dev_token
        assert ds.username == "testingacc2"

        ds = Dataset(
            token="invalid_value",
            storage=LRUCache(
                cache_storage=MemoryProvider(),
                cache_size=0,
                next_storage=MemoryProvider(),
            ),
        )
        assert ds.token == "invalid_value"
        assert ds.username == "public"

    finally:
        os.environ.pop(DEEPLAKE_AUTH_TOKEN)

    assert DEEPLAKE_AUTH_TOKEN not in os.environ
