import pytest
import platform
from hub.constants import MB
from hub.util.check_installation import pytorch_installed

if not pytorch_installed():
    pytest.skip("pytroch is not installed", allow_module_level=True)

if platform.system() in ["Windows", "Darwin"]:
    pytest.skip("mock pickling gets quite messy on win/mac", allow_module_level=True)

from unittest.mock import patch

from hub.core.io import SampleStreaming, IOBlock, Schedule
from hub.integrations.pytorch.dataset import (
    ShufflingIterableDataset,
    SubIterableDataset,
)
from hub.integrations.pytorch.common import (
    collate_fn as default_collate_fn,
)

import torch
from torch.utils.data.dataloader import DataLoader
import numpy


def list_blocks(streaming=None):
    return [
        IOBlock(chunks=["a", "b"], indexes=[1, 2, 3]),
        IOBlock(chunks=["c", "b"], indexes=[4, 5, 6]),
        IOBlock(chunks=["d", "e"], indexes=[7]),
    ]


def emit_samples(streaming, schedule: Schedule):
    for block in schedule:
        for i in block.indices():
            yield {"images": numpy.ones((5)) * i}


def throws_exception(streaming, schedule: Schedule):
    yield from emit_samples(streaming, schedule)
    raise RuntimeError("test error")


def mock_dataset(cls):
    instance = cls()
    return instance


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("hub.core.dataset.Dataset")
def test_dataloader(ds):
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=2
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("hub.core.dataset.Dataset")
def test_dataloader_batching(ds):
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=2
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == 4


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("hub.core.dataset.Dataset")
def test_more_workers_than_chunk(ds):
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=4
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("hub.core.dataset.Dataset")
def test_big_buffer_size(ds):
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=4, buffer_size=512
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


def mock_tranform_f(data):
    return {"items": torch.tensor(42)}


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("hub.core.dataset.Dataset")
def test_workers_transform(ds):
    dataset = ShufflingIterableDataset(
        mock_dataset(ds),
        use_local_cache=False,
        num_workers=4,
        transform=mock_tranform_f,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    for x in result:
        assert x["items"] == 42


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", throws_exception)
@patch("hub.core.dataset.Dataset")
def test_proppagete_exception(ds):
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=1
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    with pytest.raises(RuntimeError):
        list(dataloader)


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("hub.core.dataset.Dataset")
def test_method2(ds):
    dataset = SubIterableDataset(
        mock_dataset(ds),
        use_local_cache=False,
        num_workers=2,
        batch_size=2,
        collate_fn=default_collate_fn,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])
