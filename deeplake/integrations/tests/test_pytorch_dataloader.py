import pytest
import platform
from deeplake.util.check_installation import pytorch_installed

if not pytorch_installed():
    pytest.skip("pytroch is not installed", allow_module_level=True)

if platform.system() in ["Windows", "Darwin"]:
    pytest.skip("mock pickling gets quite messy on win/mac", allow_module_level=True)

from unittest.mock import patch

from deeplake.core.io import SampleStreaming, IOBlock, Schedule
from deeplake.integrations.pytorch.dataset import (
    ShufflingIterableDataset,
    SubIterableDataset,
)
from deeplake.integrations.pytorch.common import collate_fn as default_collate_fn
from deeplake.util.dataset import map_tensor_keys
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
@patch("deeplake.core.dataset.Dataset")
def test_dataloader(ds):
    tensors = map_tensor_keys(ds)
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=2, tensors=tensors
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("deeplake.core.dataset.Dataset")
def test_dataloader_batching(ds):
    tensors = map_tensor_keys(ds)
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=2, tensors=tensors
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == 4


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("deeplake.core.dataset.Dataset")
def test_more_workers_than_chunk(ds):
    tensors = map_tensor_keys(ds)
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=4, tensors=tensors
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("deeplake.core.dataset.Dataset")
def test_big_buffer_size(ds):
    tensors = map_tensor_keys(ds)
    dataset = ShufflingIterableDataset(
        mock_dataset(ds),
        use_local_cache=False,
        num_workers=4,
        buffer_size=512,
        tensors=tensors,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


def mock_tranform_f(data):
    return {"items": torch.tensor(42)}


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("deeplake.core.dataset.Dataset")
def test_workers_transform(ds):
    tensors = map_tensor_keys(ds)
    dataset = ShufflingIterableDataset(
        mock_dataset(ds),
        use_local_cache=False,
        num_workers=4,
        transform=mock_tranform_f,
        tensors=tensors,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    for x in result:
        assert x["items"] == 42


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", throws_exception)
@patch("deeplake.core.dataset.Dataset")
def test_proppagete_exception(ds):
    tensors = map_tensor_keys(ds)
    dataset = ShufflingIterableDataset(
        mock_dataset(ds), use_local_cache=False, num_workers=1, tensors=tensors
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    with pytest.raises(RuntimeError):
        list(dataloader)


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
@patch("deeplake.core.dataset.Dataset")
def test_method2(ds):
    tensors = map_tensor_keys(ds)
    dataset = SubIterableDataset(
        mock_dataset(ds),
        use_local_cache=False,
        num_workers=2,
        batch_size=2,
        tensors=tensors,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])
