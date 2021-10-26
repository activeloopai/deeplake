from unittest.mock import patch, Mock
from hub.core.io import SampleStreaming, IOBlock, Schedule
from hub.integrations.pytorch.dataset import ShufflingIterableDataset
from hub.integrations.pytorch.common import collate_fn as default_collate_fn

from torch.utils.data import DataLoader

import pytest
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
    for block in schedule:
        for i in block.indices():
            yield {"images": numpy.ones((5)) * i}

    raise RuntimeError("test error")


def mock_dataset():
    return Mock(tensors={}, storage=None)


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
def test_dataloader():
    dataset = ShufflingIterableDataset(
        mock_dataset(), use_local_cache=False, num_workers=2
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
def test_dataloader_batching():
    dataset = ShufflingIterableDataset(
        mock_dataset(), use_local_cache=False, num_workers=2
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == 4


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", emit_samples)
def test_more_workers_than_chunk():
    dataset = ShufflingIterableDataset(
        mock_dataset(), use_local_cache=False, num_workers=4
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    result = list(dataloader)

    assert len(result) == sum([len(block) for block in list_blocks()])


@patch.object(SampleStreaming, "list_blocks", list_blocks)
@patch.object(SampleStreaming, "read", throws_exception)
def test_proppagete_exception():
    dataset = ShufflingIterableDataset(
        mock_dataset(), use_local_cache=False, num_workers=1
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=default_collate_fn)

    with pytest.raises(RuntimeError):
        list(dataloader)
