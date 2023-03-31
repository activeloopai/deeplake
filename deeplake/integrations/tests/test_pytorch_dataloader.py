import pytest
import platform
from deeplake.util.check_installation import pytorch_installed

if not pytorch_installed():
    pytest.skip("pytroch is not installed", allow_module_level=True)

if platform.system() in ["Windows", "Darwin"]:
    pytest.skip("mock pickling gets quite messy on win/mac", allow_module_level=True)

from unittest.mock import patch

from deeplake.core.io import SampleStreaming, IOBlock, Schedule
from deeplake.integrations.pytorch.dataset import SubIterableDataset
from deeplake.integrations.pytorch.common import collate_fn as default_collate_fn
from deeplake.util.dataset import map_tensor_keys
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
            yield {"images": numpy.ones((5)) * i, "index": numpy.array([i])}


def throws_exception(streaming, schedule: Schedule):
    yield from emit_samples(streaming, schedule)
    raise RuntimeError("test error")


def mock_dataset(cls):
    instance = cls()
    return instance


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
