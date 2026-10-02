import importlib.util
import sys
import types
from pathlib import Path


def _load_get_indexes(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__path__ = []
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: True
    distributed.get_world_size = lambda: 1
    distributed.get_rank = lambda: 0
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)

    source = (
        Path(__file__).resolve().parents[1]
        / "deeplake"
        / "integrations"
        / "mm"
        / "get_indexes.py"
    )
    spec = importlib.util.spec_from_file_location("get_indexes_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_indexes


def test_non_dropping_slices_have_equal_lengths_and_cover_all_rows(monkeypatch):
    get_indexes = _load_get_indexes(monkeypatch)
    dataset = range(5)

    slices = [
        get_indexes(dataset, rank=rank, num_replicas=2, drop_last=False)
        for rank in range(2)
    ]

    assert [len(range(s.start, s.stop)) for s in slices] == [3, 3]
    assigned = [index for s in slices for index in range(s.start, s.stop)]
    assert set(assigned) == set(dataset)


def test_drop_last_still_drops_the_remainder(monkeypatch):
    get_indexes = _load_get_indexes(monkeypatch)

    slices = [
        get_indexes(range(5), rank=rank, num_replicas=2, drop_last=True)
        for rank in range(2)
    ]

    assert [(s.start, s.stop) for s in slices] == [(0, 2), (2, 4)]
