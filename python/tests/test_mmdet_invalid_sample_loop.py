"""Import the real wrapper normally; isolate unavailable training dependencies.

These exercise wrapper retry behavior, not a full MMDet/Deep Lake training run.
The real exception and array helper modules are imported from this source tree.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def wrapper_module(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "deeplake"
    for name in list(sys.modules):
        if name == "deeplake" or name.startswith("deeplake."):
            monkeypatch.delitem(sys.modules, name)

    def module(name, package_path=None, **attrs):
        value = ModuleType(name)
        if package_path is not None:
            value.__path__ = [str(package_path)]
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)

    module("deeplake", source, Dataset=object)
    module("deeplake.integrations", source / "integrations")
    module("deeplake.integrations.mm", source / "integrations/mm")
    module("deeplake.integrations.mmdet", source / "integrations/mmdet")
    module("deeplake.types", TypeKind=SimpleNamespace())
    module("mmdet", [], __version__="2.28.1")
    module("mmdet.apis", [])
    module("mmdet.apis.train", auto_scale_lr=lambda *a, **k: None)
    module(
        "mmdet.core",
        eval_map=None,
        eval_recalls=None,
        BitmapMasks=object,
        PolygonMasks=object,
    )
    module("mmcv", [])
    module("mmcv.utils", print_log=lambda *a, **k: None)
    module("terminaltables", AsciiTable=object)
    module("deeplake.integrations.mmdet.mmdet_utils_")
    module(
        "deeplake.integrations.mmdet.test_", single_gpu_test=None, multi_gpu_test=None
    )
    before_import = set(sys.modules)
    loaded = importlib.import_module("deeplake.integrations.mmdet.mmdet_dataset_")
    assert (
        Path(loaded.__file__).resolve()
        == source / "integrations/mmdet/mmdet_dataset_.py"
    )
    try:
        yield loaded
    finally:
        for name in set(sys.modules) - before_import:
            if name.startswith("deeplake."):
                sys.modules.pop(name, None)


def make_dataset(wrapper_module, size=3, invalid=(), transform=None):
    error = importlib.import_module(
        "deeplake.integrations.mm.exceptions"
    ).InvalidImageError

    class Dataset:
        schema = SimpleNamespace(columns=[SimpleNamespace(name="image")])

        def __init__(self):
            self.invalid = set(invalid)
            self.read_indices = []

        def __len__(self):
            return size

        def __getitem__(self, index):
            assert 0 <= index < size, "retry must stay inside dataset bounds"
            self.read_indices.append(index)
            assert (
                len(self.read_indices) <= size * 2 + 1
            ), "repeated invalid fallback did not terminate"
            if index in self.invalid:
                raise error("image", ValueError("decode failed"))
            return {"image": index}

    source = Dataset()
    return source, wrapper_module.MMDetTorchDataset(source, transform=transform)


def test_initial_consecutive_invalid_images_advance_to_next_valid(wrapper_module):
    source, wrapped = make_dataset(wrapper_module, invalid=(0, 1))
    assert wrapped[0] == {"image": 2}
    assert source.read_indices == [0, 1, 2]
    assert wrapped.last_successful_index == 2


def test_all_invalid_is_bounded_and_reports_no_valid_sample(wrapper_module):
    source, wrapped = make_dataset(wrapper_module, invalid=(0, 1, 2))
    with pytest.raises(RuntimeError, match="No valid sample found"):
        wrapped[2]
    assert source.read_indices == [2, 0, 1]
    assert wrapped.last_successful_index == -1


def test_known_successful_fallback_is_retained(wrapper_module):
    source, wrapped = make_dataset(wrapper_module, invalid=(0,))
    assert wrapped[2] == {"image": 2}
    assert wrapped[0] == {"image": 2}
    assert source.read_indices == [2, 0, 2]


def test_formerly_good_fallback_becoming_invalid_does_not_loop(wrapper_module):
    source, wrapped = make_dataset(wrapper_module)
    wrapped[2]
    source.invalid.update((0, 2))
    assert wrapped[0] == {"image": 1}
    assert source.read_indices == [2, 0, 2, 1]
    assert wrapped.last_successful_index == 1


def test_pipeline_none_is_rejected_before_recording_success(wrapper_module):
    source, wrapped = make_dataset(
        wrapper_module, transform=lambda row: None if row["image"] != 2 else row
    )
    assert wrapped[0] == {"image": 2}
    assert source.read_indices == [0, 1, 2]
    assert wrapped.last_successful_index == 2


def test_all_pipeline_rejections_are_bounded(wrapper_module):
    source, wrapped = make_dataset(wrapper_module, transform=lambda row: None)
    with pytest.raises(RuntimeError, match="No valid sample found"):
        wrapped[0]
    assert source.read_indices == [0, 1, 2]
    assert wrapped.last_successful_index == -1


@pytest.mark.parametrize("index", [-4, 3])
def test_out_of_range_input_remains_index_error(wrapper_module, index):
    source, wrapped = make_dataset(wrapper_module)
    with pytest.raises(IndexError):
        wrapped[index]
    assert source.read_indices == []


def test_negative_index_and_empty_dataset(wrapper_module):
    source, wrapped = make_dataset(wrapper_module)
    assert wrapped[-1] == {"image": 2}
    assert source.read_indices == [2]
    source, wrapped = make_dataset(wrapper_module, size=0)
    with pytest.raises(IndexError):
        wrapped[0]
    assert source.read_indices == []
