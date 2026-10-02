import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class InvalidImageError(Exception):
    pass


class InvalidSegmentError(Exception):
    def __init__(self, column_name, error):
        super().__init__(f"Error on {column_name} data getting: {error}")


def _load_transform(monkeypatch):
    for name in (
        "deeplake",
        "deeplake.integrations",
        "deeplake.integrations.mm",
        "deeplake.integrations.mmseg",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    exceptions = types.ModuleType("deeplake.integrations.mm.exceptions")
    exceptions.InvalidImageError = InvalidImageError
    exceptions.InvalidSegmentError = InvalidSegmentError
    monkeypatch.setitem(sys.modules, exceptions.__name__, exceptions)

    upcast = types.ModuleType("deeplake.integrations.mm.upcast_array")
    upcast.upcast_array = lambda value: value
    monkeypatch.setitem(sys.modules, upcast.__name__, upcast)

    mmcv = types.ModuleType("mmcv")
    mmcv.__path__ = []
    mmcv_utils = types.ModuleType("mmcv.utils")
    mmcv_utils.build_from_cfg = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "mmcv", mmcv)
    monkeypatch.setitem(sys.modules, "mmcv.utils", mmcv_utils)

    mmseg = types.ModuleType("mmseg")
    mmseg.__path__ = []
    datasets = types.ModuleType("mmseg.datasets")
    datasets.__path__ = []
    builder = types.ModuleType("mmseg.datasets.builder")
    builder.PIPELINES = {}
    pipelines = types.ModuleType("mmseg.datasets.pipelines")
    pipelines.Compose = lambda steps: steps
    monkeypatch.setitem(sys.modules, "mmseg", mmseg)
    monkeypatch.setitem(sys.modules, "mmseg.datasets", datasets)
    monkeypatch.setitem(sys.modules, "mmseg.datasets.builder", builder)
    monkeypatch.setitem(sys.modules, "mmseg.datasets.pipelines", pipelines)

    source = (
        Path(__file__).resolve().parents[1]
        / "deeplake"
        / "integrations"
        / "mmseg"
        / "compose_transform_.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compose_transform_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transform


def test_missing_mask_raises_typed_invalid_segment_error(monkeypatch):
    transform = _load_transform(monkeypatch)
    sample = {"image": np.zeros((2, 2, 3), dtype=np.uint8)}

    with pytest.raises(InvalidSegmentError, match="mask data getting"):
        transform(sample, "image", "mask", lambda value: value)


def test_image_only_validation_with_optional_mask_raises_typed_error(monkeypatch):
    transform = _load_transform(monkeypatch)
    sample = {"image": np.zeros((2, 2, 3), dtype=np.uint8)}

    # MMSeg validation pipelines may collect only `img`, so the integration
    # passes masks_tensor=None into transform for this supported path.
    with pytest.raises(InvalidSegmentError, match="None data getting"):
        transform(sample, "image", None, lambda value: value)
