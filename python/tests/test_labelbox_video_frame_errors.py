import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_labelbox_utils(monkeypatch):
    for name in (
        "deeplake",
        "deeplake.integrations",
        "deeplake.integrations.labelbox",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    deeplake_utils = types.ModuleType("deeplake.integrations.labelbox.deeplake_utils")
    deeplake_utils.generic_tensor_create_kwargs_ = lambda *args, **kwargs: {}
    deeplake_utils.image_tensor_create_kwargs_ = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, deeplake_utils.__name__, deeplake_utils)

    source = (
        Path(__file__).resolve().parents[1]
        / "deeplake"
        / "integrations"
        / "labelbox"
        / "labelbox_utils.py"
    )
    spec = importlib.util.spec_from_file_location("labelbox_utils_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_open_failure_is_propagated_after_retries(monkeypatch):
    module = _load_labelbox_utils(monkeypatch)
    calls = []
    av = types.ModuleType("av")

    def open_video(path, options=None):
        calls.append(path)
        raise OSError("video open failed")

    av.open = open_video
    monkeypatch.setitem(sys.modules, "av", av)

    with pytest.raises(OSError, match="video open failed"):
        list(module.frame_generator_("broken-video.mp4", retries=1))

    assert calls == ["broken-video.mp4", "broken-video.mp4"]


def test_decode_failure_is_propagated_after_yielded_frames(monkeypatch):
    module = _load_labelbox_utils(monkeypatch)

    class Frame:
        def to_ndarray(self, format):
            return format

    class Container:
        def decode(self, video):
            yield Frame()
            raise ValueError("video decode failed")

    av = types.ModuleType("av")
    av.open = lambda path, options=None: Container()
    monkeypatch.setitem(sys.modules, "av", av)

    frames = module.frame_generator_("truncated-video.mp4", retries=0)
    assert next(frames) == (0, "rgb24")
    with pytest.raises(ValueError, match="video decode failed"):
        next(frames)
