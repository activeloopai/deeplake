from mmcv.utils import build_from_cfg  # type: ignore
from mmdet.datasets.builder import PIPELINES  # type: ignore
from mmdet.datasets.pipelines import Compose  # type: ignore
import deeplake as dp
from deeplake.util.warnings import always_warn
from mmdet.datasets.pipelines import Compose
from mmdet.utils.util_distribution import *  # type: ignore


def find_tensor_with_htype(ds: dp.Dataset, htype: str, mmdet_class=None):
    tensors = [k for k, v in ds.tensors.items() if v.meta.htype == htype]
    if mmdet_class is not None:
        always_warn(
            f"No deeplake tensor name specified for '{mmdet_class} in config. Fetching it using htype '{htype}'."
        )
    if not tensors:
        always_warn(f"No tensor found with htype='{htype}'")
        return None
    t = tensors[0]
    if len(tensors) > 1:
        always_warn(f"Multiple tensors with htype='{htype}' found. choosing '{t}'.")
    return t


def build_pipeline(steps):
    return Compose(
        [
            build_from_cfg(step, PIPELINES, None)
            for step in steps
            if step["type"] not in {"LoadImageFromFile", "LoadAnnotations"}
        ]
    )
