import os
import copy
from typing import Any, Dict

from deeplake.integrations.mmlab.segmentation.registry import TRANSFORMS  # type: ignore
from deeplake.integrations.mmlab.segmentation.load_annotations import LoadAnnotations
from mmengine.dataset import Compose  # type: ignore

from deeplake.client.config import DEEPLAKE_AUTH_TOKEN

from deeplake.util.exceptions import (
    EmptyTokenException,
    EmptyDeeplakePathException,
    ConflictingDatasetParametersError,
    MissingTensorMappingError,
)

import mmengine.registry  # type: ignore

original_build_func = mmengine.registry.DATASETS.build


def build_transform(steps):
    from mmengine.registry.build_functions import build_from_cfg  # type: ignore

    transforms = []
    steps_copy = copy.deepcopy(steps)

    for step in steps_copy:
        if step["type"] == "LoadAnnotations":
            # Create LoadAnnotations instance and add to transforms list
            kwargs = step.copy()
            kwargs.pop("type")
            transform = LoadAnnotations(**kwargs)
            transforms.append(transform)
        elif step["type"] != "LoadImageFromFile":
            transform = build_from_cfg(step, TRANSFORMS, None)
            transforms.append(transform)

    return Compose(transforms)


def build_func_patch(
    cfg: Dict,
    *args,
    **kwargs,
) -> Any:
    import deeplake as dp

    creds = cfg.pop("deeplake_credentials", {})
    token = creds.pop("token", None)
    token = token or os.environ.get(DEEPLAKE_AUTH_TOKEN)
    if token is None:
        raise EmptyTokenException()

    ds_path = cfg.pop("deeplake_path", None)
    if ds_path is None or not len(ds_path):
        raise EmptyDeeplakePathException()

    deeplake_ds = dp.load(ds_path, token=token, read_only=True)[0:500:1]
    deeplake_commit = cfg.pop("deeplake_commit", None)
    deeplake_view_id = cfg.pop("deeplake_view_id", None)
    deeplake_query = cfg.pop("deeplake_query", None)

    if deeplake_view_id and deeplake_query:
        raise ConflictingDatasetParametersError()

    if deeplake_commit:
        deeplake_ds.checkout(deeplake_commit)

    if deeplake_view_id:
        deeplake_ds = deeplake_ds.load_view(id=deeplake_view_id)

    if deeplake_query:
        deeplake_ds = deeplake_ds.query(deeplake_query)

    ds_train_tensors = cfg.pop("deeplake_tensors", {})

    if "pipeline" in cfg:
        transform_pipeline = build_transform(cfg.get("pipeline"))
    else:
        transform_pipeline = None

    if not ds_train_tensors and not {"img", "gt_semantic_seg"}.issubset(
        ds_train_tensors
    ):
        raise MissingTensorMappingError()

    cfg["lazy_init"] = False
    res = original_build_func(cfg, *args, **kwargs)
    res.deeplake_dataset = deeplake_ds
    res.images_tensor = ds_train_tensors.get("img")
    res.masks_tensor = ds_train_tensors.get("gt_semantic_seg")
    return res, transform_pipeline
