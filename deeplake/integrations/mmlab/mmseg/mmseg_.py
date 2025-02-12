# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import numpy as np
import warnings

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.registry import RUNNERS

from typing import Any, Dict, Callable

import mmengine.registry

original_build_func = mmengine.registry.DATASETS.build

from deeplake.util.exceptions import (
    EmptyTokenException,
    EmptyDeeplakePathException,
    ConflictingDatasetParametersError,
    MissingTensorMappingError,
)
from deeplake.client.config import DEEPLAKE_AUTH_TOKEN

from mmengine.dataset import Compose
from mmcv.transforms.base import BaseTransform


from mmengine.registry import Registry
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS

TRANSFORMS = Registry(
    "transform",
    parent=MMSEG_TRANSFORMS,
    # locations=['mmseg.datasets.transforms.transforms'])
    locations=["deeplake.integrations.mmlab.mmseg.mmseg_"],
)

from deeplake.integrations.mmlab.mmseg.basedataset import (
    BaseDataset as DeeplakeBaseDataset,
)

mmengine.dataset.BaseDataset = DeeplakeBaseDataset


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend="pillow",
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
        )
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn(
                "`reduce_zero_label` will be deprecated, "
                "if you would like to ignore the zero label, please "
                "set `reduce_zero_label=True` when dataset "
                "initialized"
            )
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        gt_semantic_seg = results.pop("dp_seg_map", None)

        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results["gt_seg_map"] = gt_semantic_seg

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str


def build_transform(steps):
    from mmengine.registry.build_functions import build_from_cfg

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


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    pipeline: Callable,
):
    img = sample_in[images_tensor]
    if isinstance(img, (bytes, bytearray)):
        img = np.array(Image.open(io.BytesIO(img)))
    elif not isinstance(img, np.ndarray):
        img = np.array(img)

    mask = sample_in[masks_tensor]
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img = img[..., ::-1]  # rgb_to_bgr should be optional
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    shape = img.shape

    pipeline_dict = {
        "img_path": None,
        "seg_map_path": None,
        "label_map": None,  # check if the
        "seg_fields": ["gt_seg_map"],
        "sample_idx": int(sample_in["index"]),  # put the sample idx
        "img": np.ascontiguousarray(img, dtype=np.float32),
        "img_shape": shape[:2],
        "ori_shape": shape[:2],
        "dp_seg_map": np.ascontiguousarray(mask, np.int64),
    }

    return pipeline(pipeline_dict)


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


mmengine.registry.DATASETS.build = build_func_patch


from mmengine.runner import Runner
from typing import Optional, Union
from torch.utils.data import DataLoader
import copy
from functools import partial
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.dataset import worker_init_fn as default_worker_init_fn

from mmengine.registry import DATA_SAMPLERS, DATASETS, FUNCTIONS
from mmengine.utils import digit_version
from mmengine.runner.utils import _get_batch_size
from deeplake.enterprise.dataloader import DeepLakeDataLoader

_original_build_dataloader = Runner.build_dataloader

from mmengine.dist import (
    broadcast,
    get_dist_info,
    get_rank,
    get_world_size,
    init_dist,
    is_distributed,
    master_only,
)


def build_dataloader(
    dataloader: Union[DataLoader, Dict],
    seed: Optional[int] = None,
    diff_rank_seed: bool = False,
) -> DeepLakeDataLoader:
    if isinstance(dataloader, DataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop("dataset")
    dataset, transform_pipeline = build_func_patch(dataset_cfg)
    # if hasattr(dataset, "full_init"):
    #     dataset.full_init()

    num_batch_per_epoch = dataloader_cfg.pop("num_batch_per_epoch", None)
    # if num_batch_per_epoch is not None:
    #     world_size = get_world_size()
    #     num_samples = (
    #         num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
    #         world_size)
    #     dataset = _SlicedDataset(dataset, num_samples)

    # build sampler
    sampler_cfg = dataloader_cfg.pop("sampler")
    if isinstance(sampler_cfg, dict):
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg, default_args=dict(dataset=dataset, seed=sampler_seed)
        )
    else:
        # fallback to raise error in dataloader
        # if `sampler_cfg` is not a valid type
        sampler = sampler_cfg

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop("batch_sampler", None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler, batch_size=dataloader_cfg.pop("batch_size")
            ),
        )
    else:
        # fallback to raise error in dataloader
        # if `batch_sampler_cfg` is not a valid type
        batch_sampler = batch_sampler_cfg

    # build dataloader
    init_fn: Optional[partial]
    if "worker_init_fn" in dataloader_cfg:
        worker_init_fn_cfg = dataloader_cfg.pop("worker_init_fn")
        worker_init_fn_type = worker_init_fn_cfg.pop("type")
        if isinstance(worker_init_fn_type, str):
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
        elif callable(worker_init_fn_type):
            worker_init_fn = worker_init_fn_type
        else:
            raise TypeError(
                "type of worker_init_fn should be string or callable "
                f"object, but got {type(worker_init_fn_type)}"
            )
        assert callable(worker_init_fn)
        init_fn = partial(worker_init_fn, **worker_init_fn_cfg)  # type: ignore
    else:
        if seed is not None:
            disable_subprocess_warning = dataloader_cfg.pop(
                "disable_subprocess_warning", False
            )
            assert isinstance(disable_subprocess_warning, bool), (
                "disable_subprocess_warning should be a bool, but got "
                f"{type(disable_subprocess_warning)}"
            )
            init_fn = partial(
                default_worker_init_fn,
                num_workers=dataloader_cfg.get("num_workers"),
                rank=get_rank(),
                seed=seed,
                disable_subprocess_warning=disable_subprocess_warning,
            )
        else:
            init_fn = None

    # `persistent_workers` requires pytorch version >= 1.7
    if "persistent_workers" in dataloader_cfg and digit_version(
        TORCH_VERSION
    ) < digit_version("1.7.0"):
        print_log(
            "`persistent_workers` is only available when " "pytorch version >= 1.7",
            logger="current",
            level=logging.WARNING,
        )
        dataloader_cfg.pop("persistent_workers")

    # The default behavior of `collat_fn` in dataloader is to
    # merge a list of samples to form a mini-batch of Tensor(s).
    # However, in mmengine, if `collate_fn` is not defined in
    # dataloader_cfg, `pseudo_collate` will only convert the list of
    # samples into a dict without stacking the batch tensor.
    collate_fn_cfg = dataloader_cfg.pop("collate_fn", dict(type="pseudo_collate"))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop("type")
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    elif callable(collate_fn_cfg):
        collate_fn = collate_fn_cfg
    else:
        raise TypeError(
            "collate_fn should be a dict or callable object, but got "
            f"{collate_fn_cfg}"
        )

    deeplake_ds = dataset.deeplake_dataset
    images_tensor = dataset.images_tensor
    masks_tensor = dataset.masks_tensor
    classes = deeplake_ds[masks_tensor].info.class_names
    dataset.CLASSES = classes

    num_workers = dataloader_cfg.get("num_workers", 0)
    batch_size = dataloader_cfg.get("batch_size", 1)
    shuffle = dataloader_cfg.get("shuffle", False)
    tensors = [images_tensor, masks_tensor]
    drop_last = dataloader_cfg.get("drop_last", False)
    persistent_workers = dataloader_cfg.get("persistent_workers", False)

    transform_fn = partial(
        transform,
        images_tensor=images_tensor,
        masks_tensor=masks_tensor,
        pipeline=transform_pipeline,
    )

    loader = (
        deeplake_ds.dataloader()
        .transform(transform_fn)
        .shuffle(shuffle)
        .batch(batch_size=batch_size, drop_last=drop_last)
        .pytorch(
            num_workers=num_workers,
            collate_fn=collate_fn,
            tensors=tensors,
            distributed=is_distributed(),
            persistent_workers=persistent_workers,
        )
    )
    loader.dataset.__setattr__("metainfo", {"classes": classes})

    if init_fn:
        loader.worker_init_fn = init_fn

    return loader


Runner.build_dataloader = staticmethod(build_dataloader)
