from typing import Callable
from mmdet.apis.train import *
from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmcv.utils import build_from_cfg, Registry
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate
from functools import partial

from mmdet.core import BitmapMasks
import albumentations as A
import deeplake as dp
from deeplake.util.warnings import always_warn


def _coco_2_pascal(boxes):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height
    return np.stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 0] + np.clip(boxes[:, 2], 1, None),
            boxes[:, 1] + np.clip(boxes[:, 3], 1, None),
        ),
        axis=1,
    )


rand_crop = A.Compose(
    [
        A.RandomSizedBBoxSafeCrop(width=128, height=128, erosion_rate=0.2),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels", "bbox_ids"],
        min_area=25,
        min_visibility=0.6,
    ),
)


def _find_tensor_with_htype(ds: dp.Dataset, htype: str):
    tensors = [k for k, v in ds.tensors.items() if v.meta.htype == htype]
    if not tensors:
        raise ValueError("No tensor found with htype='{htype}'")
    t = tensors[0]
    if len(tensors) > 1:
        always_warn(f"Multiple tensors with htype='{htype}' found. Chosing '{t}'.")
    return t


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    boxes_tensor: str,
    labels_tensor: str,
    pipleline: Callable,
):
    img = sample_in[images_tensor]
    masks = sample_in[masks_tensor]
    bboxes = sample_in[boxes_tensor]
    labels = sample_in[labels_tensor]

    img = img[..., ::-1]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    masks = masks.transpose((2, 0, 1)).astype(np.uint8)
    bboxes = _coco_2_pascal(bboxes)

    transformed = rand_crop(
        image=img,
        masks=list(masks),
        bboxes=bboxes,
        bbox_ids=np.arange(len(bboxes)),
        labels=labels,
    )

    img = transformed["image"].astype(np.uint8)
    shape = img.shape
    labels = np.array(transformed["labels"], dtype=np.int64)
    bboxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)

    bbox_ids = transformed["bbox_ids"]
    masks = [transformed["masks"][i] for i in transformed["bbox_ids"]]
    if masks:
        masks = np.stack(masks, axis=0).astype(bool)
    else:
        masks = np.empty((0, shape[0], shape[1]), dtype=bool)

    return pipleline(
        {
            "img": img,
            "img_fields": ["img"],
            "filename": None,
            "ori_filename": None,
            "img_shape": shape,
            "ori_shape": shape,
            "gt_masks": BitmapMasks(masks, *shape[:2]),
            "gt_bboxes": bboxes,
            "gt_labels": labels,
            "bbox_fields": ["gt_bboxes"],
        }
    )


def build_dataloader(
    ds,
    cfg,
    images_tensor,
    masks_tensor,
    boxes_tensor,
    labels_tensor,
    **train_loader_config,
):
    if isinstance(ds, dp.Dataset):
        pipeline = build_pipeline(cfg.train_pipeline)
        transform = partial(
            transform,
            images_tensor=images_tensor,
            masks_tensor=masks_tensor,
            boxes_tensor=boxes_tensor,
            labels_tensor=labels_tensor,
            pipeline=pipeline,
        )
        num_workers = train_loader_config["num_workers_per_gpu"]
        shuffle = train_loader_config.get("shuffle", True)
        loader = ds.pytorch(
            num_workers=num_workers,
            shuffle=shuffle,
            transform=transform,
            tensors=["images", "categories", "boxes", "masks"],
            collate_fn=partial(
                collate, samples_per_gpu=train_loader_config["samples_per_gpu"]
            ),
        )
        loader.dataset.CLASSES = [
            c["name"] for c in ds.categories.info["category_info"]
        ]
        return loader

    return mmdet_build_dataloader(ds, **train_loader_config)


def build_pipeline(steps):
    return Compose(
        [
            build_from_cfg(step, PIPELINES, None)
            for step in steps
            if step["type"] not in {"LoadImageFromFile", "LoadAnnotations"}
        ]
    )


def train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
    images_tensor: Optional[str] = None,
    masks_tensor: Optional[str] = None,
    boxes_tensor: Optional[str] = None,
    labels_tensor: Optional[str] = None,
):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = "EpochBasedRunner" if "runner" not in cfg else cfg.runner["type"]

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False,
    )

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get("train_dataloader", {}),
    }

    data_loaders = [
        build_dataloader(
            ds,
            cfg,
            images_tensor,
            masks_tensor,
            boxes_tensor,
            labels_tensor,
            **train_loader_cfg,
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get("custom_hooks", None),
    )

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False,
        )

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get("val_dataloader", {}),
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    resume_from = None
    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
