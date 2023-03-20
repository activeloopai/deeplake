from typing import Optional, List

import deeplake as dp

from .mmdet_dataloader import MMDetDataLoader


def build_dataloader(
    dataset: dp.Dataset,
    images_tensor: str,
    masks_tensor: Optional[str],
    boxes_tensor: str,
    labels_tensor: str,
    implementation: str,
    pipeline: List,
    mode: str = "train",
    metric_format: str = "COCO",
    dist: str = False,
    shuffle: str = False,
    num_gpus: int = 1,
):
    dataloader_builder = MMDetDataLoader(
        dataset,
        images_tensor,
        masks_tensor,
        boxes_tensor,
        labels_tensor,
        implementation,
        pipeline,
        mode,
        metric_format,
        dist,
        shuffle,
        num_gpus,
    )

    return dataloader_builder.buld_dataloader()