from typing import Optional, List

import deeplake
from .mmdet_dataloader import MMDetDataLoader


def build_dataloader(
    dataset: deeplake.Dataset,
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
) -> MMDetDataLoader:
    """
    Build a dataloader for MMDet using the provided configuration.

    Args:
        dataset (deeplake.Dataset): The Deeplake dataset object.
        images_tensor (str): The name of the images tensor in the dataset.
        masks_tensor (Optional[str]): The name of the masks tensor in the dataset, if available.
        boxes_tensor (str): The name of the bounding boxes tensor in the dataset.
        labels_tensor (str): The name of the labels tensor in the dataset.
        implementation (str): The dataloader implementation to use, either "python" or "c++".
        pipeline (List): A list of data augmentation and preprocessing steps for the data.
        mode (str, optional): The mode of the dataloader, either "train" or "val". Defaults to "train".
        metric_format (str, optional): The format of the dataset metrics. Defaults to "COCO".
        dist (str, optional): Whether to use distributed training. Defaults to False.
        shuffle (str, optional): Whether to shuffle the dataset. Defaults to False.
        num_gpus (int, optional): The number of GPUs to use for training. Defaults to 1.

    Returns:
        MMDetDataLoader: The built dataloader object.
    """
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
