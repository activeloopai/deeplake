from typing import Optional, List, str, int

from mmcv.parallel import collate  # type: ignore
from functools import partial
import deeplake as dp
from ..transform.transform import transform
from ..dataset import mmdet_dataset, subiterable_dataset
from mmdet.utils.util_distribution import *  # type: ignore
from deeplake.enterprise.dataloader import dataloader
import types


class MMDetDataLoader:
    def __init__(
        self,
        dataset: dp.Dataset,
        images_tensor: str,
        masks_tensor: Optional[str],
        boxes_tensor: str,
        labels_tensor: str,
        implementation: str,
        pipeline: List,
        batch_size: int,
        num_workers: int,
        mode: str = "train",
        metric_format: str = "COCO",
        dist: str = False,
        shuffle: str = False,
        num_gpus: int = 1,
    ):
        self.dataset = dataset
        self.dataset.CLASSES = self.classes
        self.images_tensor = images_tensor
        self.masks_tensor = masks_tensor
        self.boxes_tensor = boxes_tensor
        self.labels_tensor = labels_tensor
        self.implementation = implementation
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.metric_format = metric_format
        self.dist = dist
        self.shuffle = shuffle
        self.num_gpus = num_gpus

    @property
    def poly2mask(self):
        _poly2mask = False
        if self.masks_tensor is not None:
            if self.dataset[self.masks_tensor].htype == "polygon":
                _poly2mask = True
        return _poly2mask

    @property
    def bbox_info(self):
        return self.dataset[self.boxes_tensor].info

    @property
    def classes(self):
        return self.dataset[self.labels_tensor].info.class_names

    @property
    def pipeline(self):
        return build_pipeline(self.pipeline)

    @property
    def tensors_dict(self):
        _tensors_dict = {
            "images_tensor": self.images_tensor,
            "boxes_tensor": self.boxes_tensor,
            "labels_tensor": self.labels_tensor,
        }

        if self.masks_tensor is not None:
            _tensors_dict["masks_tensor"] = self.masks_tensor
        return _tensors_dict

    @property
    def tensors(self):
        _tensors = [self.images_tensor, self.labels_tensor, self.boxes_tensor]
        if self.masks_tensor is not None:
            _tensors.append(self.masks_tensor)
        return _tensors

    @property
    def collate_fn(self):
        return partial(collate, samples_per_gpu=self.batch_size)

    @property
    def decode_method(self):
        return {self.images_tensor: "numpy"}

    @property
    def transform_fn(self):
        _transform_fn = partial(
            transform,
            images_tensor=self.images_tensor,
            masks_tensor=self.masks_tensor,
            boxes_tensor=self.boxes_tensor,
            labels_tensor=self.labels_tensor,
            pipeline=self.pipeline,
            bbox_info=self.bbox_info,
            poly2mask=self.poly2mask,
        )
        return _transform_fn

    def buld_dataloader(self):
        if self.dist and self.implementation == "python":
            raise NotImplementedError(
                "Distributed training is not supported by the python data loader. Set deeplake_dataloader_type='c++' to use the C++ dtaloader instead."
            )

        if self.implementation == "python":
            return self.load_python_dataloader()
        return self.load_indra_dataloader()

    def load_python_dataloader(self):
        loader = self.dataset.pytorch(
            tensors_dict=self.tensors_dict,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            transform=self.transform_fn,
            tensors=self.tensors,
            collate_fn=self.collate_fn,
            metrics_format=self.metrics_format,
            pipeline=self.pipeline,
            batch_size=self.batch_size,
            mode=self.mode,
            bbox_info=self.bbox_info,
            decode_method=self.decode_method,
        )

        mmdet_ds = mmdet_dataset.MMDetDataset(
            dataset=self.dataset,
            metrics_format=self.metrics_format,
            pipeline=self.pipeline,
            tensors_dict=self.tensors_dict,
            tensors=self.tensors,
            mode=self.mode,
            bbox_info=self.bbox_info,
            decode_method=self.decode_method,
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
        )

        loader.dataset.mmdet_dataset = mmdet_ds
        loader.dataset.pipeline = loader.dataset.mmdet_dataset.pipeline
        loader.dataset.evaluate = types.MethodType(
            subiterable_dataset.mmdet_subiterable_dataset_eval, loader.dataset
        )
        loader.dataset.CLASSES = self.classes
        return loader

    def load_indra_dataloader(self):
        loader = (
            dataloader(self.dataset)
            .transform(self.transform_fn)
            .shuffle(self.shuffle)
            .batch(self.batch_size)
            .pytorch(
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                tensors=self.tensors,
                distributed=self.dist,
                decode_method=self.decode_method,
            )
        )

        mmdet_ds = mmdet_dataset.MMDetDataset(
            dataset=self.dataset,
            metrics_format=self.metrics_format,
            pipeline=self.pipeline,
            tensors_dict=self.tensors_dict,
            tensors=self.tensors,
            mode=self.mode,
            bbox_info=self.bbox_info,
            decode_method=self.decode_method,
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
        )
        loader.dataset = mmdet_ds
        loader.dataset.CLASSES = self.classes
        return loader
