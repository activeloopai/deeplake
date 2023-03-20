from typing import Optional

from mmdet.apis.train import auto_scale_lr  # type: ignore
from mmdet.utils import (  # type: ignore
    build_dp,
    find_latest_checkpoint,
    get_root_logger,
)
from mmdet.core import DistEvalHook, EvalHook  # type: ignore
from mmdet.core import build_optimizer
from mmcv.runner import (  # type: ignore
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_runner,
)
from mmdet.datasets import replace_ImageToTensor  # type: ignor
import deeplake as dp
from deeplake.util.warnings import always_warn
import warnings
import mmcv  # type: ignore
from mmdet.utils.util_distribution import *  # type: ignore
from deeplake.enterprise.dataloader import indra_available
from ..dataloader import dataloader


BATCH_SIZE = 256
NUM_WORKERS = 1


class TrainDectector:
    def __init__(
        self,
        local_rank,
        model, 
        cfg, 
        ds_train, 
        ds_train_tensors,
        ds_val,
        ds_val_tensors,
        distributed,
        timestamp,
        meta,
        validate,
        port,
    ):
        self.cfg = cfg
        self._init_train_dataset(ds_train, ds_train_tensors)
        self._init_val_dataset(ds_val, ds_val_tensors)
        self.distributed = distributed
        self.timestamp = timestamp
        self.meta = meta
        self.validate = validate
        self.port = port
        self.local_rank = local_rank
        self.put_model_on_gpus(model)
        
        
        # initialize dataloaders:
        # train dataloader
        train_tensor = self.get_tensors(ds_train, ds_train_tensors)
        self.train_dataloader = dataloader.build_dataloader(
            ds_train,  # TO DO: convert it to for loop if we will suport concatting several datasets
            train_tensor["images_tensor"],
            train_tensor["masks_tensor"],
            train_tensor["boxes_tensor"],
            train_tensor["labels_tensor"],
            pipeline=cfg.get("train_pipeline", []),
            implementation=self.dl_impl,
            metric_format=self.metric_format,
            dist=self.dist,
            shuffle=self.shuffle,
            num_gpus=self.num_gpus,
        )
        
        if validate:
            # validation dataloader
            val_tensor = self.get_tensors(ds_val, ds_val_tensors)
            self.val_dataloader = dataloader.build_dataloader(
                ds_val,  # TO DO: convert it to for loop if we will suport concatting several datasets
                val_tensor["images_tensor"],
                val_tensor["masks_tensor"],
                val_tensor["boxes_tensor"],
                val_tensor["labels_tensor"],
                pipeline=cfg.get("test_pipeline", []),
                implementation=self.dl_impl,
                metric_format=self.metric_format,
                dist=self.dist,
                shuffle=False,
                num_gpus=self.num_gpus,
            )
    
    def put_model_on_gpus(self, model):
        # put model on gpus
        if self.distributed:
            find_unused_parameters = self.cfg.get("find_unused_parameters", False)
            # Sets the `find_unused_parameters` parameter in
            # # torch.nn.parallel.DistributedDataParallel
            # model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
            #                                           device_ids=[local_rank],
            #                                           output_device=local_rank,
            #                                           broadcast_buffers=False,
            #                                           find_unused_parameters=find_unused_parameters)
            force_cudnn_initialization(self.cfg.gpu_ids[self.local_rank])
            ddp_setup(self.local_rank, len(self.cfg.gpu_ids), self.port)
            self.model = build_ddp(
                model,
                self.cfg.device,
                device_ids=[self.cfg.gpu_ids[self.local_rank]],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.model = build_dp(model, self.cfg.device, device_ids=self.cfg.gpu_ids)

        
    def check_shuffle_for_val_config(self):
        deeplake_dataloader = self.cfg.data.val.get("deeplake_dataloader", False)
        if deeplake_dataloader:
            if deeplake_dataloader.get("shuffle", False):
                always_warn("shuffle argument for validation dataset will be ignored.")
    
    def _init_train_dataset(self, ds_train, ds_train_tensors):
        if ds_train is None:
            self.ds_train = load_ds_from_cfg(self.cfg.data.train)
            self.ds_train_tensors = self.cfg.data.train.get("deeplake_tensors", {})
        else:
            cfg_data = self.cfg.data.train.get("deeplake_path")
            if cfg_data:
                always_warn(
                    "A Deep Lake dataset was specified in the cfg as well as inthe dataset input to train_detector. The dataset input to train_detector will be used in the workflow."
                )
            self.ds_train = ds_train
            self.ds_train_tensors = ds_train_tensors
    
    def _init_val_dataset(self, ds_val, ds_val_tensors):
        if ds_val is None:
            self.ds_val = load_ds_from_cfg(self.cfg.data.val)
            self.ds_val_tensors = self.cfg.data.val.get("deeplake_tensors", {})
        else:
            cfg_data = self.cfg.data.val.get("deeplake_path")
            if cfg_data:
                always_warn(
                    "A Deep Lake dataset was specified in the cfg as well as inthe dataset input to train_detector. The dataset input to train_detector will be used in the workflow."
                )
            self.ds_val = ds_val
            self.ds_val_tensors = ds_val_tensors
            
    @property
    def batch_size_per_mode(self):
        return {
            "train": self.ds_train.get("batch_size"),
            "val": self.ds_val.get("batch_size")
        }
        
    @property
    def num_workers_per_mode(self):
        return {
            "train": self.ds_train.get("num_workers"),
            "val": self.ds_val.get("num_workers")
        }
    
    def batch_size(self, mode):
        # if batch size is not spicified use degault value
        samples_per_gpu = self.cfg.data.get("samples_per_gpu", BATCH_SIZE)        
        
        batch_size = self.batch_size_per_mode[mode]
        return batch_size or samples_per_gpu
    
    def num_workers(self, mode):
        workers_per_gpu = self.cfg.data.get("workers_per_gpu", NUM_WORKERS)
        
        num_workers =  self.num_workers_per_mode[mode]
        return num_workers or workers_per_gpu

    def get_tensors(self, ds, ds_tensors):
        if self.ds_train_tensors:
            return {
                "images_tensor": ds_tensors["img"],
                "boxes_tensor": ds_tensors["gt_bboxes"],
                "labels_tensor": ds_tensors["gt_labels"],
                "masks_tensor": ds_tensors.get("gt_masks")
            }
        return self._fetch_tensors(ds)
    
    def _fetch_tensors(self, ds):
        images_tensor = _find_tensor_with_htype(ds, "image", "img")
        boxes_tensor = _find_tensor_with_htype(ds, "bbox", "gt_bboxes")
        labels_tensor = _find_tensor_with_htype(
            ds, "class_label", "train gt_labels"
        )
        masks_tensor = self._get_masks_tensor(ds)
        return {
            "images_tensor": images_tensor,
            "boxes_tensor": boxes_tensor,
            "labels_tensor": labels_tensor,
            "masks_tensor": masks_tensor,
        }
        
    def _get_masks_tensor(self, ds):
        train_masks_tensor = None
        collection_keys = get_collect_keys(self.cfg)
        if "gt_masks" in collection_keys:
            train_masks_tensor = _find_tensor_with_htype(
                ds, "binary_mask", "gt_masks"
            ) or _find_tensor_with_htype(ds, "polygon", "gt_masks")
        return train_masks_tensor        
    
    def _set_model_classes(self):
        if hasattr(model, "CLASSES"):
            warnings.warn(
                "model already has a CLASSES attribute. dataset.info.class_names will not be used."
            )
        elif hasattr(self.ds_train[self.train_labels_tensor].info, "class_names"):
            self.model.CLASSES = self.ds_train[self.train_labels_tensor].info.class_names
    
    @property
    def metrics_format(self):
        return self.cfg.get("deeplake_metrics_format", "COCO")
    
    @property
    def logger(self):
        return get_root_logger(log_level=self.cfg.log_level)
    
    @property
    def runner_type(self):
        return "EpochBasedRunner" if "runner" not in self.cfg else self.cfg.runner["type"]
    
    @property
    def dl_impl(self):
        return self.cfg.get("deeplake_dataloader_type", "auto").lower()
    
    def build_dataloader(self, *args, **kwargs):
        return dataloader.build_dataloader(*args, **kwargs)
    
    def build_train_dataloader(self):
        tensors = self.get_tensors(self.ds_train, self.ds_train_tensors)
        return self.build_dataloader(
            dataset = self.ds_train,
            implementation = self.implementation,
            pipeline = self.pipeline,
            mode = self.mode,
            metric_format = self.metrics_format,
            dist = self.dist,
            shuffle = self.shuffle,
            num_gpus = self.num_gpus,
            persistent_workers = False,
            **tensors,
        )
        
    def build_val_dataloader(self):
        tensors = self.get_tensors(self.ds_val, self.ds_val_tensors)
        return self.build_dataloader(
            dataset = self.ds_val,
            implementation = self.implementation,
            pipeline = self.pipeline,
            mode = self.mode,
            metric_format = self.metrics_format,
            dist = self.dist,
            shuffle = self.shuffle,
            num_gpus = self.num_gpus,
            persistent_workers = False,
            **tensors,
        )
    
    def cast_runner_type(self):
        if self.cfg.runner.type == "IterBasedRunner":
            self.cfg.runner.type = "DeeplakeIterBasedRunner"
        elif self.cfg.runner.type == "EpochBasedRunner":
            self.cfg.runner.type = "DeeplakeEpochBasedRunner"
    
    def fp16_settings(self):
        # fp16 setting
        fp16_cfg = self.cfg.get("fp16", None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **self.cfg.optimizer_config, **fp16_cfg, distributed=self.distributed
            )
        elif self.distributed and "type" not in self.cfg.optimizer_config:
            optimizer_config = OptimizerHook(**self.cfg.optimizer_config)
        else:
            optimizer_config = self.cfg.optimizer_config
        
        return optimizer_config
    
    def build_runner(self):
        self.cast_runner_type()
        
        # build optimizer
        auto_scale_lr(self.cfg, self.distributed, self.logger)
        optimizer = build_optimizer(self.model, self.cfg.optimizer)
        self.runner = build_runner(
            self.cfg.runner,
            default_args=dict(
            model=self.model,
            optimizer=optimizer,
            work_dir=self.cfg.work_dir,
            logger=self.logger,
            meta=self.meta,
            ),
        )
        
        # an ugly workaround to make .log and .log.json filenames the same
        self.runner.timestamp = self.timestamp
        self.optimizer_config = self.fp16_settings()
        self.register_training_hooks()
        
        if self.validate:
            self.register_validation_hooks()
    
    def run(self):
        self.build_runner()
        self.resume_from()
        self.runner.run([self.train_dataloader], self.cfg.workflow)
    
    def resume_from(self):
        resume_from = None
        if self.cfg.resume_from is None and self.cfg.get("auto_resume"):
            resume_from = find_latest_checkpoint(self.cfg.work_dir)
        if resume_from is not None:
            self.cfg.resume_from = resume_from

        if self.cfg.resume_from:
            self.runner.resume(self.cfg.resume_from)
        elif self.cfg.load_from:
            self.runner.load_checkpoint(self.cfg.load_from)
    
    def register_training_hooks(self):
        # register hooks
        self.runner.register_training_hooks(
            self.cfg.lr_config,
            self.optimizer_config,
            self.cfg.checkpoint_config,
            self.cfg.log_config,
            self.cfg.get("momentum_config", None),
            custom_hooks_config=self.cfg.get("custom_hooks", None),
        )
    
        if self.distributed:
            if isinstance(self.runner, EpochBasedRunner):
                self.runner.register_hook(DistSamplerSeedHook())
        
    def register_validation_hooks(self):
        eval_cfg = self.cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = self.cfg.runner["type"] != "DeeplakeIterBasedRunner"
        eval_hook = EvalHook
        if self.distributed:
            eval_hook = DistEvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        self.runner.register_hook(eval_hook(self.val_dataloader, **eval_cfg), priority="LOW")  
