import mmcv  # type: ignore
import logging
from mmcv import runner
from torch.utils.data import DataLoader

import time
import warnings

from deeplake.constants import TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING
from .empty_memory import empty_cuda


@runner.RUNNERS.register_module()
class DeeplakeEpochBasedRunner(runner.EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        start_time = time.time()
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook("after_train_iter")
            del self.data_batch
            self._iter += 1
            iter_time = time.time()
            if iter_time - start_time > TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING:
                empty_cuda()
                start_time = iter_time

        self.call_hook("after_train_epoch")
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        start_time = time.time()
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_val_iter")
            self.run_iter(data_batch, train_mode=False)
            self.call_hook("after_val_iter")
            del self.data_batch
            iter_time = time.time()
            if iter_time - start_time > TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING:
                empty_cuda()
                start_time = iter_time
        self.call_hook("after_val_epoch")
