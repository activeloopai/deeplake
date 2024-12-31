import mmcv  # type: ignore

import torch
import logging
from mmcv import runner
from torch.utils.data import DataLoader

import time
import warnings
from typing import List, Tuple, Optional
from deeplake.integrations.constants import TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING


def empty_cuda():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return


@runner.RUNNERS.register_module()
class DeeplakeIterBasedRunner(runner.IterBasedRunner):
    def __init__(self, **kwargs):
        self.force_cleanup = kwargs.pop("force_cleanup", True)
        super().__init__(**kwargs)

    def run(
        self,
        data_loaders: List[DataLoader],
        workflow: List[Tuple[str, int]],
        max_iters: Optional[int] = None,
        **kwargs,
    ) -> None:
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                "setting max_iters in run is deprecated, "
                "please set max_iters in runner_config",
                DeprecationWarning,
            )
            self._max_iters = max_iters
        assert (
            self._max_iters is not None
        ), "max_iters must be specified during instantiation"

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s",
            runner.utils.get_host_info(),
            work_dir,
        )
        self.logger.info(
            "Hooks will be executed in the following order:\n%s", self.get_hook_info()
        )
        self.logger.info("workflow: %s, max: %d iters", workflow, self._max_iters)
        self.call_hook("before_run")

        iter_loaders = [runner.IterLoader(x) for x in data_loaders]

        self.call_hook("before_epoch")

        formatter = logging.Formatter("%(relative)ss")
        start_time = time.time()

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.format(mode)
                    )
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == "train" and self.iter >= self._max_iters:
                        break

                    iter_time = time.time()

                    if (
                        self.force_cleanup
                        and iter_time - start_time
                        > TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING
                    ):
                        empty_cuda()
                        start_time = iter_time
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_epoch")
        self.call_hook("after_run")


@runner.RUNNERS.register_module()
class DeeplakeEpochBasedRunner(runner.EpochBasedRunner):
    def __init__(self, **kwargs):
        self.force_cleanup = kwargs.pop("force_cleanup", True)
        super().__init__(**kwargs)

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
            if (
                self.force_cleanup
                and iter_time - start_time > TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING
            ):
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
            if (
                self.force_cleanup
                and iter_time - start_time > TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING
            ):
                empty_cuda()
                start_time = iter_time
        self.call_hook("after_val_epoch")
