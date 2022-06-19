from hub.hooks import (
    add_create_dataset_hook,
    add_load_dataset_hook,
    add_write_dataset_hook,
    add_read_dataset_hook,
)
import importlib
import sys


_WANDB_INSTALLED = bool(importlib.util.find_spec("wandb"))

def wandb_run():
    return getattr(sys.modules.get("wandb"), "run", None)


def dataset_created(path: str):
    pass


def dataset_loaded(path: str):
    pass


def dataset_written(path: str):
    pass


_ACTIVE_DATASET_CACHE = {}


def dataset_read(path: str):
    run = wandb_run()
    if run:
        if run.id not in _ACTIVE_DATASET_CACHE:
            _ACTIVE_DATASET_CACHE.clear()
            _ACTIVE_DATASET_CACHE[run.id] = {}
        if path not in _ACTIVE_DATASET_CACHE:
            _ACTIVE_DATASET_CACHE[path] = None
            run.config.input_datasets = [k for k in _ACTIVE_DATASET_CACHE]


def viz_html(hub_path):
    return f"""<iframe width=800 height=500 src="https://app.activeloop.ai/visualizer/iframe?url={hub_path}" />"""


if _WANDB_INSTALLED:
    add_create_dataset_hook(dataset_created, "wandb_dataset_create")
    add_load_dataset_hook(dataset_loaded, "wandb_dataset_load")
    add_write_dataset_hook(dataset_written, "wandb_dataset_write")
    add_read_dataset_hook(dataset_read, "wandb_dataset_read")
