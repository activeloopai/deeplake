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
        import wandb
        if run.id not in _ACTIVE_DATASET_CACHE:
            _ACTIVE_DATASET_CACHE.clear()
            _ACTIVE_DATASET_CACHE[run.id] = {}
        if path not in _ACTIVE_DATASET_CACHE:
            paths = _ACTIVE_DATASET_CACHE[run.id]
            is_hub_path = path.startswith("hub://")
            if is_hub_path:
                path = path + "  -  " + _plat_link(path)
            paths[path] = None
            run.config.input_datasets = list(paths)
            if path.startswith("hub://"):
                run.log({f"Dataset [{path}]": wandb.Html(viz_html(path))})


def viz_html(hub_path: str):
    return f"""<iframe width=800 height=500 src="https://app.activeloop.ai/visualizer/iframe?url={hub_path}" />"""


def _plat_link(hub_path: str):
    return f"https://app.activeloop.ai/{hub_path[len('hub://'):]}/"

def link_html(hub_path):
    return f"""<a href="{_plat_link(hub_path)}">{hub_path}</a>"""

if _WANDB_INSTALLED:
    add_create_dataset_hook(dataset_created, "wandb_dataset_create")
    add_load_dataset_hook(dataset_loaded, "wandb_dataset_load")
    add_write_dataset_hook(dataset_written, "wandb_dataset_write")
    add_read_dataset_hook(dataset_read, "wandb_dataset_read")
