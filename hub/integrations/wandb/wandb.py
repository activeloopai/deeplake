from hub.util.tag import process_hub_path
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


_READ_DATASETS = {}
_WRITTEN_DATASETS = {}
_CREATED_DATASETS = set()


def dataset_created(ds):
    path = ds.path
    _CREATED_DATASETS.add(path)


def dataset_loaded(ds):
    pass


def dataset_written(ds):
    path = ds.path
    run = wandb_run()
    if run:
        import wandb

        if run.id not in _WRITTEN_DATASETS:
            _WRITTEN_DATASETS.clear()
            _WRITTEN_DATASETS[run.id] = {}
        paths = _WRITTEN_DATASETS[run.id]
        if path not in paths:
            paths[path] = None

            # output_datasets = getattr(run.config, "output_datasets", [])  # uncomment after is merged
            try:
                output_datasets = run.config.input_datasets
            except (KeyError, AttributeError):
                output_datasets = []
            if path.startswith("hub://"):
                plat_link = _plat_link(path)
                if plat_link not in output_datasets:
                    run.log(
                        {
                            f"Hub Dataset [{path[len('hub://'):]}]": wandb.Html(
                                viz_html(path), False
                            )
                        }
                    )
                    output_datasets.append(plat_link)
                    run.config.input_datasets = output_datasets
            else:
                if path not in output_datasets:
                    output_datasets.append(path)
                    run.config.input_datasets = output_datasets
        if path in _CREATED_DATASETS:
            if not path.startswith("hub://"):
                orig_path = path
                path = "hub://" + path
            else:
                orig_path = path
            _, org, ds_name, _ = process_hub_path(orig_path)
            artifact_name = f"hub-{org}-{ds_name}"
            artifact = wandb.Artifact(artifact_name, "dataset")
            artifact.add_reference(path, name="url")
            wandb_info = ds.info.get("wandb") or {}
            wandb_info["created-by"] = {
                "run": {
                    "entity": run.entity,
                    "project": run.project,
                    "id": run.id,
                    "url": run.url,
                },
                "artifact": artifact_name,
            }
            ds.info["wandb"] = wandb_info
            ds.flush()
            _CREATED_DATASETS.remove(path)
            run.log_artifact(artifact)
    else:
        _CREATED_DATASETS.discard(path)


def dataset_read(ds):
    path = ds.path
    run = wandb_run()
    if run:
        if run.id not in _READ_DATASETS:
            _READ_DATASETS.clear()
            _READ_DATASETS[run.id] = {}
        paths = _READ_DATASETS[run.id]
        if path not in paths:
            paths[path] = None
            # input_datasets = getattr(run.config, "input_datasets", [])  # uncomment after is merged
            try:
                input_datasets = run.config.input_datasets
            except (KeyError, AttributeError):
                input_datasets = []
            if path.startswith("hub://"):
                plat_link = _plat_link(path)
                if plat_link not in input_datasets:
                    import wandb

                    run.log(
                        {
                            f"Hub Dataset [{path[len('hub://'):]}]": wandb.Html(
                                viz_html(path), False
                            )
                        }
                    )
                    input_datasets.append(plat_link)
                    run.config.input_datasets = input_datasets
            else:
                if path not in input_datasets:
                    input_datasets.append(path)
                    run.config.input_datasets = input_datasets
            wandb_info = ds.info.get("wandb")
            if wandb_info:
                run_and_artifact = wandb_info["created-by"]
                run_info = wandb_info["run"]
                artifact = run_and_artifact["artifact"]
                run.use_artifact(
                    f"{run_info['entity']}/{run_info['project']}/{artifact}:latest"
                )


def viz_html(hub_path: str):
#     return f"""
#       <div id='container'></div>
#   <script src="https://app.activeloop.ai/visualizer/vis.js"></script>
#   <script>
#     let container = document.getElementById('container')

#     window.vis.visualize('{hub_path}', null, null, container, {{
#       requireSignin: true
#     }})
#   </script>
#     """
    return f"""<iframe width=800 height=500 sandbox="allow-same-origin allow-scripts allow-popups allow-forms" src="https://app.activeloop.ai/visualizer/iframe?url={hub_path}" />"""


def _plat_link(hub_path: str):
    return f"https://app.activeloop.ai/{hub_path[len('hub://'):]}/"


def link_html(hub_path):
    return f"""<a href="{_plat_link(hub_path)}">{hub_path}</a>"""


if _WANDB_INSTALLED:
    add_create_dataset_hook(dataset_created, "wandb_dataset_create")
    add_load_dataset_hook(dataset_loaded, "wandb_dataset_load")
    add_write_dataset_hook(dataset_written, "wandb_dataset_write")
    add_read_dataset_hook(dataset_read, "wandb_dataset_read")
