"""
Hub's Weights and Biases integration allows you to track and improve reproducibility of your machine learning experiments.
Hub will automatically push all information required to reproduce the snapshot of the data like your dataset's URI, commit ID, and view IDs of any 
views that you have used in your training workflow.

Learn more about Weights and Biases `here <https://wandb.ai>`_.

Logging Dataset Creation
~~~~~~~~~~~~~~~~~~~~~~~~
If you create a Hub dataset using any of the functions mentioned in :ref:`creating-datasets`, just perform a commit on the dataset to log its 
creation on W&B.

>>> run = wandb.init(project="hub_wandb", job_type="dataset_upload")
>>> ds = hub.empty("hub://fayazrahman4u/my_dataset") # create dataset
>>> ds.create_tensor("images", htype="image", sample_compression="jpg") # create a tensor
>>> ds.images.append(hub.read("files/images/dog.jpg")) # add a sample
>>> ds.commit("creation") # commit -> trigger logging
>>> run.finish()

NOTE:
    If you created your dataset using :meth:`hub.deepcopy`, perform the commit only if you have head changes.

NOTE:
    If you make changes to an existing dataset, commit the changes with an active Weights and Biases run to log it's state.

Logging Dataset Read
~~~~~~~~~~~~~~~~~~~~
A dataset read will be logged if you iterate over a dataset or call :meth:`Dataset.pytorch() <hub.core.dataset.Dataset.pytorch>` 
or :meth:`Tensor.numpy() <hub.core.tensor.Tensor.numpy>` on its tensors.

>>> run = wandb.init(project="hub_wandb", job_type="torch dataloader")
>>> train_loader = ds.pytorch()
>>> run.finish()

>>> run = wandb.init(project="hub_wandb", job_type="iteration")
>>> for sample in ds:
>>>     print(sample["images"].shape)
>>> run.finish()
"""

from typing import Dict, Set
from hub.util.tag import process_hub_path
from hub.util.hash import hash_inputs
from hub.constants import WANDB_JSON_FILENMAE
from hub.hooks import (
    add_create_dataset_hook,
    add_load_dataset_hook,
    add_write_dataset_hook,
    add_read_dataset_hook,
    add_commit_dataset_hook,
)
import importlib
import sys
import os
import json
import warnings
import hub

_WANDB_INSTALLED = bool(importlib.util.find_spec("wandb"))


def wandb_run():
    if not hub.constants.WANDB_INTEGRATION_ENABLED:
        return
    return getattr(sys.modules.get("wandb"), "run", None)


_READ_DATASETS: Dict[str, Set[str]] = {}
_WRITTEN_DATASETS: Dict[str, Set[str]] = {}


def dataset_created(ds):
    pass


def dataset_loaded(ds):
    pass


def artifact_name_from_ds_path(ds) -> str:
    path = ds.path
    hash = hash_inputs(path)
    if path.startswith("hub://"):
        _, org, ds_name, _ = process_hub_path(path)
        artifact_name = f"hub-{org}-{ds_name}"
        if "/.queries/" in path:
            vid = path.split("/.queries/", 1)[1]
            artifact_name += f"-view-{vid}"
    else:
        pfix = path.split("://", 1)[0] if "://" in path else "local"
        artifact_name = f"{pfix}"
    artifact_name += f"-commit-{ds.commit_id}"
    artifact_name += f"-{hash[:8]}"
    return artifact_name


def artifact_from_ds(ds):
    import wandb

    path = ds.path
    name = artifact_name_from_ds_path(ds)
    artifact = wandb.Artifact(name, "dataset")
    if "://" not in path and os.path.exists(path):
        path = "file://" + path
    artifact.add_reference(path, name="url")
    return artifact


def _is_public(ds_path):
    return True
    # TODO: We need api for this.
    try:
        hub.load(
            ds_path,
            token=hub.client.client.HubBackendClient(token="").request_auth_token(
                username="public", password=""
            ),
        )
        return True
    except Exception:
        return False


def get_ds_key(ds):
    entry = getattr(ds, "_view_entry", None)
    if entry:
        return hash_inputs(entry)
    return (hash_inputs(ds.path, ds.commit_id),)


def dataset_config(ds):
    if hasattr(ds, "_view_entry"):
        entry = ds._view_entry
        source_ds_path = entry.source_dataset_path
        commit_id = entry.info["source-dataset-version"]
        vid = entry.id
        ret = {
            "Dataset": source_ds_path,
            "Commit ID": commit_id,
            "View ID": vid,
        }
        if source_ds_path.startswith("hub://") and ds.path.startswith("hub://"):
            ret["URL"] = _plat_url(ds)
        q = entry.query
        if q:
            ret["Query"] = q
        if entry.virtual:
            ret["Index"] = ds.index.to_json()
        else:
            ret["Index"] = list(ds.sample_indices)
        return ret

    ret = {
        "Dataset": ds.path,
        "Commit ID": ds.commit_id,
    }
    if ds.path.startswith("hub://"):
        ret["URL"] = _plat_url(ds)
    if not ds.index.is_trivial():
        ret["Index"] = ds.index.to_json()
    q = getattr(ds, "_query", None)
    if q:
        ret["Query"] = q
    return ret


def log_dataset(dsconfig):
    # TODO: This is disabled until the embedded visualizer is actually useful for users.
    return
    url = dsconfig.get("URL")
    if not url:
        return
    import wandb

    run = wandb.run
    url_prefix = "https://app.activeloop.ai/"
    url = url[len(url_prefix) :]
    # TODO : commit and view id are not supported by visualizer. Remove below line once they are supported.
    url = "/".join(url.split("/")[:2])
    run.log({f"Hub Dataset - {url}": wandb.Html(_viz_html("hub://" + url))}, step=0)


def dataset_written(ds):
    pass


def dataset_committed(ds):
    run = wandb_run()
    key = get_ds_key(ds)
    if run:
        if run.id not in _WRITTEN_DATASETS:
            _WRITTEN_DATASETS.clear()
            _WRITTEN_DATASETS[run.id] = {}
        keys = _WRITTEN_DATASETS[run.id]
        if key not in keys:
            keys[key] = None
            try:
                output_datasets = run.config.output_datasets
            except (KeyError, AttributeError):
                output_datasets = []
            dsconfig = dataset_config(ds)
            output_datasets.append(dsconfig)
            log_dataset(dsconfig)
            run.config.output_datasets = output_datasets
            artifact = artifact_from_ds(ds)
            wandb_info = read_json(ds)
            try:
                commits = wandb_info["commits"]
            except KeyError:
                commits = {}
                wandb_info["commits"] = commits
            info = {}
            commits[ds.commit_id] = info
            info["created-by"] = {
                "run": {
                    "entity": run.entity,
                    "project": run.project,
                    "id": run.id,
                    "url": run.url,
                },
                "artifact": artifact.name,
            }
            write_json(ds, wandb_info)
            run.log_artifact(artifact)


def _filter_input_datasets(input_datasets):
    ret = []
    for i, dsconfig in enumerate(input_datasets):
        if "Index" not in dsconfig:
            rm = False
            for j, dsconfig2 in enumerate(input_datasets):
                if (
                    i != j
                    and dsconfig2["Dataset"] == dsconfig["Dataset"]
                    and dsconfig2["Commit ID"] == dsconfig["Commit ID"]
                ):
                    rm = True
                    break
            if not rm:
                ret.append(dsconfig)
        else:
            ret.append(dsconfig)
    return ret


def dataset_read(ds):
    run = wandb_run()
    if not run:
        return
    if run.id not in _READ_DATASETS:
        _READ_DATASETS.clear()
        _READ_DATASETS[run.id] = {}
    keys = _READ_DATASETS[run.id]
    key = get_ds_key(ds)
    idx = ds.index.to_json()
    if key not in keys or idx not in keys[key]:
        if key not in keys:
            keys[key] = []
        keys[key].append(idx)
        try:
            input_datasets = run.config.input_datasets
        except (KeyError, AttributeError):
            input_datasets = []
        dsconfig = dataset_config(ds)
        if dsconfig not in input_datasets:
            input_datasets.append(dsconfig)
            input_datasets = _filter_input_datasets(input_datasets)
            for config in input_datasets:
                log_dataset(config)
            run.config.input_datasets = input_datasets
        if run._settings.mode != "online":
            return
        if hasattr(ds, "_view_entry"):
            ds = ds._view_entry._src_ds
        wandb_info = read_json(ds).get("commits", {}).get(ds.commit_id)
        if wandb_info:
            try:
                run_and_artifact = wandb_info["created-by"]
                run_info = run_and_artifact["run"]
                artifact = run_and_artifact["artifact"]
                artifact_path = (
                    f"{run_info['entity']}/{run_info['project']}/{artifact}:latest"
                )
                run.use_artifact(artifact_path)
            except Exception as e:
                warnings.warn(
                    f"Wandb integration: Error while using wandb artifact: {e}"
                )
        else:
            # For datasets that were not created during a wandb run,
            # we want to "use" an artifact that is not logged by any run.
            # This is not possible with wandb yet.

            # artifact = artifact_from_ds(ds)
            # run.use_artifact(artifact)
            pass


def _viz_html(hub_path: str):
    if _is_public(hub_path):
        return f"""<iframe width="100%" height="100%" sandbox="allow-same-origin allow-scripts allow-popups allow-forms" src="https://app.activeloop.ai/visualizer/iframe?url={hub_path}" />"""
    return f"""
      <div id='container'></div>
  <script src="https://app.activeloop.ai/visualizer/vis.js"></script>
  <script>
    let container = document.getElementById('container')

    window.vis.visualize('{hub_path}', null, null, container, {{
      requireSignin: true
    }})
  </script>
    """


def _plat_url(ds, http=True):
    prefix = "https://app.activeloop.ai/" if http else "hub://"
    if hasattr(ds, "_view_entry"):
        entry = ds._view_entry
        _, org, ds_name, _ = process_hub_path(entry.source_dataset_path)
        commit_id = entry.info["source-dataset-version"]
        return f"{prefix}{org}/{ds_name}/{commit_id}?view={entry.id}"
    _, org, ds_name, _ = process_hub_path(ds.path)
    ret = f"{prefix}{org}/{ds_name}"
    if ds.commit_id:
        ret += f"/{ds.commit_id}"
    return ret


def link_html(hub_path):
    return f"""<a href="{_plat_url(hub_path)}">{hub_path}</a>"""


if _WANDB_INSTALLED:
    add_create_dataset_hook(dataset_created, "wandb_dataset_create")
    add_load_dataset_hook(dataset_loaded, "wandb_dataset_load")
    add_write_dataset_hook(dataset_written, "wandb_dataset_write")
    add_read_dataset_hook(dataset_read, "wandb_dataset_read")
    add_commit_dataset_hook(dataset_committed, "wandb_dataset_commit")


def read_json(ds):
    try:
        return json.loads(ds.base_storage[WANDB_JSON_FILENMAE].decode("utf-8"))
    except KeyError:
        return {}


def write_json(ds, dat):
    ds.base_storage[WANDB_JSON_FILENMAE] = json.dumps(dat).encode("utf-8")
