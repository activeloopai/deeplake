from hub.util.tag import process_hub_path
from hub.util.hash import hash_inputs
from hub.hooks import (
    add_create_dataset_hook,
    add_load_dataset_hook,
    add_write_dataset_hook,
    add_read_dataset_hook,
)
import importlib
import sys
import warnings


_WANDB_INSTALLED = bool(importlib.util.find_spec("wandb"))


def wandb_run():
    return getattr(sys.modules.get("wandb"), "run", None)


_READ_DATASETS = {}
_WRITTEN_DATASETS = {}
_CREATED_DATASETS = set()


def dataset_created(ds):
    _CREATED_DATASETS.add(get_ds_key(ds))


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
        pfix = pfix.split("://", 1)[0] if "://" in path else "local"
        artifact_name = f"{pfix}"
    artifact_name += f"-commit-{ds.commit_id}"
    if ds.has_head_changes:
        warnings.warn(
            "Creating artifact for dataset with head changes. State of the dataset during artifact consumption will be differnt from the state when it was logged."
        )
        artifact_name += f"-has-head-changes"
    artifact_name += f"-{hash[:8]}"
    return artifact_name


def artifact_from_ds(ds):
    import wandb

    path = ds.path
    name = artifact_name_from_ds_path(ds)
    artifact = wandb.Artifact(name, "dataset")
    artifact.add_reference(path, name="url")
    return artifact


def get_ds_key(ds):
    return hash_inputs(ds.path, ds.commit_id)


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
            _, org_id, ds_name, _ = process_hub_path(source_ds_path)
            ret[
                "URL"
            ] = f"https://app.activeloop.ai/{org_id}/{ds_name}/{commit_id}?view={vid}"
        q = entry.query
        if q:
            ret["query"] = q
        return ret

    ret = {
        "Dataset": ds.path,
        "Commit ID": ds.commit_id,
    }
    if ds.path.startswith("hub://"):
        ret["URL"] = (
            "https://app.activeloop.ai/" + ds.path[len("hub://") :] + "/" + ds.commit_id
        )
    if not ds.index.is_trivial():
        ret["index"] = ds.index.to_json()
    return ret


def dataset_written(ds):
    path = ds.path
    run = wandb_run()
    key = get_ds_key(ds)
    if run:
        import wandb

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
            path = dsconfig["Dataset"]
            if path.startswith("hub://"):
                import wandb

                run.log(
                    {
                        f"Hub Dataset [{path[len('hub://'):]}]": wandb.Html(
                            viz_html(path), False
                        )
                    }
                )

            output_datasets.append(dsconfig)
            run.config.output_datasets = output_datasets
        if key in _CREATED_DATASETS:
            artifact = artifact_from_ds(ds)
            wandb_info = ds.info.get("wandb") or {"commits": {}}
            commits = wandb_info["commits"]
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
            ds.info["wandb"] = wandb_info
            ds.flush()
            _CREATED_DATASETS.remove(key)
            run.log_artifact(artifact)
    else:
        _CREATED_DATASETS.discard(key)


def dataset_read(ds):
    path = ds.path
    run = wandb_run()
    if not run:
        return
    if run.id not in _READ_DATASETS:
        _READ_DATASETS.clear()
        _READ_DATASETS[run.id] = {}
    keys = _READ_DATASETS[run.id]
    key = get_ds_key(ds)
    if key not in keys:
        keys[key] = None
        try:
            input_datasets = run.config.input_datasets
        except (KeyError, AttributeError):
            input_datasets = []
        dsconfig = dataset_config(ds)
        path = dsconfig["Dataset"]
        if dsconfig not in input_datasets:

            if path.startswith("hub://"):
                import wandb

                run.log(
                    {
                        f"Hub Dataset [{path[len('hub://'):]}]": wandb.Html(
                            viz_html(path), False
                        )
                    }
                )

            input_datasets.append(dsconfig)
            run.config.input_datasets = input_datasets

        # TODO consider optimized datasets:
        wandb_info = ds.info.get("wandb", {}).get("commits", {}).get(ds.commit_id)
        if wandb_info:
            run_and_artifact = wandb_info["created-by"]
            run_info = run_and_artifact["run"]
            artifact = run_and_artifact["artifact"]
            run.use_artifact(
                f"{run_info['entity']}/{run_info['project']}/{artifact}:latest"
            )
        else:
            # For datasets that were not created during a wandb run,
            # we want to "use" an artifact that is not logged by any run.
            # This is not possible with wandb yet.

            # artifact = artifact_from_ds(ds)
            # run.use_artifact(artifact)
            pass


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
    return f"""<iframe width="100%" height="100%" sandbox="allow-same-origin allow-scripts allow-popups allow-forms" src="https://app.activeloop.ai/visualizer/iframe?url={hub_path}" />"""


def _plat_link(ds):
    path = ds.path
    if "/.queries/" in path:
        if "/queries/" in path:
            entry = getattr((ds._view_base or ds), "_view_entry")
            if not entry:
                _, org, ds_name, _ = process_hub_path(path)
                return f"https://app.activeloop.ai/{org}/{ds_name}"
            source_ds_path = entry.info["source-dataset"]
            commit_id = entry.info["source-dataset-version"]
            _, org, ds_name, _ = process_hub_path(source_ds_path)
            return (
                f"https://app.activeloop.ai/{org}/{ds_name}/{commit_id}?view={entry.id}"
            )
        else:
            _, org, ds_name, _ = process_hub_path(path)
            vid = path.split("/.queries/")[1]
            return (
                f"https://app.activeloop.ai/{org}/{ds_name}/{ds.commit_id}?view={vid}"
            )
    _, org, ds_name, _ = process_hub_path(path)
    return f"https://app.activeloop.ai/{org}/{ds_name}/{ds.commit_id}"


def link_html(hub_path):
    return f"""<a href="{_plat_link(hub_path)}">{hub_path}</a>"""


if _WANDB_INSTALLED:
    add_create_dataset_hook(dataset_created, "wandb_dataset_create")
    add_load_dataset_hook(dataset_loaded, "wandb_dataset_load")
    add_write_dataset_hook(dataset_written, "wandb_dataset_write")
    add_read_dataset_hook(dataset_read, "wandb_dataset_read")
