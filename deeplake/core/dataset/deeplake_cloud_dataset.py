import posixpath
from typing import Any, Dict, Optional, Union
from deeplake.client.utils import get_user_name
from deeplake.constants import HUB_CLOUD_DEV_USERNAME
from deeplake.core.dataset import Dataset
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.bugout_reporter import deeplake_reporter
from deeplake.util.exceptions import (
    InvalidSourcePathError,
    RenameError,
    ReadOnlyModeError,
)
from deeplake.util.link import save_link_creds
from deeplake.util.path import is_hub_cloud_path
from deeplake.util.tag import process_hub_path
from deeplake.util.logging import log_visualizer_link
from deeplake.util.storage import storage_provider_from_hub_path
from warnings import warn
import time
import deeplake


class DeepLakeCloudDataset(Dataset):
    """Subclass of :class:`Dataset`. Deep Lake cloud datasets are those datasets which are stored on Activeloop servers, their paths look like:
    ``hub://username/dataset_name``."""

    def _first_load_init(self, verbose=True):
        self._set_org_and_name()
        if self.is_first_load:
            if self.is_actually_cloud:
                if self.verbose and verbose:
                    log_visualizer_link(self.path)
            else:
                # NOTE: this can happen if you override `deeplake.core.dataset.FORCE_CLASS`
                warn(
                    f'Created a Deep Lake cloud dataset @ "{self.path}" which does not have the "hub://" prefix. Note: this dataset should only be used for testing!'
                )
            self.link_creds.populate_all_managed_creds(verbose=self.verbose and verbose)

    @property
    def client(self):
        if self._client is None:
            self.__dict__["_client"] = DeepLakeBackendClient(token=self._token)
        return self._client

    @property
    def is_actually_cloud(self) -> bool:
        """Datasets that are connected to Deep Lake cloud can still technically be stored anywhere.
        If a dataset is in Deep Lake cloud but stored without ``hub://`` prefix, it should only be used for testing.
        """
        return is_hub_cloud_path(self.path)  # type: ignore

    @property
    def token(self):
        """Get attached token of the dataset"""
        if self._token is None:
            self.__dict__["_token"] = self.client.get_token()
        return self._token

    def _set_org_and_name(self):
        if self.is_actually_cloud:
            if self.org_id is not None:
                return
            _, org_id, ds_name, subdir = process_hub_path(self.path)
            if subdir:
                ds_name += "/" + subdir
        else:
            # if this dataset isn't actually pointing to a datset in the cloud
            # a.k.a this dataset is trying to simulate a Deep Lake cloud dataset
            # it's safe to assume they want to use the dev org
            org_id = HUB_CLOUD_DEV_USERNAME
            ds_name = self.path.replace("/", "_").replace(".", "")
        self.__dict__["org_id"] = org_id
        self.__dict__["ds_name"] = ds_name

    def _is_sub_ds(self):
        return "/" in self.ds_name

    def _register_dataset(self):
        # called in super()._populate_meta
        self._set_org_and_name()
        if self._is_sub_ds():
            return
        self.client.create_dataset_entry(
            self.org_id,
            self.ds_name,
            self.version_state["meta"].__getstate__(),
            public=self.public,
        )
        self._send_dataset_creation_event()

    def _send_event(
        self,
        event_id: str,
        event_group: str,
        deeplake_meta: Dict[str, Any],
        has_head_changes: Optional[bool] = None,
    ):
        username = get_user_name()
        has_head_changes = (
            has_head_changes if has_head_changes is not None else self.has_head_changes
        )
        common_meta = {
            "username": username,
            "commit_id": self.commit_id,
            "pending_commit_id": self.pending_commit_id,
            "has_head_changes": has_head_changes,
        }
        deeplake_meta.update(common_meta)
        event_dict = {
            "id": event_id,
            "event_group": event_group,
            "ts": time.time(),
            "deeplake_meta": deeplake_meta,
            "creator": "Deep Lake",
        }
        deeplake.event_queue.put((self.client, event_dict))

    def _send_query_progress(
        self,
        query_id: str = "",
        query_text: str = "",
        start: bool = False,
        end: bool = False,
        progress: int = 0,
        status="",
    ):
        deeplake_meta = {
            "query_id": query_id,
            "query_text": query_text,
            "progress": progress,
            "start": start,
            "end": end,
            "status": status,
        }
        event_id = f"{self.org_id}/{self.ds_name}.query"
        self._send_event(
            event_id=event_id, event_group="query", deeplake_meta=deeplake_meta
        )

    def _send_compute_progress(
        self,
        compute_id: str = "",
        start: bool = False,
        end: bool = False,
        progress: int = 0,
        status="",
    ):
        deeplake_meta = {
            "compute_id": compute_id,
            "progress": progress,
            "start": start,
            "end": end,
            "status": status,
        }
        event_id = f"{self.org_id}/{self.ds_name}.compute"
        self._send_event(
            event_id=event_id, event_group="hub_compute", deeplake_meta=deeplake_meta
        )

    def _send_pytorch_progress(
        self,
        pytorch_id: str = "",
        start: bool = False,
        end: bool = False,
        progress: int = 0,
        status="",
    ):
        deeplake_meta = {
            "pytorch_id": pytorch_id,
            "progress": progress,
            "start": start,
            "end": end,
            "status": status,
        }
        event_id = f"{self.org_id}/{self.ds_name}.pytorch"
        self._send_event(
            event_id=event_id, event_group="pytorch", deeplake_meta=deeplake_meta
        )

    def _send_commit_event(self, commit_message: str, commit_time, author: str):
        # newly created commit can't have head_changes
        deeplake_meta = {
            "commit_message": commit_message,
            "commit_time": str(commit_time),
            "author": author,
        }
        event_id = f"{self.org_id}/{self.ds_name}.commit"
        self._send_event(
            event_id=event_id,
            event_group="dataset_commit",
            deeplake_meta=deeplake_meta,
            has_head_changes=False,
        )

    def _send_branch_creation_event(self, branch_name: str):
        deeplake_meta = {"branch_name": branch_name}
        event_id = f"{self.org_id}/{self.ds_name}.branch_created"
        self._send_event(
            event_id=event_id,
            event_group="dataset_branch_creation",
            deeplake_meta=deeplake_meta,
            has_head_changes=False,
        )

    def _send_dataset_creation_event(self):
        deeplake_meta = {}
        event_id = f"{self.org_id}/{self.ds_name}.dataset_created"
        self._send_event(
            event_id=event_id,
            event_group="dataset_creation",
            deeplake_meta=deeplake_meta,
            has_head_changes=False,
        )

    def make_public(self):
        self._set_org_and_name()
        if not self.public:
            self.client.update_privacy(self.org_id, self.ds_name, public=True)
            self.__dict__["public"] = True

    def make_private(self):
        self._set_org_and_name()
        if self.public:
            self.client.update_privacy(self.org_id, self.ds_name, public=False)
            self.__dict__["public"] = False

    def delete(self, large_ok=False):
        super().delete(large_ok=large_ok)
        if self._is_sub_ds():
            return
        self.client.delete_dataset_entry(self.org_id, self.ds_name)

    def rename(self, path):
        self.storage.check_readonly()
        path = path.rstrip("/")
        root, new_name = posixpath.split(path)
        if root != posixpath.split(self.path)[0]:
            raise RenameError
        self.client.rename_dataset_entry(self.org_id, self.ds_name, new_name)

        self.ds_name = new_name
        self.path = path

    def __getstate__(self) -> Dict[str, Any]:
        self._set_org_and_name()
        state = super().__getstate__()
        return state

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        self._client = None
        self._first_load_init(verbose=False)

    def visualize(
        self, width: Union[int, str, None] = None, height: Union[int, str, None] = None
    ):
        from deeplake.visualizer import visualize

        deeplake_reporter.feature_report(feature_name="visualize", parameters={})
        visualize(self.path, token=self.token, width=width, height=height)

    def add_creds_key(self, creds_key: str, managed: bool = False):
        """Adds a new creds key to the dataset. These keys are used for tensors that are linked to external data.

        Examples:

            >>> # create/load a dataset
            >>> ds = deeplake.dataset("hub://username/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")

        Args:
            creds_key (str): The key to be added.
            managed (bool): If ``True``, the creds corresponding to the key will be fetched from activeloop platform.
                Note, this is only applicable for datasets that are connected to activeloop platform.
                Defaults to ``False``.
        """
        self.link_creds.add_creds_key(creds_key, managed=managed)
        save_link_creds(self.link_creds, self.storage)
        self.link_creds.warn_missing_managed_creds()

    def update_creds_key(self, old_creds_key: str, new_creds_key: str):
        """Replaces the old creds key with the new creds key. This is used to replace the creds key used for external data."""
        if old_creds_key in self.link_creds.managed_creds_keys:
            raise ValueError(
                f"""Cannot update managed creds key directly. If you want to update it, follow these steps:-
                1. ds.change_creds_management("{old_creds_key}", False)
                2. ds.update_creds_key("{old_creds_key}", "{new_creds_key}")
                3. [OPTIONAL] ds.change_creds_management("{new_creds_key}", True)
                """
            )
        super().update_creds_key(old_creds_key, new_creds_key)
        self.link_creds.warn_missing_managed_creds()

    def change_creds_management(self, creds_key: str, managed: bool):
        """Changes the management status of the creds key.

        Args:
            creds_key (str): The key whose management status is to be changed.
            managed (bool): The target management status. If ``True``, the creds corresponding to the key will be fetched from activeloop platform.

        Raises:
            ValueError: If the dataset is not connected to activeloop platform.
            KeyError: If the creds key is not present in the dataset.

        Examples:

            >>> # create/load a dataset
            >>> ds = deeplake.dataset("hub://username/dataset")
            >>> # add a new creds key
            >>> ds.add_creds_key("my_s3_key")
            >>> # Populate the name added with creds dictionary
            >>> # These creds are only present temporarily and will have to be repopulated on every reload
            >>> ds.populate_creds("my_s3_key", {})
            >>> # Change the management status of the key to True. Before doing this, ensure that the creds have been created on activeloop platform
            >>> # Now, this key will no longer use the credentials populated in the previous step but will instead fetch them from activeloop platform
            >>> # These creds don't have to be populated again on every reload and will be fetched every time the dataset is loaded
            >>> ds.change_creds_management("my_s3_key", True)

        """

        key_index = self.link_creds.creds_mapping[creds_key] - 1
        changed = self.link_creds.change_creds_management(creds_key, managed)
        if changed:
            save_link_creds(
                self.link_creds, self.storage, managed_info=(managed, key_index)
            )

    def _load_link_creds(self):
        """Loads the link creds from the storage."""
        super()._load_link_creds()
        if self.link_creds.client is None:
            self._set_org_and_name()
            self.link_creds.org_id = self.org_id
            self.link_creds.client = self.client

    def _temp_write_access(self):
        if not self.read_only or self._locked_out:
            return memoryview(b"")  # No-op context manager

        class _TmpWriteAccess:
            def __enter__(self2):
                self2.orig_storage = self.base_storage
                storage = storage_provider_from_hub_path(
                    self.path, read_only=False, token=self._token
                )
                if storage.read_only:
                    raise ReadOnlyModeError(
                        f"You do not have permission to materialize views in this dataset ({self.path})."
                    )
                self.base_storage = storage

            def __exit__(self2, *_, **__):
                self.base_storage = self2.orig_storage

        return _TmpWriteAccess()

    def connect(self, *args, **kwargs):
        raise InvalidSourcePathError(
            f"Source dataset is already accessible via Deep Lake path {self.path}"
        )
