from typing import Optional
from hub.constants import AGREEMENT_FILENAME, HUB_CLOUD_DEV_USERNAME
from hub.core.dataset import Dataset
from hub.client.client import HubBackendClient
from hub.client.log import logger
from hub.util.agreement import handle_dataset_agreement
from hub.util.path import is_hub_cloud_path
from warnings import warn


class HubCloudDataset(Dataset):
    def __init__(self, path, *args, **kwargs):
        self._client = None
        self.path = path
        self.org_id, self.ds_name = None, None
        self._set_org_and_name()

        super().__init__(*args, **kwargs)

        if self.is_actually_cloud:
            handle_dataset_agreement(self.agreement, path, self.ds_name, self.org_id)
        else:
            # NOTE: this can happen if you override `hub.core.dataset.FORCE_CLASS`
            warn(
                f'Created a hub cloud dataset @ "{self.path}" which does not have the "hub://" prefix. Note: this dataset should only be used for testing!'
            )

    @property
    def client(self):
        if self._client is None:
            self._client = HubBackendClient(token=self._token)
        return self._client

    @property
    def is_actually_cloud(self) -> bool:
        """Datasets that are connected to hub cloud can still technically be stored anywhere.
        If a dataset is hub cloud but stored without `hub://` prefix, it should only be used for testing.
        """

        return is_hub_cloud_path(self.path)

    @property
    def token(self):
        """Get attached token of the dataset"""
        if self._token is None:
            self._token = self.client.get_token()
        return self._token

    def _set_org_and_name(self):
        if self.is_actually_cloud:
            split_path = self.path.split("/")
            self.org_id, self.ds_name = split_path[2], split_path[3]
        else:
            # if this dataset isn't actually pointing to a datset in the cloud
            # a.k.a this dataset is trying to simulate a hub cloud dataset
            # it's safe to assume they want to use the dev org
            self.org_id = HUB_CLOUD_DEV_USERNAME
            self.ds_name = self.path.replace("/", "_").replace(".", "")

    def _register_dataset(self):
        # called in super()._populate_meta

        self.client.create_dataset_entry(
            self.org_id,
            self.ds_name,
            self.version_state["meta"].__getstate__(),
            public=self.public,
        )

    def make_public(self):
        if not self.public:
            self.client.update_privacy(self.org_id, self.ds_name, public=True)
            self.public = True

    def make_private(self):
        if self.public:
            self.client.update_privacy(self.org_id, self.ds_name, public=False)
            self.public = False

    def delete(self, large_ok=False):
        super().delete(large_ok=large_ok)

        self.client.delete_dataset_entry(self.org_id, self.ds_name)
        logger.info(f"Hub Dataset {self.path} successfully deleted.")

    @property
    def agreement(self) -> Optional[str]:
        try:
            agreement_bytes = self.storage[AGREEMENT_FILENAME]
            return agreement_bytes.decode("utf-8")
        except KeyError:
            return None

    def add_agreeement(self, agreement: str):
        self.storage.check_readonly()
        self.storage[AGREEMENT_FILENAME] = agreement.encode("utf-8")
