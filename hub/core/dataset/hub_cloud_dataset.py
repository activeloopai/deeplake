from hub.core.dataset import Dataset
from hub.client.client import HubBackendClient
from hub.client.log import logger
from hub.client.utils import get_user_name
from hub.util.path import is_hub_cloud_path

from warnings import warn


class HubCloudDataset(Dataset):
    def __init__(self, path, *args, **kwargs):
        self._client = None
        self.path = path
        self.org_id, self.ds_name = None, None
        self._set_org_and_name()

        super().__init__(*args, **kwargs)

        self._register_dataset()
        # NOTE: this can happen if you override `hub.core.dataset.FORCE_CLASS`
        if not self.is_actually_cloud:
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

    def check_credentials(self):
        """If terms of access are unagreed to, this method will raise an error and trigger
        user-interaction requirement for agreeing. It's basically just an alias for `get_dataset_credentials`
        """

        self.client.get_dataset_credentials(self.org_id, self.ds_name)

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
            self.org_id = get_user_name()
            self.ds_name = self.path.replace("/", "_").replace(".", "")


    def _is_registered(self) -> bool:
        return self.client.is_dataset_registered(self.org_id, self.ds_name)

    def _register_dataset(self):
        if self._is_registered():
            self.check_credentials()

        else:
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

    def add_terms_of_access(self, terms: str):
        """Users must agree to these terms before being able to access this dataset."""

        self.check_credentials()
        self.client.add_terms_of_access(self.org_id, self.ds_name, terms)
