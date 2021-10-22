from hub.core.dataset import Dataset
from hub.client.client import HubBackendClient
from hub.util.path import is_hub_cloud_path

from warnings import warn


class HubCloudDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: better logging? (so we don't spam tests)
        if not self.is_actually_cloud:
            warn(
                f'Created a hub cloud dataset @ "{self.path}" which does not have the "{HUB_CLOUD_PREFIX}" prefix. Note: this dataset should only be used for testing!'
            )

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

    def _set_derived_attributes(self):
        super()._set_derived_attributes()

        if self.is_actually_cloud():
            split_path = self.path.split("/")
            self.org_id, self.ds_name = split_path[2], split_path[3]
        else:
            self.org_id, self.ds_name = None, None

        self.client = HubBackendClient(token=self._token)

    def _populate_meta(self):
        super()._populate_meta()

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
        """Users trying to access this dataset must agree to these terms first.

        Note:
            Only applicable for public hub cloud datasets (path begins with "hub://".
        """

        # if not self.path.startswith("hub://"):
        # raise Exception("Terms of access can only be applied to datasets stored in hub cloud (path begins with hub://")

        print(self.client)
