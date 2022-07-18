from typing import Dict, Optional, Any
from hub.client.log import logger

from hub.util.tag import process_hub_path
from hub.util.path import get_org_id_and_ds_name, is_hub_cloud_path
from hub.constants import HUB_CLOUD_DEV_USERNAME


class ViewEntry:
    """Represents a view saved inside a dataset."""

    def __init__(self, info: Dict, dataset, external: bool = False):
        self.info = info
        self._ds = dataset
        self._external = external

    def __getitem__(self, key: str):
        return self.info[key]

    def get(self, key: str, default: Optional[Any] = None):
        return self.info.get(key, default)

    @property
    def id(self) -> str:
        """Returns id of the view."""
        return self.info["id"].split("]")[-1]

    @property
    def query(self) -> Optional[str]:
        return self.info.get("query")

    @property
    def message(self) -> str:
        """Returns the message with which the view was saved."""
        return self.info.get("message", "")

    def __str__(self):
        return f"View(id='{self.id}', message='{self.message}', virtual={self.virtual})"

    __repr__ = __str__

    @property
    def virtual(self) -> bool:
        return self.info["virtual-datasource"]

    def load(self, verbose=True):
        "Loads the view and returns the `hub.Dataset`."
        ds = self._ds._sub_ds(
            ".queries/" + (self.info.get("path") or self.info["id"]),
            lock=False,
            verbose=False,
        )
        url = ds.path
        org_id, ds_name = get_org_id_and_ds_name(ds.path)
        if self.virtual:
            ds = ds._get_view(inherit_creds=not self._external)
        ds._view_entry = self
        msg = "This dataset can be visualized in Jupyter Notebook by ds.visualize()"
        url = f"https://app.activeloop.ai/{org_id}/{ds_name}"
        if verbose:
            if "/.queries/" in url:  # Is a view
                if "/queries/" in url:  # Stored in user queries ds
                    entry = getattr(ds, "_view_entry", None)
                    if entry:
                        source_ds_url = entry.info["source-dataset"]
                        if is_hub_cloud_path(source_ds_url):
                            org_id, ds_name = get_org_id_and_ds_name(source_ds_url)
                            source_ds_url = (
                                f"https://app.activeloop.ai/{org_id}/{ds_name}"
                            )
                            view_id = url.split("/.queries/", 1)[1]
                            if view_id.endswith("_OPTIMIZED"):
                                view_id = view_id[: -len("_OPTIMIZED")]
                            url = source_ds_url + "?view=" + view_id
                            logger.info(msg + " or at " + url)
                        else:
                            logger.info(msg + ".")
                    else:
                        logger.info(msg + ".")
                else:  # Stored in ds
                    ds_url, view_id = url.split("/.queries/", 1)
                    if view_id.endswith("_OPTIMIZED"):
                        view_id = view_id[: -len("_OPTIMIZED")]
                    url = ds_url + "?view=" + view_id
                    logger.info(msg + " or at " + url)
        return ds

    def optimize(
        self, unlink=True, num_workers=0, scheduler="threaded", progressbar=True
    ):
        """Optimizes the dataset view by copying and rechunking the required data. This is necessary to achieve fast streaming
            speeds when training models using the dataset view. The optimization process will take some time, depending on
            the size of the data.

        Args:
            unlink (bool): - If True, this unlinks linked tensors (if any) by copying data from the links to the view.
                    - This does not apply to linked videos. Set `hub.\0constants._UNLINK_VIDEOS` to `True` to change this behavior.
            num_workers (int): Number of workers to be used for the optimization process. Defaults to 0.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Only applicable if `optimize=True`. Defaults to 'threaded'.
            progressbar (bool): Whether to display a progressbar.

        Returns:
            `hub.core.dataset.view_entry.ViewEntry`

        Examples:
            ```
            # save view
            ds[:10].save_view(view_id="first_10")

            # optimize view
            ds.get_view("first_10").optimize()

            # load optimized view
            ds.load_view("first_10")
            ```
        """
        self.info = self._ds._optimize_saved_view(
            self.info["id"],
            external=self._external,
            unlink=unlink,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
        )
        return self

    def delete(self):
        """Deletes the view."""
        self._ds.delete_view(id=self.info["id"])
