from typing import Dict

import deeplake
from deeplake.client.client import DeepLakeBackendClient


class CfgObject:
    def __init__(self, cfg: Dict[str, deeplake.core.dataset.Dataset]) -> None:
        """
        Initializes a CfgObject instance.

        Args:
            cfg (Dict[str, deeplake.core.dataset.Dataset]): The configuration dictionary.
        """
        self.cfg = cfg
        self._init_credentials()
        self._init_token()
        self._init_uname()

    def _init_credentials(self) -> None:
        """Initializes credentials from the configuration."""
        self.creds = self.cfg.get("deeplake_credentials", {})

    def _init_token(self) -> None:
        """Initializes the token from the credentials."""
        self.token = self.creds.get("token", None)

    def _init_uname(self) -> None:
        """
        Initializes the username and requests an authentication token if the token is not provided.
        """
        if self.token is None:
            uname = self.creds.get("username")
            if uname is not None:
                pword = self.creds["password"]
                client = DeepLakeBackendClient()
                self.token = client.request_auth_token(username=uname, password=pword)

    def load(self) -> deeplake.core.dataset.Dataset:
        """
        Loads a dataset from DeepLake.

        Returns:
            deeplake.core.dataset.Dataset: The loaded dataset or query result.
        """
        ds_path = self.cfg["deeplake_path"]

        self.ds = deeplake.load(ds_path, token=self.token, read_only=True)
        self.load_commit()
        return self.load_view_or_query()

    def load_commit(self) -> None:
        """
        Checks out the specified commit from the dataset.
        """
        deeplake_commit = self.cfg.get("deeplake_commit")
        if deeplake_commit:
            self.ds.checkout(deeplake_commit)

    def load_view_or_query(self) -> deeplake.core.dataset.Dataset:
        """
        Loads a view or performs a query on the dataset.

        Returns:
            deeplake.core.dataset.Dataset: The loaded view or query result.

        Raises:
            Exception: If both a view_id and a query are specified simultaneously.
        """
        deeplake_view_id = self.cfg.get("deeplake_view_id")
        deeplake_query = self.cfg.get("deeplake_query")

        if deeplake_view_id and deeplake_query:
            raise Exception(
                "A query and view_id were specified simultaneously for a dataset in the config. Please specify either the deeplake_query or the deeplake_view_id."
            )

        if deeplake_view_id:
            return self.ds.load_view(id=deeplake_view_id)
        return self.ds.query(deeplake_query)
