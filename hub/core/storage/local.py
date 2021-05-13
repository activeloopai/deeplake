from hub.core.storage.mapped_provider import MappedProvider
import os
import shutil
import fsspec  # type: ignore


class LocalProvider(MappedProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, root: str):
        """Initializes the LocalProvider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."
        """
        self.mapper = fsspec.filesystem("file").get_mapper(
            root, check=False, create=False
        )

    def clear(self):
        # shutil is much faster than mapper.clear()
        if os.path.exists(self.mapper.root):
            shutil.rmtree(self.mapper.root)
