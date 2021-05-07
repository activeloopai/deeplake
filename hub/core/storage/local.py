from hub.core.storage.provider_mapper import ProviderMapper
import fsspec


class LocalProvider(ProviderMapper):
    """
    Provider class for using the local filesystem.
    """

    def __init__(self, root: str):
        """
        Initializes the LocalProvider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."

        Returns:
            None

        Raises:
            None
        """
        self.mapper = fsspec.filesystem("file").get_mapper(
            root, check=False, create=False
        )
