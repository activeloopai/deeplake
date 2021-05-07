from hub.core.storage.provider import Provider
import fsspec


class LocalProvider(Provider):
    """
    Provider class for using the local filesystem.
    """

    def __init__(self, root: str):
        """
        Initializes the LocalProvider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): the root of the provider.

        Returns:
            None

        Raises:
            None
        """
        self.mapper = fsspec.filesystem("file").get_mapper(
            root, check=False, create=False
        )
