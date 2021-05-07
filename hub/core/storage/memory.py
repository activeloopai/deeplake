from hub.core.storage.provider import Provider
import fsspec


class MemoryProvider(Provider):
    """
    Provider class for using the memory.
    """

    def __init__(self, root):
        """
        Initializes the MemoryProvider.

        Example:
            memory_provider = MemoryProvider("abcd/def")

        Args:
            root (str): the root of the provider.

        Returns:
            None

        Raises:
            None
        """
        self.mapper = fsspec.filesystem("memory").get_mapper(
            root=root, check=False, create=False
        )
