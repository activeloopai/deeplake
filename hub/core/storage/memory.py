from hub.core.storage.provider_mapper import ProviderMapper
import fsspec


class MemoryProvider(ProviderMapper):
    """
    Provider class for using the memory.
    """

    def __init__(self, root):
        """
        Initializes the MemoryProvider.

        Example:
            memory_provider = MemoryProvider("abcd/def")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.

        Returns:
            None

        Raises:
            None
        """
        self.mapper = fsspec.filesystem("memory").get_mapper(
            root=root, check=False, create=False
        )
