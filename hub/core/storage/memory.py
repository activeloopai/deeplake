from hub.core.storage.mapped_provider import MappedProvider
import fsspec  # type: ignore


class MemoryProvider(MappedProvider):
    """Provider class for using the memory."""

    def __init__(self, root):
        """Initializes the MemoryProvider.

        Example:
            memory_provider = MemoryProvider("abcd/def")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
        """
        self.mapper = fsspec.filesystem("memory").get_mapper(
            root=root, check=False, create=False
        )
