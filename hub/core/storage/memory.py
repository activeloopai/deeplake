from hub.core.storage.provider import Provider
import fsspec


class MemoryProvider(Provider):
    def __init__(self):
        """
        Initializes the MemoryProvider

        Example:
            local_provider = MemoryProvider("/home/ubuntu/Documents/")

        Args:
            root (str): the root of the provider

        Returns:
            None

        Raises:
            Exception #TODO Proper
        """
        self.mapper = fsspec.filesystem("memory").get_mapper(
            root="./", check=False, create=False
        )
