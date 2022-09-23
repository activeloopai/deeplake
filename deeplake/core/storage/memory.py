from typing import Any, Dict
from deeplake.core.storage.lru_cache import _get_nbytes
from deeplake.core.storage.provider import StorageProvider


class MemoryProvider(StorageProvider):
    """Provider class for using the memory."""

    def __init__(self, root: str = ""):
        self.dict: Dict[str, Any] = {}
        self.root = root

    def __getitem__(
        self,
        path: str,
    ):
        """Gets the object present at the path within the given byte range.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> my_data = memory_provider["abc.txt"]

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """
        return self.dict[path]

    def __setitem__(
        self,
        path: str,
        value: bytes,
    ):
        """Sets the object present at the path with the value

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> memory_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self.dict[path] = value

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> for my_data in memory_provider:
            ...    pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.dict

    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> del memory_provider["abc.txt"]

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        del self.dict[path]

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:

            >>> memory_provider = MemoryProvider("xyz")
            >>> len(memory_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.dict)

    def _all_keys(self):
        """Lists all the objects present at the root of the Provider.

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        return set(self.dict.keys())

    def clear(self, prefix=""):
        """Clears the provider."""
        self.check_readonly()
        if prefix:
            self.dict = {k: v for k, v in self.dict.items() if not k.startswith(prefix)}
        else:
            self.dict = {}

    def __getstate__(self) -> str:
        """Does NOT save the in memory data in state."""
        return self.root

    def __setstate__(self, state: str):
        self.__init__(root=state)  # type: ignore

    def get_object_size(self, key: str) -> int:
        return _get_nbytes(self[key])
