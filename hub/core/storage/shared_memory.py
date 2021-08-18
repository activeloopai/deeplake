from typing import List
from multiprocessing.shared_memory import SharedMemory
from hub.core.storage.provider import StorageProvider


class SharedMemoryProvider(StorageProvider):
    """Provider class for using shared memory."""

    def __init__(self, root: str = ""):
        self.root = root
        self.sizes = {}

        # keeps the shared memory objects in memory, otherwise getitem throws warnings as the shared memory is deleted
        self.active_shared_memories = {}

    def __getitem__(self, path: str):
        """Gets the object present at the path within the given byte range.

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            my_data = shm_provider["abc.txt"]

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """
        shared_memory = SharedMemory(name=path)
        self.active_shared_memories[path] = shared_memory
        chunk_size = self.sizes[path]
        return shared_memory.buf[:chunk_size]

    def __setitem__(
        self,
        path: str,
        value: bytes,
    ):
        """Sets the object present at the path with the value

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            shm_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        size = len(value)
        self.sizes[path] = size
        try:
            shared_memory = SharedMemory(create=True, size=size, name=path)
        except FileExistsError:
            shared_memory = SharedMemory(name=path)
            shared_memory.unlink()
            shared_memory = SharedMemory(create=True, size=size, name=path)

        # needs to be sliced as some OS (like macOS) allocate extra space
        shared_memory.buf[:size] = value
        shared_memory.close()

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            for my_data in shm_provider:
                pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.sizes

    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            del shm_provider["abc.txt"]

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        try:
            self.check_readonly()
            shared_memory = SharedMemory(name=path)
            shared_memory.close()
            shared_memory.unlink()
            del self.active_shared_memories[path]
            del self.sizes[path]
        except (FileNotFoundError, KeyError):
            pass

    def delete_items(self, paths: List[str]):
        """Deletes the items from the provider.

        Args:
            paths (List[str]): List of paths to be deleted.
        """
        self.check_readonly()
        for path in paths:
            del self[path]

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            len(shm_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.sizes)

    def clear(self):
        """Clears the provider."""
        self.check_readonly()
        paths = list(self.sizes.keys())
        for path in paths:
            del self[path]

    def __getstate__(self) -> str:
        raise NotImplementedError

    def __setstate__(self, state: str):
        raise NotImplementedError

    def update_sizes(self, dict):
        """Updates the sizes of the files.

        Args:
            dict (Dict[str, int]): Dictionary of the form {path: size}.
        """
        self.sizes.update(dict)
