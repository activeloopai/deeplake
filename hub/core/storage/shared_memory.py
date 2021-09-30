from typing import List, Optional, Set
from multiprocessing.shared_memory import SharedMemory
from hub.core.storage.provider import StorageProvider


class SharedMemoryProvider(StorageProvider):
    """Provider class for using shared memory."""

    def __init__(self, root: str = ""):
        self.root = root
        self.files: Set[str] = set()
        # keeps the shared memory objects in memory, otherwise getitem throws warnings as the shared memory is deleted
        self.last_active_shared_memory: Optional[SharedMemory] = None

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
        self.last_active_shared_memory = shared_memory
        chunk_size = int.from_bytes(shared_memory.buf[:4], "little")
        return shared_memory.buf[4 : chunk_size + 4]

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
        self.files.add(path)
        try:
            shared_memory = SharedMemory(create=True, size=size + 4, name=path)
        except FileExistsError:
            shared_memory = SharedMemory(name=path)
            shared_memory.unlink()
            shared_memory = SharedMemory(create=True, size=size + 4, name=path)

        # needs to be sliced as some OS (like macOS) allocate extra space
        shared_memory.buf[:4] = size.to_bytes(4, "little")
        shared_memory.buf[4 : size + 4] = value
        shared_memory.close()

    def _all_keys(self):
        """Lists all the objects present at the root of the Provider.

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        return self.files

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            for my_data in shm_provider:
                pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.files

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
            self.files.remove(path)
        except (FileNotFoundError, KeyError):
            pass

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:
            shm_provider = SharedMemoryProvider("xyz")
            len(shm_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.files)

    def clear(self):
        """Clears the provider."""
        self.check_readonly()
        paths = list(self.files)
        for path in paths:
            del self[path]

    def __getstate__(self) -> str:
        raise NotImplementedError

    def __setstate__(self, state: str):
        raise NotImplementedError

    def update_files(self, files: List[str]):
        """Updates the files present in the provider.

        Args:
            files (List[str]): List of files to be updated.
        """
        self.check_readonly()
        self.files.update(files)
