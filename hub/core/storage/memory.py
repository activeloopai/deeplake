from hub.core.storage.provider import StorageProvider


class MemoryProvider(StorageProvider):
    """Provider class for using the memory."""

    def __init__(self, root):
        self.mapper = {}

    def __getitem__(
        self,
        path: str,
    ):
        """Gets the object present at the path within the given byte range.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            my_data = local_provider["abc.txt"]

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """
        return self.mapper[path]

    def __setitem__(
        self,
        path: str,
        value: bytes,
    ):
        """Sets the object present at the path with the value

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            local_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.
        """
        self.mapper[path] = value

    def __iter__(self):
        """Generator function that iterates over the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            for my_data in local_provider:
                pass

        Yields:
            bytes: the bytes of the object that it is iterating over.
        """
        yield from self.mapper.items()

    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            del local_provider["abc.txt"]

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
        """
        del self.mapper[path]

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            len(local_provider)

        Returns:
            int: the number of files present inside the root
        """
        return len(self.mapper)
