from hub.core.storage.provider import StorageProvider
import os


class LocalProvider(StorageProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, root: str):
        """Initializes the LocalProvider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."

        Raises:
            Exception: If the root is a file instead of a directory.
        """
        if os.path.isfile(root):
            raise Exception  # TODO better exception
        self.root = root

    def __getitem__(self, path: str):
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
        try:
            full_path = self._check_is_file(path)
            file = open(full_path, "rb")
            return file.read()
        except Exception:
            raise KeyError

    def __setitem__(self, path: str, value: bytes):
        """Sets the object present at the path with the value

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            local_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            Exception: If a directory is present at the path.
        """
        full_path = self._check_is_file(path)
        file = open(full_path, "wb")
        file.write(value)

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
        try:
            full_path = self._check_is_file(path)
            os.remove(full_path)
        except Exception:
            raise KeyError

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            for my_data in local_provider:
                pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._list_keys()

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            len(local_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._list_keys())

    def _list_keys(self):
        """Helper function that lists all the objects present at the root of the Provider.

        Returns:
            list: list of all the objects found at the root of the Provider.
        """
        ls = []
        for r, d, f in os.walk(self.root):
            for file in f:
                ls.append(os.path.relpath(os.path.join(r, file), self.root))
        return ls

    def _check_is_file(self, path: str):
        """Helper function to check if the path is a file.
        If the file doesn't exist, the path to it is created.

        Args:
            path (str): the path to the object relative to the root of the provider.

        Returns:
            str: the full path to the requested file.

        Raises:
            Exception: If there is a folder present at the path.
        """
        full_path = os.path.join(self.root, path)
        if os.path.isdir(full_path):
            raise Exception  # TODO better exception
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return full_path
