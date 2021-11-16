import os
import shutil
from typing import Optional, Set

from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import DirectoryAtPathException, FileAtPathException


class LocalProvider(StorageProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, root: str):
        """Initializes the LocalProvider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."

        Raises:
            FileAtPathException: If the root is a file instead of a directory.
        """
        if os.path.isfile(root):
            raise FileAtPathException(root)
        self.root = root
        self.files: Optional[Set[str]] = None

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
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
        """
        try:
            full_path = self._check_is_file(path)
            with open(full_path, "rb") as file:
                return file.read()
        except DirectoryAtPathException:
            raise
        except FileNotFoundError:
            raise KeyError(path)

    def __setitem__(self, path: str, value: bytes):
        """Sets the object present at the path with the value

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            local_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            Exception: If unable to set item due to directory at path or permission or space issues.
            FileAtPathException: If the directory to the path is a file instead of a directory.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        full_path = self._check_is_file(path)
        directory = os.path.dirname(full_path)
        if os.path.isfile(directory):
            raise FileAtPathException(directory)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(full_path, "wb") as file:
            file.write(value)
        if self.files is not None:
            self.files.add(path)

    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            del local_provider["abc.txt"]

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        try:
            full_path = self._check_is_file(path)
            os.remove(full_path)
            if self.files is not None:
                self.files.discard(path)
        except DirectoryAtPathException:
            raise
        except FileNotFoundError:
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
        yield from self._all_keys()

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            len(local_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._all_keys())

    def _all_keys(self, refresh: bool = False) -> Set[str]:
        """Lists all the objects present at the root of the Provider.

        Args:
            refresh (bool): refresh keys

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        if self.files is None or refresh:
            full_path = os.path.expanduser(self.root)
            key_set = set()
            for root, dirs, files in os.walk(full_path):
                for file in files:
                    key_set.add(
                        os.path.relpath(os.path.join(full_path, file), full_path)
                    )
            self.files = key_set
        return self.files

    def _check_is_file(self, path: str):
        """Checks if the path is a file. Returns the full_path to file if True.

        Args:
            path (str): the path to the object relative to the root of the provider.

        Returns:
            str: the full path to the requested file.

        Raises:
            DirectoryAtPathException: If a directory is found at the path.
        """
        full_path = os.path.join(self.root, path)
        full_path = os.path.expanduser(full_path)
        if os.path.isdir(full_path):
            raise DirectoryAtPathException
        return full_path

    def clear(self):
        """Deletes ALL data on the local machine (under self.root). Exercise caution!"""
        self.check_readonly()
        self.files = set()
        full_path = os.path.expanduser(self.root)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)

    def __contains__(self, key) -> bool:
        full_path = self._check_is_file(key)
        return os.path.exists(full_path)
