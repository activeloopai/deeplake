import os
import pathlib
import posixpath
import shutil
from typing import Optional, Set

from deeplake.core.storage.provider import StorageProvider
from deeplake.util.exceptions import (
    DirectoryAtPathException,
    FileAtPathException,
    PathNotEmptyException,
)
from deeplake.util.path import relpath


class LocalProvider(StorageProvider):
    """Provider class for using the local filesystem."""

    def __init__(self, root: str):
        """Initializes the LocalProvider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root."

        Raises:
            FileAtPathException: If the root is a file instead of a directory.
        """
        if os.path.isfile(root):
            raise FileAtPathException(root)
        self.root = root
        self.files: Optional[Set[str]] = None
        self._all_keys()

        self.expiration: Optional[str] = None
        self.db_engine: bool = False
        self.repository: Optional[str] = None

    def subdir(self, path: str, read_only: bool = False):
        sd = self.__class__(os.path.join(self.root, path))
        if self.expiration:
            sd._set_hub_creds_info(
                self.hub_path, self.expiration, self.db_engine, self.repository
            )
        sd.read_only = read_only
        return sd

    def __getitem__(self, path: str):
        """Gets the object present at the path within the given byte range.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> my_data = local_provider["abc.txt"]

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

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> local_provider["abc.txt"] = b"abcd"

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

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> del local_provider["abc.txt"]

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

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> for my_data in local_provider:
            ...    pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._all_keys()

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:

            >>> local_provider = LocalProvider("/home/ubuntu/Documents/")
            >>> len(local_provider)

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
                        posixpath.relpath(
                            posixpath.join(pathlib.Path(root).as_posix(), file),
                            pathlib.Path(full_path).as_posix(),
                        )
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
        full_path = posixpath.join(self.root, path)
        full_path = os.path.expanduser(full_path)
        full_path = str(pathlib.Path(full_path))
        if os.path.isdir(full_path):
            raise DirectoryAtPathException
        return full_path

    def clear(self, prefix=""):
        """Deletes ALL data with keys having given prefix on the local machine (under self.root). Exercise caution!"""
        self.check_readonly()
        full_path = os.path.expanduser(self.root)
        if prefix and self.files:
            self.files = set(file for file in self.files if not file.startswith(prefix))
            full_path = os.path.join(full_path, prefix)
        else:
            self.files = set()
        if os.path.exists(full_path):
            shutil.rmtree(full_path)

    def rename(self, path):
        """Renames root folder"""
        if os.path.isfile(path) or (os.path.isdir(path) and len(os.listdir(path)) > 0):
            raise PathNotEmptyException(use_hub=False)
        os.rename(self.root, path)
        self.root = path

    def __contains__(self, key) -> bool:
        full_path = self._check_is_file(key)
        return os.path.exists(full_path)

    def __getstate__(self):
        return self.root

    def __setstate__(self, state):
        self.__init__(state)

    def get_presigned_url(self, key: str) -> str:
        return os.path.join(self.root, key)

    def get_object_size(self, key: str) -> int:
        return os.stat(os.path.join(self.root, key)).st_size

    def get_bytes(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        try:
            full_path = self._check_is_file(path)
            with open(full_path, "rb") as file:
                if start_byte is not None:
                    file.seek(start_byte)
                if end_byte is None:
                    return file.read()
                else:
                    return file.read(end_byte - (start_byte or 0))
        except DirectoryAtPathException:
            raise
        except FileNotFoundError:
            raise KeyError(path)

    def _set_hub_creds_info(
        self,
        hub_path: str,
        expiration: str,
        db_engine: bool = True,
        repository: Optional[str] = None,
    ):
        """Sets the tag and expiration of the credentials. These are only relevant to datasets using Deep Lake storage.
        This info is used to fetch new credentials when the temporary 12 hour credentials expire.

        Args:
            hub_path (str): The deeplake cloud path to the dataset.
            expiration (str): The time at which the credentials expire.
            db_engine (bool): Whether Activeloop DB Engine enabled.
            repository (str, Optional): Backend repository where the dataset is stored.
        """
        self.hub_path = hub_path
        self.tag = hub_path[6:]  # removing the hub:// part from the path
        self.expiration = expiration
        self.db_engine = db_engine
        self.repository = repository
