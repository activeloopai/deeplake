from collections.abc import MutableMapping
from typing import Optional
from hub.util.assert_byte_indexes import assert_byte_indexes
from hub.constants import BYTE_PADDING
from hub.core.storage.provider import Provider


class MappedProvider(Provider):
    """A subclass of Provider. This uses a mapper to implement all the methods.

    To add a new provider using MappedProvider:
    - Create a subclass of this class
    - Assign an object of type MutableMap to self.mapper in __init__.
    """

    def __init__(self):
        self.mapper = {}

    def __getitem__(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        """Gets the object present at the path within the given byte range.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            my_data = local_provider["abc.txt"]

        Args:
            path (str): The path relative to the root of the provider.
            start_byte (int, optional): If only specific bytes starting from start_byte are required.
            end_byte (int, optional): If only specific bytes up to end_byte are required.

        Returns:
            bytes: The bytes of the object present at the path within the given byte range.

        Raises:
            InvalidBytesRequestedError: If `start_byte` > `end_byte` or `start_byte` < 0 or `end_byte` < 0.
            KeyError: If an object is not found at the path.
        """
        assert_byte_indexes(start_byte, end_byte)
        return self.mapper[path][start_byte:end_byte]

    def __setitem__(
        self,
        path: str,
        value: bytes,
        start_byte: Optional[int] = None,
        overwrite: Optional[bool] = False,
    ):
        """Sets the object present at the path with the value

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            local_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.
            start_byte (int, optional): If only specific bytes starting from start_byte are to be assigned.
            overwrite (boolean, optional): If the value is True, if there is an object present at the path
                it is completely overwritten, without fetching it's data.

        Raises:
            InvalidBytesRequestedError: If `start_byte` < 0.
        """
        start_byte = start_byte or 0
        end_byte = start_byte + len(value)
        assert_byte_indexes(start_byte, end_byte)
        # file already exists and doesn't need to be overwritten
        if path in self.mapper and not overwrite:
            current_value = bytearray(self.mapper[path])
            # need to pad with zeros at the end to write extra bytes
            if end_byte > len(current_value):
                current_value = current_value.ljust(end_byte, BYTE_PADDING)
            current_value[start_byte:end_byte] = value
            self.mapper[path] = current_value
        # file doesn't exist or needs to be overwritten completely
        else:
            # need to pad with zeros at the start to write from an offset
            if start_byte != 0:
                value = value.rjust(end_byte, BYTE_PADDING)
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
        yield from self.mapper

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

        Args:
            path (str): the path to the object relative to the root of the provider.

        Returns:
            int: the number of files present inside the root
        """
        return len(self.mapper)
