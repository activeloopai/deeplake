from collections.abc import MutableMapping
from typing import Optional
from hub.core.storage.utils import check_byte_indexes


class Provider(MutableMapping):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        """
        Gets the object present at the path. 
        Optionally, if the start_byte and/or end_byte arguments are specified, it only returns required bytes

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            my_data = local_provider("abc.txt")

        Args:
            path (str): the path relative to the root of the provider
            start_byte (int, optional): If only specific bytes starting from start_byte are required.
            end_byte (int, optional): If only specific bytes upto end_byte are required.

        Returns:
            bytes: The bytes of the object present at the path

        Raises:
            Exception #TODO Proper
        """
        check_byte_indexes(start_byte, end_byte)
        return self.mapper[path][slice(start_byte, end_byte)]

    def __setitem__(
        self,
        path: str,
        value: bytes,
        start_byte: Optional[int] = None,
        overwrite: bool = False,
    ):
        """
        Sets the object present at the path with the value

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            local_provider("abc.txt") = b"abcd"

        Args:
            path (str): the path relative to the root of the provider
            value (bytes): the value to be assigned at the path
            start_byte (int, optional): If only specific bytes starting from start_byte are to be assigned
            overwrite (boolean, optional): If the value is True, if there is an object present at the path
                it is completely overwritten, without fetching it's data.

        Returns:
            None

        Raises:
            Exception #TODO Proper
        """
        start_byte = start_byte or 0
        end_byte = start_byte + len(value)
        check_byte_indexes(start_byte, end_byte)
        # file already exists and doesn't need to be overwritten
        if path in self.mapper and not overwrite:
            current_value = bytearray(self.mapper[path])
            # need to pad with zeros at the end to write extra bytes
            if end_byte > len(current_value):
                current_value = current_value.ljust(end_byte, b"\0")
            current_value[start_byte:end_byte] = value
            self.mapper[path] = current_value
        # file doesn't exist or needs to be overwritten completely
        else:
            start_byte = start_byte or 0
            end_byte = end_byte or len(value)
            check_byte_indexes(start_byte, end_byte)

            # need to pad with zeros at the start to write from an offset
            if start_byte != 0:
                value = value.rjust(end_byte, b"\0")
            self.mapper[path] = value

    def __iter__(self):
        """Generator function that iterates over the provider

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            for my_data in local_provider:
                pass
        Args:
            None
        
        Yields:
            bytes: the bytes of the objects that it is iterating over
        
        Raises:
            Exception #TODO Proper
        """
        yield from self.mapper.items()

    def __delitem__(self, path: str):
        """
        Delete the object present at the path. 

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            del local_provider("abc.txt")

        Args:
            path (str): the path to the object relative to the root of the provider

        Returns:
            None

        Raises:
            Exception #TODO Proper
        """
        del self.mapper[path]

    def __len__(self):
        """
        Returns the number of files present inside the root of the provider

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            len(local_provider)

        Args:
            path (str): the path to the object relative to the root of the provider

        Returns:
            None

        Raises:
            Exception #TODO Proper
        """
        return len(self.mapper)
