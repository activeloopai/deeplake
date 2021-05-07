from collections.abc import MutableMapping
from typing import Optional
from abc import ABC, abstractmethod


class Provider(ABC, MutableMapping):
    """
    An abstract base class for implementing a provider.
    To add a new provider using Provider, create a subclass and implement all 5 abstract methods below.
    Alternatively, you can inherit from ProviderMapper and have a simpler implementation.
    """

    @abstractmethod
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
            path (str): The path relative to the root of the provider.
            start_byte (int, optional): If only specific bytes starting from start_byte are required.
            end_byte (int, optional): If only specific bytes upto end_byte are required.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            InvalidBytesRequestedError: If `start_byte` > `end_byte` or `start_byte` < 0 or `end_byte` < 0.
            KeyError: If an object is not found at the path.
        """
        pass

    @abstractmethod
    def __setitem__(
        self,
        path: str,
        value: bytes,
        start_byte: Optional[int] = None,
        overwrite: Optional[bool] = False,
    ):
        """
        Sets the object present at the path with the value

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            local_provider("abc.txt") = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.
            start_byte (int, optional): If only specific bytes starting from start_byte are to be assigned.
            overwrite (boolean, optional): If the value is True, if there is an object present at the path
                it is completely overwritten, without fetching it's data.

        Returns:
            None

        Raises:
            InvalidBytesRequestedError: If `start_byte` < 0.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """Generator function that iterates over the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            for my_data in local_provider:
                pass
        Args:
            None

        Yields:
            bytes: the bytes of the object that it is iterating over.

        Raises:
            None
        """
        pass

    @abstractmethod
    def __delitem__(self, path: str):
        """
        Delete the object present at the path.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            del local_provider("abc.txt")

        Args:
            path (str): the path to the object relative to the root of the provider.

        Returns:
            None

        Raises:
            KeyError: If an object is not found at the path.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the number of files present inside the root of the provider.

        Example:
            local_provider = LocalProvider("/home/ubuntu/Documents/")
            len(local_provider)

        Args:
            path (str): the path to the object relative to the root of the provider.

        Returns:
            int: the number of files present inside the root

        Raises:
            None
        """
        pass
