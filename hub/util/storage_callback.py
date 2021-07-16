from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import CallbackInitializationError
from hub.core.storage.cachable import Cachable


def callback(check_only: bool = False):
    """Decorator for methods that should require `initialize_callback_location` to be called first."""

    # TODO: update docstring

    def outer(func):
        def inner(obj: "CachableCallback", *args, **kwargs):
            if not obj._is_callback_initialized():
                raise CallbackInitializationError(
                    "Must first call `initialize_callback_location` before any other methods may be called."
                )

            y = func(obj, *args, **kwargs)

            if not check_only:
                obj.callback()

            return y

        return inner

    return outer


class CachableCallback(Cachable):
    def __init__(self):
        # TODO: docstring (warn that this may be very slow and shouldn't be used often or should be optimized)

        self._key = None
        self._storage = None

    def _is_callback_initialized(self) -> bool:
        key_ex = self._key is not None
        storage_ex = self._storage is not None
        return key_ex and storage_ex

    def initialize_callback_location(self, key: str, storage: StorageProvider):
        """Must be called once before any other method calls.

        Args:
            key (str): The key for where in `storage` bytes are serialized with each callback call.
            storage (LRUCache): The storage for where bytes are serialized with each callback call.

        Raises:
            CallbackInitializationError: Cannot re-initialize.
        """

        if self._is_callback_initialized():
            raise CallbackInitializationError(
                f"`initialize_callback_location` was already called. key={self._key}"
            )

        self._key = key
        self._storage = storage

    def callback(self):
        # TODO
        raise NotImplementedError
