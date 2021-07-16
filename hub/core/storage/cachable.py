from abc import ABC
import json
from hub.util.exceptions import CallbackInitializationError


class Cachable(ABC):
    """Cachable objects can be stored in memory cache without being converted into bytes until flushed.

    For example, `Chunk` is a subclass of `Cachable`. This enables us to update chunk state without serializing
    and deserializing until it's finalized and ready to be flushed from the cache.
    """

    def __init__(self, buffer: bytes = None):
        if buffer:
            self.frombuffer(buffer)

    @property
    def nbytes(self):
        # do not implement, each class should do this because it could be very slow if `tobytes` is called
        raise NotImplementedError

    def tobytes(self) -> bytes:
        return bytes(json.dumps(self.as_dict()), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        instance.__dict__.update(json.loads(buffer))
        return instance


class CachableCallback(Cachable):
    def __init__(self):
        # TODO: docstring (warn that this may be very slow and shouldn't be used often or should be optimized)
        # TODO: mention in docstring "use_callback"

        self._key = None
        self._storage = None

    def _is_callback_initialized(self) -> bool:
        key_ex = self._key is not None
        storage_ex = self._storage is not None
        return key_ex and storage_ex

    def initialize_callback_location(self, key, storage):
        """Must be called once before any other method calls.

        Args:
            key: The key for where in `storage` bytes are serialized with each callback call.
            storage: The storage for where bytes are serialized with each callback call.

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
        self._storage[self._key] = self


def use_callback(check_only: bool = False):
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
