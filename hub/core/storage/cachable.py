from abc import ABC
import json
from typing import Any, Dict
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

    def __getstate__(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def tobytes(self) -> bytes:
        return bytes(json.dumps(self.__getstate__()), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        instance.__setstate__(json.loads(buffer))
        return instance


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


class CachableCallback(Cachable):
    def __init__(self):
        """CachableCallback objects can be stored in memory cache and when modifier methods are called, this class is synchronized
        with the cache. This means the user doesn't have to do `ds.cache[cache_key] = ds.info`.

        Note:
            This class should be used as infrequently as possible, as it may lead to slowdowns.
            When extending this class, methods that should have a callback called should be decorated with
                `@use_callback()`.
        """

        self._key = None
        self._cache = None

    def _is_callback_initialized(self) -> bool:
        key_ex = self._key is not None
        cache_ex = self._cache is not None
        return key_ex and cache_ex

    def initialize_callback_location(self, key, cache):
        """Must be called once before any other method calls.

        Args:
            key: The key for where in `cache` bytes are serialized with each callback call.
            cache: The cache for where bytes are serialized with each callback call.

        Raises:
            CallbackInitializationError: Cannot re-initialize.
        """

        if self._is_callback_initialized():
            raise CallbackInitializationError(
                f"`initialize_callback_location` was already called. key={self._key}"
            )

        self._key = key
        self._cache = cache

    def callback(self):
        self._cache[self._key] = self

    @use_callback(check_only=True)
    def flush(self):
        self._cache.flush()
