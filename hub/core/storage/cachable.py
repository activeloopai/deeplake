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
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    # sortkeys / indent make the json more human readable
    def tobytes(self) -> bytes:
        return bytes(
            json.dumps(
                {str(k): v for k, v in self.__getstate__().items()},
                sort_keys=True,
                indent=4,
            ),
            "utf-8",
        )

    def copy(self):
        return self.frombuffer(self.tobytes())

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        instance.__setstate__(json.loads(buffer))
        return instance


def use_callback(check_only: bool = False):
    """Decorator function for `CachableCallback` and it's subclasses.

    Note:
        Must call `@use_callback()` not `@use_callback`.
        Use this decorator on a field method that should use the `CachableCallback.callback` method.
        All methods that are decorated will require that `CachableCallback.initialize_callback_location`
            is called first. Also, after the function executes, `CachableCallback.callback` is called.

    Args:
        check_only (bool): If True, the callback is not actually called. Only the requirement check is executed. Defaults to False.

    Returns:
        Decorator function.
    """

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
        key_exists = self._key is not None
        cache_exists = self._cache is not None
        return key_exists and cache_exists

    def initialize_callback_location(self, key, cache, version_state=None):
        """Must be called once before any other method calls.

        Args:
            key: The key for where in `cache` bytes are serialized with each callback call.
            cache: The cache for where bytes are serialized with each callback call.
            version_state: The version state of the dataset, includes commit_id, commit_node, branch, branch_commit_map and commit_node_map.

        Raises:
            CallbackInitializationError: Cannot re-initialize.
        """

        if self._is_callback_initialized():
            raise CallbackInitializationError(
                f"`initialize_callback_location` was already called. key={self._key}"
            )

        self._key = key
        self._cache = cache
        self._version_state = version_state

    def callback(self):
        self._cache[self._key] = self

    @use_callback(check_only=True)
    def flush(self):
        self._cache.flush()
