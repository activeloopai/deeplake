from hub.util.storage_callback import CachableCallback, callback


_VALUE_ERROR_MSG = '`info.update` should be called with a single dictionary or **kwargs values. Example: `info.update({"key1": 1}, key2=2, key3=3)`'


class Info(CachableCallback):
    def __init__(self):
        """Contains **optional** key/values that datasets/tensors use for human-readability.
        See the `Meta` class for required key/values for datasets/tensors.

        Note:
            Since `Info` is rarely written to and mostly by the user, every modifier will call `storage[key] = self`.
            This is so the user doesn't have to call `flush`.
            Must call `initialize_callback_location` before using any methods.
        """

        self._dict = {}
        super().__init__()

    @property
    @callback(check_only=True)
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    @callback(check_only=True)
    def __len__(self):
        return len(self._dict)

    @callback(check_only=True)
    def as_dict(self) -> dict:
        # TODO: optimize this
        return self._dict.copy()

    @callback()
    def update(self, *args, **kwargs):
        # TODO: convert everything to immutable structures

        if len(args) > 1:
            raise ValueError(_VALUE_ERROR_MSG)  # TODO: exceptions.py

        if len(args) == 1:
            if not isinstance(args[0], dict):
                raise ValueError(_VALUE_ERROR_MSG)

        self._dict.update(*args, **kwargs)
