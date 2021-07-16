from hub.util.storage_callback import CachableCallback, callback


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
        raise NotImplementedError

    @callback()
    def update(self, *args, **kwargs):
        raise NotImplementedError
