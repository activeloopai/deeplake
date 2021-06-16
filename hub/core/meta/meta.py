from hub.util.exceptions import (
    MetaAlreadyExistsError,
    MetaDoesNotExistError,
    MetaInvalidKey,
    MetaInvalidRequiredMetaKey,
)
from hub.util.callbacks import (
    convert_from_callback_objects,
    convert_to_callback_objects,
)
import json
from hub.core.storage.provider import StorageProvider

import hub


class Meta:
    _initialized: bool = False

    def __init__(
        self,
        key: str,
        storage: StorageProvider,
        required_meta: dict = None,
        allow_custom_meta=True,
    ):
        """Internal use only. Synchronizes `required_meta` properties with `storage`. When any update method is called on
        properties defined in `required_meta`, `self._write()` is called which pushes these updates to `storage`.

        Important Note!!!:
            If you are trying to use this constructor for `DatasetMeta`, `TensorMeta`, or `IndexMeta`, you should
                instead use their respective static `create(...)` methods as all of these arguments will be auto-populated.

        Args:
            key (str): Key relative to `storage` where this instance will be synchronized to.
            storage (StorageProvider): Destination of this meta.
            required_meta (dict): A dictionary that describes what properties this Meta should keep track of.
                - If `required_meta` is `{"meta_key": []}`, you will be able to access `meta.meta_key`.
                - If you wanted to update `meta_key`, simply do: `meta.meta_key.append(10)` and it will be
                    immediately syncrhonized.
                - If the value of a property is a `dict` or `list`, it will be recursively converted into `CallbackDict` and
                `CallbackList` respectively.
            allow_custom_meta (bool): If `True`, a `custom_meta` property will be added to `required_meta`. This is
                intended for users to populate themselves and should be empty upon initialization.

        Raises:
            MetaAlreadyExistsError: If trying to initialize with `required_meta` when `key` already exists in `storage`.
            MetaDoesNotExistError: If trying to initialize without `required_meta` when `key` does not exist in `storage`.
            MetaInvalidRequiredMetaKey: `version` will be automatically added to `required_meta`.
        """

        self.key = key
        self.storage = storage

        if self.key in self.storage:
            if required_meta is not None:
                raise MetaAlreadyExistsError(self.key, required_meta)
            self._read()

        else:
            if required_meta is None:
                raise MetaDoesNotExistError(self.key)

            if "version" in required_meta:
                raise MetaInvalidRequiredMetaKey("version", self.__class__.__name__)

            required_meta["version"] = hub.__version__

            if "custom_meta" not in required_meta and allow_custom_meta:
                required_meta["custom_meta"] = {}

            self.from_dict(required_meta)
            self._write()

        self._initialized = True

    def to_dict(self):
        d = {}
        for key in self._required_keys:
            value = getattr(self, key)
            d[key] = convert_from_callback_objects(value)
        return d

    def from_dict(self, meta: dict):
        for key, value in meta.items():
            new_value = convert_to_callback_objects(value, self._write)
            setattr(self, key, new_value)
        self._required_keys = list(meta.keys())
        return self

    def _write(self):
        self.storage[self.key] = bytes(json.dumps(self.to_dict()), "utf8")

    def _read(self):
        meta = json.loads(self.storage[self.key])
        return self.from_dict(meta)

    def __setattr__(self, *args):
        super().__setattr__(*args)
        if self._initialized:
            # can only call `_write` for subsequent setattrs
            self._write()

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise MetaInvalidKey(name, self._required_keys)

    def __iter__(self):
        return self.to_dict().__iter__()

    def __str__(self):
        if self._initialized:
            return f"Meta(key={self.key}, data={self.to_dict()})"
        else:
            return f"UninitializedMeta"

    def __repr__(self):
        return str(self)
