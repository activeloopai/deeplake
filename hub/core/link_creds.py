import json
from typing import Optional
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.storage.provider import StorageProvider
from hub.core.storage.s3 import S3Provider


class LinkCreds(HubMemoryObject):
    def __init__(self):
        self.creds_keys = []
        self.creds_dict = {}  # keys to actual creds dictionary
        self.creds_mapping = {}  # keys to numbers, for encoding
        self.managed_creds_keys = set()  # keys which are managed
        self.used_creds_keys = set()  # keys which are used by one or more samples
        self.storage_providers = {}
        self.default_s3_provider = None
        self.default_gcs_provider = None

    def get_default_provider(self, provider_type):
        if provider_type == "s3":
            if self.default_s3_provider is None:
                self.default_s3_provider = S3Provider("s3://bucket/path")
            return self.default_s3_provider
        else:
            if self.default_gcs_provider is None:
                from hub.core.storage.gcs import GCSProvider

                self.default_gcs_provider = GCSProvider("gcs://bucket/path")
            return self.default_gcs_provider

    def get_storage_provider(self, key: Optional[str], provider_type):
        assert provider_type in {"s3", "gcs"}
        if key in {"ENV", None}:
            return self.get_default_provider(provider_type)
        if key not in self.creds_keys:
            raise KeyError(f"Creds key {key} does not exist")
        if key not in self.creds_dict:
            raise ValueError(
                f"Creds key {key} hasn't been populated. Populate it using ds.populate_creds()"
            )

        provider: StorageProvider
        creds = self.creds_dict[key]

        if provider_type == "s3":
            if key in self.storage_providers:
                provider = self.storage_providers[key]
                if isinstance(provider, S3Provider):
                    return provider

            provider = S3Provider("s3://bucket/path", **creds)
        else:
            from hub.core.storage.gcs import GCSProvider

            if key in self.storage_providers:
                provider = self.storage_providers[key]
                if isinstance(provider, GCSProvider):
                    return provider

            provider = GCSProvider("gcs://bucket/path", **creds)
        self.storage_providers[key] = provider
        return provider

    def add_creds_key(self, creds_key: str, managed: bool = False):
        if creds_key in self.creds_keys:
            raise ValueError(f"Creds key {creds_key} already exists")
        self.creds_keys.append(creds_key)
        self.creds_mapping[creds_key] = len(self.creds_keys)
        if managed:
            self.managed_creds_keys.add(creds_key)

    def replace_creds(self, old_creds_key: str, new_creds_key: str):
        for i in range(len(self.creds_keys)):
            if self.creds_keys[i] == old_creds_key:
                self.creds_keys[i] = new_creds_key

        if old_creds_key in self.creds_dict:
            self.creds_dict[new_creds_key] = self.creds_dict[old_creds_key]
            del self.creds_dict[old_creds_key]

        self.creds_mapping[new_creds_key] = self.creds_mapping[old_creds_key]
        del self.creds_mapping[old_creds_key]

        if old_creds_key in self.managed_creds_keys:
            self.managed_creds_keys.remove(old_creds_key)
            self.managed_creds_keys.add(new_creds_key)

        if old_creds_key in self.used_creds_keys:
            self.used_creds_keys.remove(old_creds_key)
            self.used_creds_keys.add(new_creds_key)

        if old_creds_key in self.storage_providers:
            self.storage_providers[new_creds_key] = self.storage_providers[
                old_creds_key
            ]
            del self.storage_providers[old_creds_key]

    def populate_creds(self, creds_key: str, creds):
        if creds_key not in self.creds_keys:
            raise KeyError(f"Creds key {creds_key} does not exist")
        self.creds_dict[creds_key] = creds

    def add_to_used_creds(self, creds_key: str):
        if creds_key not in self.used_creds_keys:
            self.used_creds_keys.add(creds_key)
            return True
        return False

    def tobytes(self) -> bytes:
        d = {
            "creds_keys": self.creds_keys,
            "managed_creds_keys": list(self.managed_creds_keys),
            "used_creds_keys": list(self.used_creds_keys),
        }
        return json.dumps(d).encode("utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        obj = cls()
        if buffer:
            d = json.loads(buffer.decode("utf-8"))
            obj.creds_keys = list(d["creds_keys"])
            obj.creds_mapping = {k: i + 1 for i, k in enumerate(obj.creds_keys)}
            obj.managed_creds_keys = set(d["managed_creds_keys"])
            obj.used_creds_keys = set(d["used_creds_keys"])
        obj.is_dirty = False
        return obj

    def get_encoding(self, key):
        if key == "ENV":
            return 0
        if key is None:
            if len(self.creds_keys) == 1:
                key = self.creds_keys[0]
            else:
                raise ValueError(
                    "creds_key can be None only when the dataset has exactly 1 creds_key. For 0 or more than 2 creds_keys, None isn't allowed. If you want to use creds from the environment, pass creds_key='ENV'"
                )
        if key not in self.creds_keys:
            raise ValueError(f"Creds key {key} does not exist")
        return self.creds_mapping[key]

    def get_creds_key(self, encoding):
        if encoding > len(self.creds_keys):
            raise KeyError(f"Encoding {encoding} not found.")
        return None if encoding == 0 else self.creds_keys[encoding - 1]

    @property
    def nbytes(self):
        return len(self.tobytes())

    def __getstate__(self):
        return {
            "creds_keys": self.creds_keys,
            "creds_dict": self.creds_dict,
            "managed_creds_keys": self.managed_creds_keys,
            "used_creds_keys": self.used_creds_keys,
        }

    def __setstate__(self, state):
        self.creds_keys = state["creds_keys"]
        self.creds_dict = state["creds_dict"]
        self.managed_creds_keys = state["managed_creds_keys"]
        self.used_creds_keys = state["used_creds_keys"]
        self.creds_mapping = {key: i + 1 for i, key in enumerate(self.creds_keys)}
        self.storage_providers = {}
        self.default_s3_provider = None
        self.default_gcs_provider = None

    def __len__(self):
        return len(self.creds_keys)

    @property
    def missing_keys(self) -> list:
        return [key for key in self.creds_keys if key not in self.creds_dict]
