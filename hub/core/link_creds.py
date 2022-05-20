from typing import Optional
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.storage.provider import StorageProvider
from hub.core.storage.s3 import S3Provider


class LinkCreds(HubMemoryObject):
    def __init__(self):
        self.creds_keys = []
        self.creds_dict = {}  # keys to actual creds dictionary
        self.creds_mapping = {}  # keys to numbers, for encoding
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

    def add_creds(self, creds_key: str):
        if creds_key in self.creds_keys:
            raise ValueError(f"Creds key {creds_key} already exists")
        self.creds_keys.append(creds_key)
        self.creds_mapping[creds_key] = len(self.creds_keys)

    def populate_creds(self, creds_key: str, creds):
        if creds_key not in self.creds_keys:
            raise KeyError(f"Creds key {creds_key} does not exist")
        self.creds_dict[creds_key] = creds

    def tobytes(self) -> bytes:
        return bytes(",".join(self.creds_keys), "utf-8")

    def get_encoding(self, key):
        if key in {None, "ENV"}:
            return 0
        if key not in self.creds_keys:
            raise ValueError(f"Creds key {key} does not exist")
        return self.creds_mapping[key]

    def get_creds_key(self, encoding):
        if encoding > len(self.creds_keys):
            raise KeyError(f"Encoding {encoding} not found.")
        return None if encoding == 0 else self.creds_keys[encoding - 1]

    @classmethod
    def frombuffer(cls, buffer: bytes):
        obj = cls()
        if buffer:
            obj.creds_keys = list(buffer.decode("utf-8").split(","))
            obj.creds_mapping = {k: i + 1 for i, k in enumerate(obj.creds_keys)}
        obj.is_dirty = False
        return obj

    @property
    def nbytes(self):
        return len(self.tobytes())

    def __getstate__(self):
        return {
            "creds_keys": self.creds_keys,
            "creds_dict": self.creds_dict,
        }

    def __setstate__(self, state):
        self.creds_keys = state["creds_keys"]
        self.creds_dict = state["creds_dict"]
        self.creds_mapping = {key: i + 1 for i, key in enumerate(self.creds_keys)}
        self.storage_providers = {}
        self.default_s3_provider = None
        self.default_gcs_provider = None

    def __len__(self):
        return len(self.creds_keys)

    @property
    def missing_keys(self) -> list:
        return [key for key in self.creds_keys if key not in self.creds_dict]
