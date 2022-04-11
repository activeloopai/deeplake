from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.storage.s3 import S3Provider


class LinkCreds(HubMemoryObject):
    def __init__(self):
        self.creds_keys = []
        self.creds_dict = {}  # keys to actual creds dictionary
        self.creds_mapping = {}  # keys to numbers, for encoding
        self.storage_providers = None
        self.is_dirty = False
        self.default_s3_provider = None
        self.default_gcs_provider = None

    def get_default_provider(self, provider_type):
        if provider_type == "s3":
            if self.default_s3_provider is None:
                self.default_s3_provider = S3Provider("s3://bucket/path")
            return self.default_s3_provider
        if provider_type == "gcs":
            if self.default_gcs_provider is None:
                from hub.core.storage.gcs import GCSProvider

                self.default_gcs_provider = GCSProvider("gcs://bucket/path")
            return self.default_gcs_provider
        raise ValueError(f"Provider type {provider_type} not supported")

    def get_storage_provider(self, key: str, provider_type):
        if key in {"ENV", None}:
            return self.get_default_provider(provider_type)
        if key not in self.creds_keys:
            raise ValueError(f"Creds key {key} does not exist")
        if key not in self.creds_dict:
            raise ValueError(
                f"Creds key {key} hasn't been populated. Populate it using ds.populate_creds()"
            )

        if key in self.storage_providers:
            return self.storage_providers[key]
        if provider_type == "s3":
            creds = self.creds_dict[key]
            provider = S3Provider("s3://bucket/path", **creds)
            self.storage_providers[key] = provider
            return provider
        if provider_type == "gcs":
            from hub.core.storage.gcs import GCSProvider

            creds = self.creds_dict[key]
            provider = GCSProvider("gcs://bucket/path", **creds)
            self.storage_providers[key] = provider
            return provider
        raise ValueError("Unexpected provider type")

    def add_creds(self, creds_key: str):
        if creds_key in self.creds_keys:
            raise ValueError(f"Creds key {creds_key} already exists")
        self.creds_keys.append(creds_key)
        self.creds_mapping[creds_key] = len(self.creds_keys)
        self.is_dirty = True

    def populate_creds(self, creds_key: str, creds):
        if creds_key not in self.creds_keys:
            raise ValueError(f"Creds key {creds_key} does not exist")
        self.creds_dict[creds_key] = creds

    def tobytes(self) -> bytes:
        return bytes(",".join(self.chunks), "utf-8")

    def get_encoding(self, key):
        if key in {None, "ENV"}:
            return 0
        if key not in self.creds_keys:
            raise ValueError(f"Creds key {key} does not exist")
        return self.creds_mapping[key]

    def get_creds_key(self, encoding):
        return None if encoding == 0 else self.creds_keys[encoding]

    @staticmethod
    def frombuffer(cls, buffer: bytes):
        obj = cls()
        if buffer:
            obj.creds_keys = list(buffer.decode("utf-8").split(","))
            obj.creds_mapping = {k: i + 1 for i, k in enumerate(obj.creds_keys)}
        obj.is_dirty = False
        return obj

    def nbytes(self):
        return 8 + ((len(self.creds_keys) - 1) * 9) if self.creds_keys else 0

    def __getstate__(self):
        return {
            "creds_keys": self.creds_keys,
            "creds_dict": self.creds_dict,
        }

    def __setstate__(self, state):
        self.creds_keys = state["creds_keys"]
        self.creds_dict = state["creds_dict"]
        self.creds_mapping = {key: i + 1 for i, key in enumerate(self.creds_keys)}
        self.is_dirty = False
        self.storage_providers = None
        self.default_s3_provider = None
        self.default_gcs_provider = None
