import configparser
from typing import Tuple
from collections.abc import MutableMapping
import json

import fsspec
import gcsfs

from hub.store.cache import Cache

from hub.client.hub_control import HubControlClient


def _flatten(list_):
    """
    Helper function to flatten the list
    """
    return [item for sublist in list_ for item in sublist]


def _connect(tag):
    """Connects to the backend and receive credentials"""

    creds = HubControlClient().get_config()
    dataset = HubControlClient().get_dataset_path(tag)

    if dataset and "path" in dataset:
        path = dataset["path"]
    else:
        sub_tags = tag.split("/")
        real_tag = sub_tags[-1]
        if len(sub_tags) > 1 and sub_tags[0] != creds["_id"]:
            username = creds["bucket"].split("/")[-1]
            creds["bucket"] = creds["bucket"].replace(username, sub_tags[0])

        path = f"{creds['bucket']}/{real_tag}"
    return path, creds


def get_fs_and_path(url: str, token=None) -> Tuple[fsspec.AbstractFileSystem, str]:
    if url.startswith("s3://"):
        token = token or dict()
        token = read_aws_creds(token) if isinstance(token, str) else token
        return (
            fsspec.filesystem(
                "s3",
                key=token.get("aws_access_key_id"),
                secret=token.get("aws_secret_access_key"),
                token=token.get("aws_session_token"),
            ),
            url[5:],
        )
    elif url.startswith("gcs://"):
        return gcsfs.GCSFileSystem(token=token), url[6:]
    elif url.startswith("abs://"):
        # TODO: Azure
        raise NotImplementedError()
    elif (
        url.startswith("../")
        or url.startswith("./")
        or url.startswith("/")
        or url.startswith("~/")
    ):
        return fsspec.filesystem("file"), url
    else:
        # TOOD check if url is username/dataset:version
        url, creds = _connect(url)
        return (
            fsspec.filesystem(
                "s3",
                key=creds["access_key"],
                secret=creds["secret_key"],
                token=creds["session_token"],
                client_kwargs={
                    "endpoint_url": creds["endpoint"],
                    "region_name": creds["region"],
                },
            ),
            url,
        )


def read_aws_creds(filepath: str):
    parser = configparser.ConfigParser()
    parser.read(filepath)
    return {section: dict(parser.items(section)) for section in parser.sections()}


def _get_storage_map(fs, path):
    return fs.get_mapper(path, check=False, create=False)


def get_storage_map(fs, path, memcache=2 ** 26):
    store = _get_storage_map(fs, path)
    return Cache(store, memcache)


class MetaStorage(MutableMapping):
    @classmethod
    def to_str(cls, obj):
        if isinstance(obj, memoryview):
            obj = obj.tobytes()
        if isinstance(obj, bytes):
            obj = obj.decode("utf-8")
        return obj

    def __init__(self, path, fs_map: MutableMapping, meta_map: MutableMapping):
        self._fs_map = fs_map
        self._meta = meta_map
        self._path = path

    def __getitem__(self, k: str) -> bytes:
        if k.startswith("."):
            return bytes(
                json.dumps(
                    json.loads(self.to_str(self._meta[".hub.dataset"]))[k][self._path]
                ),
                "utf-8",
            )
        else:
            return self._fs_map[k]

    def get(self, k: str) -> bytes:
        if k.startswith("."):
            meta_ = self._meta.get(".hub.dataset")
            if not meta_:
                return None
            meta = json.loads(self.to_str(meta_))
            metak = meta.get(k)
            if not metak:
                return None
            item = metak.get(self._path)
            return bytes(json.dumps(item), "utf-8") if item else None
        else:
            return self._fs_map.get(k)

    def __setitem__(self, k: str, v: bytes):
        if k.startswith("."):
            meta = json.loads(self.to_str(self._meta[".hub.dataset"]))
            meta[k] = meta.get(k) or {}
            meta[k][self._path] = json.loads(self.to_str(v))
            self._meta[".hub.dataset"] = bytes(json.dumps(meta), "utf-8")
        else:
            self._fs_map[k] = v

    def __len__(self):
        return len(self._fs_map) + 1

    def __iter__(self):
        yield ".zarray"
        yield from self._fs_map

    def __delitem__(self, k: str):
        if k.startswith("."):
            meta = json.loads(self.to_str(self._meta[".hub.dataset"]))
            meta[k] = meta.get(k) or dict()
            meta[k][self._path] = None
            self._meta[".hub.dataset"] = bytes(json.dumps(meta), "utf-8")
        else:
            del self._fs_map[k]

    # def listdir(self):
    #     res = []
    #     for i in self:
    #         res += [i]
    #     return res

    # def rmdir(self):
    #     for i in self.listdir():
    #         del self[i]

    def commit(self):
        self._meta.commit()
        self._fs_map.commit()


if __name__ == "__main__":
    test_meta_storage()