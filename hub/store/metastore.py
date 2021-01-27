from hub.api.versioning import VersionNode
import json
from collections.abc import MutableMapping
import posixpath


class MetaStorage(MutableMapping):
    @classmethod
    def to_str(cls, obj):
        if isinstance(obj, memoryview):
            obj = obj.tobytes()
        if isinstance(obj, bytes):
            obj = obj.decode("utf-8")
        return obj

    def __init__(self, path, fs_map: MutableMapping, meta_map: MutableMapping, version_node: VersionNode):
        self._fs_map = fs_map
        self._meta = meta_map
        self._path = path
        self._version_node = version_node

    def find_node(self, k : str) -> str:
        ls = [path for path in self._fs_map.keys() if path.startswith(k)]
        cur_node = self._version_node
        while cur_node is not None:
            path = f"{k}-{cur_node.commit_id}"
            if path in ls:
                return path
            cur_node = cur_node.parent
        return None

    def __getitem__(self, k: str, check=True) -> bytes:
        filename = posixpath.split(k)[1]
        if not filename.startswith(".") and check:
            filename = self.find_node(filename) or f"{filename}-{self._version_node.commit_id}"
        if filename.startswith("."):
            return bytes(
                json.dumps(
                    json.loads(self.to_str(self._meta["meta.json"]))[k][self._path]
                ),
                "utf-8",
            )
        else:
            return self._fs_map[filename]

    def get(self, k: str) -> bytes:
        # print(f"k is in get {k}")
        filename = posixpath.split(k)[1]
        if filename.startswith("."):
            meta_ = self._meta.get("meta.json")
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

    def __setitem__(self, k: str, v: bytes, check=True):
        filename = posixpath.split(k)[1]
        if not filename.startswith(".") and check:
            old_filename = self.find_node(filename)
            filename = f"{filename}-{self._version_node.commit_id}"
            if old_filename:
                data = self.__getitem__(old_filename, False)
                self.__setitem__(filename, data, False)

        if filename.startswith("."):
            meta = json.loads(self.to_str(self._meta["meta.json"]))
            meta[k] = meta.get(k) or {}
            meta[k][self._path] = json.loads(self.to_str(v))
            self._meta["meta.json"] = bytes(json.dumps(meta), "utf-8")
        else:
            self._fs_map[filename] = v

    def __len__(self):
        return len(self._fs_map) + 1

    def __iter__(self):
        yield ".zarray"
        yield from self._fs_map

    def __delitem__(self, k: str):
        print(f"k is in delitem {k}")
        filename = posixpath.split(k)[1]
        if filename.startswith("."):
            meta = json.loads(self.to_str(self._meta["meta.json"]))
            meta[k] = meta.get(k) or dict()
            meta[k][self._path] = None
            self._meta["meta.json"] = bytes(json.dumps(meta), "utf-8")
        else:
            del self._fs_map[k]

    def flush(self):
        self._meta.flush()
        self._fs_map.flush()

    def commit(self):
        """ Deprecated alias to flush()"""
        self.flush()

    def close(self):
        self._meta.close()
        self._fs_map.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
