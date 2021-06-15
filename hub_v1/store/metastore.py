"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import json
from collections.abc import MutableMapping
import posixpath
from hub_v1 import defaults


# TODO: Better version control for PB scale data
# Retrieving chunks with version control is fast, but the memory used to store the chunk_commit_map would be huge ~GBs
class MetaStorage(MutableMapping):
    @classmethod
    def to_str(cls, obj):
        if isinstance(obj, memoryview):
            obj = obj.tobytes()
        if isinstance(obj, bytes):
            obj = obj.decode("utf-8")
        return obj

    def __init__(self, path, fs_map: MutableMapping, meta_map: MutableMapping, ds):
        self._fs_map = fs_map
        self._meta = meta_map
        self._path = path
        self._ds = ds

    def find_chunk(self, k: str) -> str:
        ls = self._ds._chunk_commit_map[self._path][k]
        cur_node = self._ds._version_node
        while cur_node is not None:
            if cur_node.commit_id in ls:
                return f"{k}:{cur_node.commit_id}"
            cur_node = cur_node.parent
        return None

    def __getitem__(self, k: str, check=True) -> bytes:
        filename = posixpath.split(k)[1]
        if filename.startswith("."):
            return bytes(
                json.dumps(
                    json.loads(self.to_str(self._meta[defaults.META_FILE]))[k][
                        self._path
                    ]
                ),
                "utf-8",
            )
        if check:
            if self._ds._commit_id:
                k = self.find_chunk(k) or f"{k}:{self._ds._commit_id}"
        return self._fs_map[k]

    def get(self, k: str, check=True) -> bytes:
        filename = posixpath.split(k)[1]
        if filename.startswith("."):
            meta_ = self._meta.get(defaults.META_FILE)
            if not meta_:
                return None
            meta = json.loads(self.to_str(meta_))
            metak = meta.get(k)
            if not metak:
                return None
            item = metak.get(self._path)
            return bytes(json.dumps(item), "utf-8") if item else None
        else:
            if check:
                if self._ds._commit_id:
                    k = self.find_chunk(k) or f"{k}:{self._ds._commit_id}"
            return self._fs_map.get(k)

    def __setitem__(self, k: str, v: bytes, check=True):
        filename = posixpath.split(k)[1]
        if filename.startswith("."):
            meta = json.loads(self.to_str(self._meta[defaults.META_FILE]))
            meta[k] = meta.get(k) or {}
            meta[k][self._path] = json.loads(self.to_str(v))
            self._meta[defaults.META_FILE] = bytes(json.dumps(meta), "utf-8")
        else:
            chunk_key = k.split(":")[0]
            if check and self._ds._commit_id:
                old_filename = self.find_chunk(k)
                k = f"{k}:{self._ds._commit_id}"
                if old_filename and k != old_filename:
                    self.copy_chunk(old_filename, k)
            commit_id = k.split(":")[-1]
            self._ds._chunk_commit_map[self._path][chunk_key].add(commit_id)
            self._fs_map[k] = v

    def copy_all_chunks(self, from_commit_id: str, to_commit_id: str):
        ls = {
            chunk
            for chunk, commit_ids in self._ds._chunk_commit_map[self._path].items()
            if from_commit_id in commit_ids
        }

        for chunk in ls:
            from_path = f"{chunk}:{from_commit_id}"
            to_path = f"{chunk}:{to_commit_id}"
            self.copy_chunk(from_path, to_path)

    def copy_chunk(self, from_chunk: str, to_chunk: str):
        data = self.__getitem__(from_chunk, False)
        self.__setitem__(to_chunk, data, False)

    def __len__(self):
        return len(self._fs_map) + 1

    def __iter__(self):
        yield ".zarray"
        yield from self._fs_map

    def __delitem__(self, k: str):
        filename = posixpath.split(k)[1]
        if not filename.startswith("."):
            filename = self.find_chunk(filename) or f"{filename}:{self._ds._commit_id}"
        if filename.startswith("."):
            meta = json.loads(self.to_str(self._meta[defaults.META_FILE]))
            meta[k] = meta.get(k) or dict()
            meta[k][self._path] = None
            self._meta[defaults.META_FILE] = bytes(json.dumps(meta), "utf-8")
        else:
            chunk_key = k.split(":")[0]
            if self._ds._commit_id:
                k = self.find_chunk(k) or f"{k}:{self._ds._commit_id}"
            commit_id = k.split(":")[-1]
            try:
                self._ds._chunk_commit_map[self._path][chunk_key].remove(commit_id)
            except Exception:
                try:
                    del self._fs_map[k]
                except Exception:
                    pass

    def flush(self):
        self._meta.flush()
        self._fs_map.flush()

    def commit(self):
        """Deprecated alias to flush()"""
        self.flush()

    def close(self):
        self._meta.close()
        self._fs_map.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
