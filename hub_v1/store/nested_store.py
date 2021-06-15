"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from collections.abc import MutableMapping

import posixpath


class NestedStore(MutableMapping):
    def __init__(self, storage: MutableMapping, root: str):
        self._storage = storage
        self._root = posixpath.normpath(root)

    def __getitem__(self, k):
        return self._storage[posixpath.join(self._root, k)]

    def __setitem__(self, k, v):
        self._storage[posixpath.join(self._root, k)] = v

    def __delitem__(self, k):
        del self._storage[posixpath.join(self._root, k)]

    def __iter__(self):
        prefix = self._root + "/"
        for item in self._storage:
            item: str
            if item.startswith(prefix):
                yield item[len(prefix) :]

    def __len__(self):
        return sum(1 for _ in self)

    def flush(self):
        self._storage.flush()

    def commit(self):
        """ Deprecated alias to flush()"""
        self.flush()

    def close(self):
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
