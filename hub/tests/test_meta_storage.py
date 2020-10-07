import fsspec
import hub.utils as utils
import os
import json

import fsspec

import hub.utils as utils
from hub.utils import MetaStorage

def test_meta_storage():
    os.makedirs("./data/test/test_meta_storage/internal_tensor", exist_ok=True)
    fs: fsspec.AbstractFileSystem = fsspec.filesystem("file")
    meta_map = fs.get_mapper("./data/test/test_meta_storage")
    meta_map[".hub.dataset"] = bytes(json.dumps(dict()), "utf-8")
    fs_map = fs.get_mapper("./data/test/test_meta_storage/internal_tensor")
    ms = MetaStorage("internal_tensor", fs_map, meta_map)

