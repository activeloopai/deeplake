from collections import defaultdict

import numpy as np
from deeplake.util.exceptions import DynamicTensorNumpyError  # type: ignore

_valid_formats = {
    "ascii": "",
    "ascii_with_normals": "",
    "binary_big_endian": ">",
    "binary_little_endian": "<",
}


class PlyReaderBase:
    """Represents a parser class for ply data.

    Args:
        filename (str): path to the 3d file
    """

    def __init__(self, filename, fmt):
        self.element_name_to_property_dtypes = defaultdict(dict)
        self.filename = filename
        self.line = []
        self.count = 2
        self.points_size = None
        self.mesh_size = None
        self.has_texture = False
        self.comments = []
        self.stream = self._open_stream(self.filename)
        self.obj_info = None
        self.fmt = fmt
        self.meta_data = self.create_meta_data()
        self.ext = self.read_header()

    @staticmethod
    def _open_stream(filename):
        if isinstance(filename, str):
            return open(filename, "rb")
        return filename

    def _parse_element(self, line):
        line = line.split()
        name = line[1].decode()
        size = int(line[2])
        if name == "vertex":
            self.points_size = size
        elif name == "face":
            self.mesh_size = size

        self.meta_data[name] = size
        return name

    @staticmethod
    def _parse_properties(fmt, ext, line, has_texture, meta_data, name):
        raise NotImplementedError

    @staticmethod
    def _parse_comments(line, meta_data):
        line = line.split(b" ", 1)
        comments = line[1].decode().rstrip()
        meta_data["comments"].append(comments)

    def create_meta_data(self):
        meta_data = {
            "dimensions_names": [],
            "element_name_to_property_dtypes": defaultdict(dict),
            "dimensions_names_to_dtype": {},
            "comments": [],
            "fmt": self.fmt,
            "extension": "ply",
        }
        return meta_data

    def read_header(self):
        ext = _valid_formats[self.fmt]
        self.stream.seek(0)
        while b"end_header" not in self.line and self.line != b"":
            self.line = self.stream.readline()
            if b"element" in self.line:
                name = self._parse_element(self.line)
            elif b"property" in self.line:
                self.has_texture = self._parse_properties(
                    self.fmt, ext, self.line, self.has_texture, self.meta_data, name
                )
            elif b"comment" in self.line:
                self._parse_comments(self.line, self.meta_data)

            self.count += 1
        self.end_header = self.stream.tell()
        return ext

    def read(self):
        stream_bytes = self.stream.read()
        data = self._parse_data(
            self.ext, self.fmt, self.meta_data, stream_bytes, dtype=np.float32
        )
        self.stream.close()
        return data

    def _parse_data(self, ext, fmt, meta_data, stream_bytes, dtype=np.float32):
        raise NotImplementedError

    @staticmethod
    def _convert_dict_to_list_of_tuples(property_name_to_dtypes):
        return [(key, value) for key, value in property_name_to_dtypes.items()]
