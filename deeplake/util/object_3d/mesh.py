import os
import re
import sys

from deeplake.util import exceptions
from deeplake.util.exceptions import DynamicTensorNumpyError  # type: ignore
from deeplake.util.path import convert_pathlib_to_string_if_needed
from deeplake.util.object_3d.object_base_3d import ObjectBase3D
from deeplake.util.object_3d import mesh_parser, mesh_reader


sys_byteorder = (">", "<")[sys.byteorder == "little"]

_valid_formats = {
    "ascii": "",
    "binary_big_endian": ">",
    "binary_little_endian": "<",
}


ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "b1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)


class MeshPly(ObjectBase3D):
    """Represents a ObjectBase3D class that supports compressed 3d object with las extension.

    Args:
        path (str): path to the 3d file
    """

    def __init__(self, path):
        super().__init__(path)

    @property
    def ext(self):
        if isinstance(self.path, str):
            return self.read_ext_from_path()
        return self.read_extension_from_bytes()

    def read_ext_from_path(self):
        ext = self.path.split(".")[-1]
        if ext == "ply":
            return ext
        raise exceptions.UnsupportedExtensionError(ext, "mesh")

    def read_extension_from_bytes(self):
        stream = self.path.readline()
        if b"ply" in stream:
            self.path.seek(0)
            return "ply"
        raise exceptions.UnsupportedMeshExtension(stream, "mesh")

    @property
    def reader(self):
        return mesh_reader.get_mesh_reader(self.ext)

    def _parse_3d_data(self, path):
        data = self.reader(path)
        return data

    def _extract_dimension_names_from_heading(self):
        header = convert_pathlib_to_string_if_needed(
            self.data.header
        )  # windows tests could fail
        dimensions = re.findall("property.*", header)
        return dimensions

    def _parse_meta_data(self):
        full_meta_data = self.data.meta_data
        return full_meta_data

    def _parse_dimensions_names(self):
        return (
            self.data.meta_data["dimensions_names"],
            self.data.meta_data["dimensions_names_to_dtype"],
        )

    @property
    def decompressed_3d_data(self):
        return self.data.read()

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype


def parse_mesh_to_dict(full_arr, sample_info):
    # we assume that the format of files that we append is the same
    fmt = sample_info[0]["fmt"]
    ext = sample_info[0]["extension"]
    parser = mesh_parser.get_mesh_parser(ext)(fmt, full_arr, sample_info)
    return parser.data


def get_mesh_vertices(tensor_name, index, ret, sample_info, aslist):
    # we assume that the format of files that we append is the same
    fmt = sample_info[0]["fmt"]
    ext = sample_info[0]["extension"]
    parser = mesh_parser.get_mesh_parser(ext)(
        fmt, ret, sample_info, tensor_name, index, aslist
    )
    return parser.numpy
