import sys
from io import BytesIO, StringIO

import numpy as np

from deeplake.util.exceptions import DynamicTensorNumpyError  # type: ignore
from deeplake.util.object_3d import ply_reader_base

sys_byteorder = (">", "<")[sys.byteorder == "little"]

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


class PlyASCIIWithNormalsReader(ply_reader_base.PlyReaderBase):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(filename, fmt="ascii_with_normals")

    def _parse_properties(self, fmt, ext, line, has_texture, meta_data, name):
        dimensions_names = []
        line = line.split()
        self.meta_data["dimensions_names_to_dtype"][line[2].decode()] = ply_dtypes[
            line[1]
        ]
        self.meta_data["element_name_to_property_dtypes"][name][
            line[2].decode()
        ] = ply_dtypes[line[1]]

        dimensions_names.append(line[2].decode())

        self.meta_data["dimensions_names"] += dimensions_names
        return has_texture

    def _parse_data(self, ext, fmt, meta_data, stream_bytes, dtype=np.float32):
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            pd = None

        stream_bytes = str(stream_bytes, "utf-8")
        stream_bytes = StringIO(stream_bytes)
        bottom = 0 if self.mesh_size is None else self.mesh_size
        if pd is None:
            raise ModuleNotFoundError(
                "pandas is not installed. Run `pip install pandas`."
            )
        points = pd.read_csv(
            stream_bytes,
            sep=" ",
            header=None,
            engine="python",
            skipfooter=bottom,
            usecols=meta_data["dimensions_names"],
            names=meta_data["dimensions_names"],
        )
        points = points.dropna(axis=1).to_numpy()
        return points

    @property
    def shape(self):
        return (self.points_size, len(self.meta_data["dimensions_names"]))

    @property
    def dtype(self):
        return np.dtype(self.meta_data["dimensions_names_to_dtype"]["x"])


class PlyASCIIReader(ply_reader_base.PlyReaderBase):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(filename, fmt="ascii")

    def get_mesh_names(self, line):
        has_texture = False
        if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
            mesh_names = ["n_points", "v1", "v2", "v3"]
        else:
            has_texture = True
            mesh_names = ["n_coords"] + [
                "v1_u",
                "v1_v",
                "v2_u",
                "v2_v",
                "v3_u",
                "v3_v",
            ]
        return mesh_names, has_texture

    def _read_properties_with_list(self, line, meta_data, name, mesh_names):
        dimensions_names = []

        # the first number has different dtype than the list
        meta_data["dimensions_names_to_dtype"][mesh_names[0]] = ply_dtypes[line[2]]
        meta_data["element_name_to_property_dtypes"][name][mesh_names[0]] = ply_dtypes[
            line[2]
        ]
        # rest of the numbers have the same dtype
        dt = ply_dtypes[line[3]]

        dimensions_names.append(mesh_names[0])
        return dimensions_names, dt

    def _parse_properties(self, fmt, ext, line, has_texture, meta_data, name):
        line = line.split()
        mesh_names, has_texture = self.get_mesh_names(line)
        dimensions_names = []
        if b"list" in line:
            dimensions_names, dt = self._read_properties_with_list(
                line, meta_data, name, mesh_names
            )
            for j in range(1, len(mesh_names)):
                meta_data["dimensions_names_to_dtype"][mesh_names[j]] = dt
                meta_data["element_name_to_property_dtypes"][name][mesh_names[j]] = dt
                dimensions_names.append(mesh_names[j])
        else:
            meta_data["dimensions_names_to_dtype"][line[2].decode()] = ply_dtypes[
                line[1]
            ]
            meta_data["element_name_to_property_dtypes"][name][
                line[2].decode()
            ] = ply_dtypes[line[1]]
        dimensions_names.append(line[2].decode())
        meta_data["dimensions_names"] += dimensions_names
        return has_texture

    def _parse_data(self, ext, fmt, meta_data, stream_bytes, dtype=np.float32):
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            pd = None

        stream_bytes = str(stream_bytes, "utf-8")
        stream_bytes = StringIO(stream_bytes)
        bottom = 0 if self.mesh_size is None else self.mesh_size
        vertex_names = list(
            meta_data["element_name_to_property_dtypes"]["vertex"].keys()
        )
        if pd is None:
            raise ModuleNotFoundError(
                "pandas is not installed. Run `pip install pandas`."
            )
        points = pd.read_csv(
            stream_bytes,
            sep=" ",
            header=None,
            engine="python",
            skipfooter=bottom,
            usecols=meta_data["dimensions_names"],
            names=meta_data["dimensions_names"],
        )

        points = points.dropna(axis=1).to_numpy()

        if self.mesh_size:
            face_names = [
                dim_name
                for dim_name in meta_data["dimensions_names"]
                if dim_name not in vertex_names
            ]
            face_names = np.array(face_names)
            stream_bytes.seek(0)
            mesh = pd.read_csv(
                stream_bytes,
                sep=" ",
                skiprows=self.points_size,
                header=None,
                engine="python",
                usecols=face_names,
                names=face_names,
            )
            return np.asanyarray([points, mesh.to_numpy()], dtype=object)
        return points

    @property
    def shape(self):
        return (self.points_size, len(self.meta_data["dimensions_names"]))

    @property
    def dtype(self):
        if "face" not in self.meta_data["element_name_to_property_dtypes"]:
            return np.dtype(self.meta_data["dimensions_names_to_dtypes"]["x"])
        return np.dtype("O")


class PlyBinReader(ply_reader_base.PlyReaderBase):
    def __init__(self, filename, file_format):
        super().__init__(filename, fmt=file_format)

    def get_mesh_names(self, line):
        has_texture = False
        if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
            mesh_names = ["n_points", "v1", "v2", "v3"]
        else:
            has_texture = True
            mesh_names = ["n_coords"] + [
                "v1_u",
                "v1_v",
                "v2_u",
                "v2_v",
                "v3_u",
                "v3_v",
            ]
        return mesh_names, has_texture

    def _read_properties_with_list(self, ext, line, meta_data, name, mesh_names):
        dimensions_names = []

        # the first number has different dtype than the list
        meta_data["dimensions_names_to_dtype"][mesh_names[0]] = (
            ext + ply_dtypes[line[2]]
        )
        meta_data["element_name_to_property_dtypes"][name][mesh_names[0]] = (
            ext + ply_dtypes[line[2]]
        )
        # rest of the numbers have the same dtype
        dt = ext + ply_dtypes[line[3]]

        dimensions_names.append(mesh_names[0])
        return dimensions_names, dt

    def _parse_properties(self, fmt, ext, line, has_texture, meta_data, name):
        line = line.split()
        mesh_names, has_texture = self.get_mesh_names(line)
        dimensions_names = []
        if b"list" in line:
            dimensions_names, dt = self._read_properties_with_list(
                ext, line, meta_data, name, mesh_names
            )
            for j in range(1, len(mesh_names)):
                meta_data["dimensions_names_to_dtype"][mesh_names[j]] = dt
                meta_data["element_name_to_property_dtypes"][name][mesh_names[j]] = dt
                dimensions_names.append(mesh_names[j])
        else:
            meta_data["dimensions_names_to_dtype"][line[2].decode()] = (
                ext + ply_dtypes[line[1]]
            )
            meta_data["element_name_to_property_dtypes"][name][line[2].decode()] = (
                ext + ply_dtypes[line[1]]
            )
        dimensions_names.append(line[2].decode())
        meta_data["dimensions_names"] += dimensions_names
        return has_texture

    def _parse_data(self, ext, fmt, meta_data, stream_bytes, dtype=np.float32):
        f = self._open_stream(self.filename)
        f.seek(self.end_header)
        points_dtype = self._convert_dict_to_list_of_tuples(
            meta_data["element_name_to_property_dtypes"]["vertex"]
        )
        if isinstance(f, BytesIO):
            points = np.frombuffer(f.read(), dtype=points_dtype, count=self.points_size)
            f.seek(self.end_header + points.nbytes)
        else:
            points = np.fromfile(f, dtype=points_dtype, count=self.points_size)

        if ext != sys_byteorder:
            points = points.byteswap().newbyteorder()

        if self.mesh_size:
            mesh_dtype = self._convert_dict_to_list_of_tuples(
                meta_data["element_name_to_property_dtypes"]["face"]
            )
            if isinstance(f, BytesIO):
                mesh = np.frombuffer(f.read(), dtype=mesh_dtype, count=self.mesh_size)
            else:
                mesh = np.fromfile(f, dtype=mesh_dtype, count=self.mesh_size)

            if ext != sys_byteorder:
                mesh = mesh.byteswap().newbyteorder()
            return np.asanyarray([points, mesh], dtype=object)
        return points

    @property
    def shape(self):
        return (self.points_size, len(self.meta_data["dimensions_names"]))

    @property
    def dtype(self):
        if "face" not in self.meta_data["element_name_to_property_dtypes"]:
            return np.dtype(self.meta_data["dimensions_names_to_dtypes"]["x"])
        return np.dtype("O")
