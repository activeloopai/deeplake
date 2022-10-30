from deeplake.util.exceptions import DynamicTensorNumpyError  # type: ignore
from deeplake.util.object_3d import ply_readers

PLY_FORMAT_TO_PLY_READER = {
    "ascii": ply_readers.PlyASCIIReader,
    "ascii_with_normals": ply_readers.PlyASCIIWithNormalsReader,
    "binary_little_endian": ply_readers.PlyBinReader,
    "binary_big_endian": ply_readers.PlyBinReader,
}


class PlyReader:
    def __init__(self, filename):
        self.filename = filename
        self.stream = self._open_stream(filename)

        format = self.get_format()
        self.ply_reader = self.get_ply_reader(format)(filename, format)

    @staticmethod
    def _open_stream(filename):
        if isinstance(filename, str):
            return open(filename, "rb")
        return filename

    def get_format(self):
        self.stream.seek(0)
        if b"ply" not in self.stream.readline():
            raise ValueError("The file does not start with the word ply")
        format = self.stream.readline().split()[1].decode()

        line = []

        while b"end_header" not in line and line != b"":
            line = self.stream.readline()
            if (b"nx" in line) or (b"ny" in line) or (b"nz" in line):
                format += "_with_normals"
                break
        return format

    @staticmethod
    def get_ply_reader(format):
        return PLY_FORMAT_TO_PLY_READER[format]

    def read(self):
        return self.ply_reader.read()

    @property
    def meta_data(self):
        return self.ply_reader.meta_data

    @property
    def dtype(self):
        return self.ply_reader.dtype

    @property
    def shape(self):
        return self.ply_reader.shape
