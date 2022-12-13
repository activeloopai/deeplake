from deeplake.util.object_3d import ply_parsers

FILE_FORMAT_TO_DECODER_CLASS = {
    "binary_little_endian": ply_parsers.PlyBinParser,
    "binary_big_endian": ply_parsers.PlyBinParser,
    "ascii_with_normals": ply_parsers.PlyASCIIWithNormalsParser,
    "ascii": ply_parsers.PlyASCIIParser,
}


class PlyParser:
    def __init__(
        self, fmt, ret, sample_info, tensor_name=None, index=None, aslist=False
    ):
        self.ply_parser = FILE_FORMAT_TO_DECODER_CLASS[fmt](
            ret, sample_info, tensor_name, index, aslist
        )

    @property
    def data(self):
        return self.ply_parser.data()

    @property
    def numpy(self):
        return self.ply_parser.numpy()
