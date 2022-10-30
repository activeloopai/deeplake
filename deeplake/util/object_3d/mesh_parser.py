from deeplake.util.object_3d import ply_parser


MESH_EXTENSION_TO_MESH_READER = {"ply": ply_parser.PlyParser}


def get_mesh_parser(ext):
    return MESH_EXTENSION_TO_MESH_READER[ext]
