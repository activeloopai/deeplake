from deeplake.util.object_3d import ply_reader


MESH_EXTENSION_TO_MESH_READER = {"ply": ply_reader.PlyReader}


def get_mesh_reader(ext):
    return MESH_EXTENSION_TO_MESH_READER[ext]
