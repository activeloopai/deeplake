from pathlib import Path
from deeplake.util.exceptions import UnsupportedExtensionError
from deeplake.util.object_3d.point_cloud import PointCloudLas
from deeplake.util.object_3d.mesh import MeshPly


_POINT_CLOUD_EXTENSIONS_TO_CLASS = {
    ".las": PointCloudLas,
    ".ply": MeshPly,
}


def read_3d_data(point_cloud_path):
    try:
        import laspy as lp  # type: ignore
        from laspy import LasData
    except ImportError:
        raise ModuleNotFoundError("laspy not found. Install using `pip install laspy`")

    if isinstance(point_cloud_path, str):
        extension = Path(point_cloud_path).suffix

        if extension not in _POINT_CLOUD_EXTENSIONS_TO_CLASS:
            raise UnsupportedExtensionError(extension)
    else:
        point_cloud_bytes = point_cloud_path.read()
        point_cloud_path.seek(0)
        if point_cloud_bytes.startswith(b"ply"):
            extension = ".ply"
        else:
            extension = ".las"

    PointCloud = _POINT_CLOUD_EXTENSIONS_TO_CLASS[extension]
    point_cloud = PointCloud(point_cloud_path)
    return point_cloud
