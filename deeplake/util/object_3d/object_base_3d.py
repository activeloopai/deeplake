from abc import ABC, abstractmethod


class ObjectBase3D(ABC):
    """Represents a base class for working with 3d data.

    Args:
        path (str): path to the compressed 3d object file
    """

    def __init__(self, path):
        self.path = path
        self.data = self._parse_3d_data(path)
        (
            self.dimensions_names,
            self.dimensions_names_to_dtype,
        ) = self._parse_dimensions_names()
        self.headers = []
        self.meta_data = self._parse_meta_data()

    @abstractmethod
    def _parse_3d_data(self, path):
        raise NotImplementedError("PointCloudBase._parse_3d_data is not implemented")

    @abstractmethod
    def _parse_dimensions_names(self):
        raise NotImplementedError(
            "PointCloudBase._parse_dimensions_names is not implemented"
        )

    @abstractmethod
    def _parse_meta_data(self):
        raise NotImplementedError("PointCloudBase._parse_meta_data is not implemented")

    def __len__(self):
        return len(self.data)
