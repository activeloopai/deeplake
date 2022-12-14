import numpy as np

from deeplake.util.exceptions import DynamicTensorNumpyError  # type: ignore


class PlyParserBase:
    def __init__(
        self, ply_arr, sample_info, tensor_name=None, index=None, aslist=False
    ):
        self.ply_arr = ply_arr
        self.sample_info = sample_info
        self.tensor_name = tensor_name
        self.index = index
        self.aslist = aslist

    def numpy(self):
        if self.type is list:
            return self._get_ply_arr_as_list()
        return self._get_ply_arr_as_numpy()

    def data(self):
        raise NotImplementedError

    def _get_ply_arr_as_list(self):
        raise NotImplementedError

    def _get_ply_arr_as_numpy(self):
        raise NotImplementedError

    def _cast_mesh_to_the_biggest_dtype(self, array, dtypes):
        casted_array = []
        dtypes = [np.dtype(dtype) for dtype in dtypes]
        max_dtype = np.max(dtypes)

        column_names = array.dtype.names
        for column in column_names:
            casted_column = array[column].astype(max_dtype)
            casted_array.append(casted_column.reshape(-1, 1))
        casted_array = np.concatenate(casted_array, axis=1)
        return casted_array

    @property
    def type(self):
        if isinstance(self.ply_arr, list):
            return list
        return np.array
