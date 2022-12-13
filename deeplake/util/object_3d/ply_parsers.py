import numpy as np

from deeplake.util.exceptions import DynamicTensorNumpyError  # type: ignore
from deeplake.util.object_3d import ply_parser_base


class PlyASCIIWithNormalsParser(ply_parser_base.PlyParserBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_ply_arr_as_numpy(self):
        if self.ply_arr.shape[0] == 1:
            return self.ply_arr[0, :, :3]
        return self.ply_arr[..., :3]

    def _get_ply_arr_as_list(self):
        ply_arr_list = []
        for arr_idx, arr in enumerate(self.ply_arr):
            ply_arr_list.append(arr[..., :3])
        return ply_arr_list

    def data(self):
        if len(self.sample_info) == 0:
            return {}
        value = []
        for sample_index, arrs in enumerate(self.ply_arr):
            value_dict = {}  # type: ignore
            arrs = arrs.transpose(1, 0)
            if len(self.sample_info[sample_index]) == 0:
                value.append(value_dict)
                continue

            dimension_names = list(
                self.sample_info[sample_index]["dimensions_names_to_dtype"].keys()
            )
            for arr_idx, arr in enumerate(arrs):
                dimension_name = dimension_names[arr_idx]
                value_dict[dimension_name] = arr
            value.append(value_dict)

        if len(value) == 1:
            value = value[0]
        return value


class PlyASCIIParser(ply_parser_base.PlyParserBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_ply_arr_as_numpy(self):
        ply_arr_list = self.ply_arr.tolist()

        if len(ply_arr_list) > 1 and not self.aslist:
            raise DynamicTensorNumpyError(self.tensor_name, self.index, "shape")

        ret_list = []
        for arr_idx, arr in enumerate(ply_arr_list):
            vertices = arr[0]
            dtypes = self.sample_info[arr_idx]["element_name_to_property_dtypes"][
                "vertex"
            ].values()
            if len(vertices.dtype.descr) > 1:
                vertices = self._cast_mesh_to_the_biggest_dtype(vertices, dtypes=dtypes)
            ret_list.append(vertices)
        if len(ret_list) == 1 and not self.aslist:
            ret_list = ret_list[0]
        ply_arr_list = ret_list
        return ply_arr_list

    def _get_ply_arr_as_list(self):
        ply_arr_list = []
        for arr_idx, arr in enumerate(self.ply_arr):
            if arr.dtype == np.dtype("O"):
                vertices = arr[0]

                dtypes = self.sample_info[arr_idx]["element_name_to_property_dtypes"][
                    "vertex"
                ].values()
                if len(vertices.dtype.descr) > 1:
                    vertices = self._cast_mesh_to_the_biggest_dtype(
                        vertices, dtypes=dtypes
                    )
                ply_arr_list.append(vertices)
            else:
                ply_arr_list.append(arr[..., :3])
        return ply_arr_list

    def data(self):
        if len(self.sample_info) == 0:
            return {}
        value = []
        for sample_index, arrs in enumerate(self.ply_arr):
            value_dict = {}  # type: ignore

            if len(self.sample_info[sample_index]) == 0:
                value.append(value_dict)  # type: ignore
                continue

            element_names = list(
                self.sample_info[sample_index]["element_name_to_property_dtypes"].keys()
            )
            for arr_idx, arr in enumerate(arrs):
                dimension_names = self.sample_info[sample_index][
                    "element_name_to_property_dtypes"
                ][element_names[arr_idx]]
                for idx, dimension_name in enumerate(dimension_names):
                    value_dict[dimension_name] = arr[:, idx]
            value.append(value_dict)

        if len(value) == 1:
            value = value[0]
        return value


class PlyBinParser(ply_parser_base.PlyParserBase):
    def __init__(self, *args, **kwargs):
        super(PlyBinParser, self).__init__(*args, **kwargs)

    def _get_ply_arr_as_numpy(self):
        ply_arr_list = self.ply_arr.tolist()

        if len(ply_arr_list) > 1 and not self.aslist:
            raise DynamicTensorNumpyError(self.tensor_name, self.index, "shape")

        ret_list = []
        for arr_idx, arr in enumerate(ply_arr_list):
            vertices = arr[0]
            dtypes = self.sample_info[arr_idx]["element_name_to_property_dtypes"][
                "vertex"
            ].values()
            if len(vertices.dtype.descr) > 1:
                vertices = self._cast_mesh_to_the_biggest_dtype(vertices, dtypes=dtypes)
            ret_list.append(vertices)
        if len(ret_list) == 1 and not self.aslist:
            ret_list = ret_list[0]
        return ret_list

    def _get_ply_arr_as_list(self):
        ret_list = []
        for arr_idx, arr in enumerate(self.ply_arr):
            if arr.dtype == np.dtype("O"):
                vertices = arr[0]

                dtypes = self.sample_info[arr_idx]["element_name_to_property_dtypes"][
                    "vertex"
                ].values()
                vertices = self._cast_mesh_to_the_biggest_dtype(vertices, dtypes=dtypes)
                ret_list.append(vertices)
            else:
                ret_list.append(arr[..., :3])
        return ret_list

    def data(self):
        if len(self.sample_info) == 0:
            return {}

        value = []
        for sample_index, arrs in enumerate(self.ply_arr):
            value_dict = {}

            if len(self.sample_info[sample_index]) == 0:
                value.append(value_dict)  # type: ignore
                continue

            for arr_idx, arr in enumerate(arrs):
                for dimension_name, dtype in self.sample_info[sample_index]["dimensions_names_to_dtype"].items():  # type: ignore
                    if arr.dtype.names:
                        if dimension_name in arr.dtype.names:
                            value_dict[dimension_name] = arr[dimension_name]
            value.append(value_dict)

        if len(value) == 1:
            value = value[0]
        return value
