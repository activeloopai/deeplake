import sys
from typing import Dict, List

import numpy as np

from deeplake.util.object_3d.object_base_3d import ObjectBase3D


sys_byteorder = (">", "<")[sys.byteorder == "little"]


NUMPY_DTYPE_TO_TYPESTR: Dict = {
    np.dtype("float32"): "<f4",
    np.dtype("int32"): "<i4",
    np.dtype("uint32"): "<u4",
    np.dtype("float16"): "<f2",
    np.dtype("int16"): "<u2",
    np.dtype("uint16"): "<u2",
    np.dtype("uint8"): "|u1",
    np.dtype("int8"): "|i1",
}


class PointCloudLas(ObjectBase3D):
    """Represents a ObjectBase3D class that supports compressed 3d object files with las extension.

    Args:
        point_cloud (str): path to the compressed 3d object file
    """

    def __init__(self, point_cloud):
        super().__init__(point_cloud)

    def _parse_3d_data(self, path):
        import laspy as lp  # type: ignore

        return lp.read(path)

    @property
    def shape(self):
        return self.decompressed_3d_data.shape

    @property
    def dtype(self):
        return self.decompressed_3d_data.dtype

    @property
    def decompressed_3d_data(self, dtype=np.float32):
        decompressed_3d_data = np.concatenate(
            [
                np.expand_dims(self.data[dim_name], -1)
                for dim_name in self.meta_data["dimension_names"]
            ],
            axis=1,
        )
        decompressed_3d_data = decompressed_3d_data.astype(dtype)
        return decompressed_3d_data

    def _parse_dimensions_names(self):
        dimensions_names = list(self.data.point_format.dimension_names)
        dimension_name_to_dtype_dict = {}

        for dimension_name in dimensions_names:
            dimension_name_to_dtype_dict[dimension_name] = NUMPY_DTYPE_TO_TYPESTR[
                self.data[dimension_name].dtype
            ]
        return dimensions_names, dimension_name_to_dtype_dict

    def _parse_meta_data(self):
        from laspy import LasData

        meta_data = {"dimension_names": self.dimensions_names}

        if isinstance(self.data, LasData):
            _parse_las_header_to_metadata(meta_data=meta_data, point_cloud=self.data)

        meta_data["dimensions_names_to_dtype"] = self.dimensions_names_to_dtype
        return meta_data


def _default_version_parser(point_cloud):
    return {
        "major": point_cloud.header.DEFAULT_VERSION.major,
        "minor": point_cloud.header.DEFAULT_VERSION.minor,
    }


def _creation_date_parser(point_cloud):
    return {
        "year": point_cloud.header.creation_date.year,
        "month": point_cloud.header.creation_date.month,
        "day": point_cloud.header.creation_date.day,
    }


def _global_encoding_parser(point_cloud):
    return {
        "GPS_TIME_TYPE_MASK": point_cloud.header.global_encoding.GPS_TIME_TYPE_MASK,
        "SYNTHETIC_RETURN_NUMBERS_MASK": point_cloud.header.global_encoding.SYNTHETIC_RETURN_NUMBERS_MASK,
        "WAVEFORM_EXTERNAL_MASK": point_cloud.header.global_encoding.WAVEFORM_EXTERNAL_MASK,
        "WAVEFORM_INTERNAL_MASK": point_cloud.header.global_encoding.WAVEFORM_INTERNAL_MASK,
        "WKT_MASK": point_cloud.header.global_encoding.WKT_MASK,
        "gps_time_type": point_cloud.header.global_encoding.gps_time_type,
        "synthetic_return_numbers": point_cloud.header.global_encoding.synthetic_return_numbers,
        "value": point_cloud.header.global_encoding.value,
        "waveform_data_packets_external": point_cloud.header.global_encoding.waveform_data_packets_external,
        "waveform_data_packets_internal": point_cloud.header.global_encoding.waveform_data_packets_internal,
        "wkt": point_cloud.header.global_encoding.wkt,
    }


_LAS_HEADER_FILED_NAME_TO_PARSER = {
    "DEFAULT_VERSION": _default_version_parser,
    "version": _default_version_parser,
    "creation_date": _creation_date_parser,
    "global_encoding": _global_encoding_parser,
}


def _parse_las_header_to_metadata(meta_data, point_cloud):
    meta_data.update(
        {
            "las_header": {
                "DEFAULT_VERSION": _LAS_HEADER_FILED_NAME_TO_PARSER["DEFAULT_VERSION"](
                    point_cloud
                ),
                "file_source_id": point_cloud.header.file_source_id,
                "system_identifier": point_cloud.header.system_identifier,
                "generating_software": point_cloud.header.generating_software,
                "creation_date": _LAS_HEADER_FILED_NAME_TO_PARSER["creation_date"](
                    point_cloud
                ),
                "point_count": point_cloud.header.point_count,
                "scales": point_cloud.header.scales.tolist(),
                "offsets": point_cloud.header.offsets.tolist(),
                "number_of_points_by_return": point_cloud.header.number_of_points_by_return.tolist(),
                "start_of_waveform_data_packet_record": point_cloud.header.start_of_waveform_data_packet_record,
                "start_of_first_evlr": point_cloud.header.start_of_first_evlr,
                "number_of_evlrs": point_cloud.header.number_of_evlrs,
                "version": _LAS_HEADER_FILED_NAME_TO_PARSER["version"](point_cloud),
                "maxs": point_cloud.header.maxs.tolist(),
                "mins": point_cloud.header.mins.tolist(),
                "major_version": point_cloud.header.major_version,
                "minor_version": point_cloud.header.minor_version,
                "global_encoding": _LAS_HEADER_FILED_NAME_TO_PARSER["global_encoding"](
                    point_cloud
                ),
                "uuid": str(point_cloud.header.uuid),
            },
            "vlrs": point_cloud.vlrs,
        }
    )


def cast_point_cloud_array_to_proper_dtype(
    full_arr, sample_index, dimension_index, dtype
):
    if isinstance(full_arr, List):
        return full_arr[sample_index][:, dimension_index].astype(np.dtype(dtype))
    return full_arr[sample_index, :, dimension_index].astype(np.dtype(dtype))


def parse_point_cloud_to_dict(full_arr, ndim, sample_info):
    if ndim == 2:
        value_dict = {}  # type: ignore

        if len(sample_info) == 0:
            return value_dict

        dimension_index = 0
        for dimension_name, dtype in sample_info["dimensions_names_to_dtype"].items():  # type: ignore
            value_dict[dimension_name] = full_arr[..., dimension_index].astype(np.dtype(dtype))  # type: ignore
            dimension_index += 1
        return value_dict

    value = []  # type: ignore
    for sample_index in range(len(full_arr)):
        value_dict = {}  # type: ignore

        if len(sample_info[sample_index]) == 0:
            value.append(value_dict)  # type: ignore
            continue

        dimension_index = 0
        for dimension_name, dtype in sample_info[sample_index]["dimensions_names_to_dtype"].items():  # type: ignore
            value_dict[dimension_name] = cast_point_cloud_array_to_proper_dtype(
                full_arr, sample_index, dimension_index, dtype
            )
            dimension_index += 1
        value.append(value_dict)  # type: ignore

    if len(value) == 1:
        value = value[0]  # type: ignore
    return value
