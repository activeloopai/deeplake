from typing import Dict, List

import numpy as np


POINT_CLOUD_FIELD_NAME_TO_TYPESTR: Dict = {
    "X": "<i4",
    "Y": "<i4",
    "Z": "<i4",
    "intensity": "<u2",
    "return_number": "|u1",
    "number_of_returns": "|u1",
    "scan_direction_flag": "|u1",
    "edge_of_flight_line": "|u1",
    "classification": "|u1",
    "synthetic": "|u1",
    "key_point": "|u1",
    "withheld": "|u1",
    "scan_angle_rank": "|i1",
    "user_data": "|u1",
    "point_source_id": "<u2",
    "red": "<u2",
    "green": "<u2",
    "blue": "<u2",
}


def default_version_parser(point_cloud):
    return {
        "major": point_cloud.header.DEFAULT_VERSION.major,
        "minor": point_cloud.header.DEFAULT_VERSION.minor,
    }


def creation_date_parser(point_cloud):
    return {
        "year": point_cloud.header.creation_date.year,
        "month": point_cloud.header.creation_date.month,
        "day": point_cloud.header.creation_date.day,
    }


def global_encoding_parser(point_cloud):
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


LAS_HEADER_FILED_NAME_TO_PARSER = {
    "DEFAULT_VERSION": default_version_parser,
    "version": default_version_parser,
    "creation_date": creation_date_parser,
    "global_encoding": global_encoding_parser,
}


def cast_point_cloud_array_to_proper_dtype(
    full_arr, sample_index, dimension_index, dtype
):
    if isinstance(full_arr, List):
        return full_arr[sample_index][:, dimension_index].astype(np.dtype(dtype))
    return full_arr[sample_index, :, dimension_index].astype(np.dtype(dtype))
