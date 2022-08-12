from typing import Dict


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
