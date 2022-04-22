# type: ignore

from typing import Tuple, Optional
from hub.htype import DEFAULT_HTYPE, HTYPE_CONFIGURATIONS
from hub.util.exceptions import TensorMetaInvalidHtype


def parse_complex_htype(htype: Optional[str]) -> Tuple[bool, bool, str]:
    is_sequence = False
    is_link = False

    if not htype:
        htype = DEFAULT_HTYPE

    elif htype.startswith("sequence"):
        is_sequence, is_link, htype = parse_sequence_start(htype)

    elif htype.startswith("link"):
        is_sequence, is_link, htype = parse_link_start(htype)

    if "[" in htype or "]" in htype:
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))

    return is_sequence, is_link, htype


def parse_sequence_start(htype):
    if htype == "sequence":
        return True, False, DEFAULT_HTYPE
    if htype[len("sequence")] != "[" or htype[-1] != "]":
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
    htype = htype.split("[", 1)[1][:-1]
    if not htype:
        return True, False, DEFAULT_HTYPE
    if htype.startswith("link"):
        if htype == "link":
            return True, True, DEFAULT_HTYPE
        if htype[len("link")] != "[" or htype[-1] != "]":
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        htype = htype.split("[", 1)[1][:-1]
        if not htype:
            return True, True, DEFAULT_HTYPE
        if "[" in htype or "]" in htype:
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        return True, True, htype
    return True, False, htype


def parse_link_start(htype):
    if htype == "link":
        return False, True, DEFAULT_HTYPE
    if htype[len("link")] != "[" or htype[-1] != "]":
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
    htype = htype.split("[", 1)[1][:-1]
    if not htype:
        return False, True, DEFAULT_HTYPE
    if htype.startswith("sequence"):
        if htype == "sequence":
            return True, True, DEFAULT_HTYPE
        if htype[len("sequence")] != "[" or htype[-1] != "]":
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        htype = htype.split("[", 1)[1][:-1]
        if not htype:
            return True, True, DEFAULT_HTYPE
        if "[" in htype or "]" in htype:
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        return True, True, htype
    return False, True, htype
