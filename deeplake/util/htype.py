from typing import Tuple, Optional
from deeplake.htype import htype as HTYPE, HTYPE_CONFIGURATIONS
from deeplake.util.exceptions import TensorMetaInvalidHtype


def parse_complex_htype(htype: Optional[str]) -> Tuple[bool, bool, Optional[str]]:
    is_sequence = False
    is_link = False

    if not htype:
        return False, False, None

    elif htype.startswith("sequence"):
        is_sequence, is_link, htype = parse_sequence_start(htype)

    elif htype.startswith("link"):
        is_sequence, is_link, htype = parse_link_start(htype)

    if htype and ("[" in htype or "]" in htype):
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))

    if is_link and htype in (None, HTYPE.DEFAULT):
        if is_sequence:
            raise ValueError(
                "Can't create a linked tensor with a generic htype, you need to specify htype, for example sequence[link[image]] or link[sequence[image]]"
            )
        raise ValueError(
            "Can't create a linked tensor with a generic htype, you need to specify htype, for example link[image]"
        )
    return is_sequence, is_link, htype


def parse_sequence_start(htype):
    if htype == "sequence":
        return True, False, None
    if htype[len("sequence")] != "[" or htype[-1] != "]":
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
    htype = htype.split("[", 1)[1][:-1]
    if not htype:
        return True, False, None
    if htype.startswith("link"):
        if htype == "link":
            return True, True, None
        if htype[len("link")] != "[" or htype[-1] != "]":
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        htype = htype.split("[", 1)[1][:-1]
        if not htype:
            return True, True, None
        if "[" in htype or "]" in htype:
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        return True, True, htype
    return True, False, htype


def parse_link_start(htype):
    if htype == "link":
        return False, True, None
    if htype[len("link")] != "[" or htype[-1] != "]":
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
    htype = htype.split("[", 1)[1][:-1]
    if not htype:
        return False, True, None
    if htype.startswith("sequence"):
        if htype == "sequence":
            return True, True, None
        if htype[len("sequence")] != "[" or htype[-1] != "]":
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        htype = htype.split("[", 1)[1][:-1]
        if not htype:
            return True, True, None
        if "[" in htype or "]" in htype:
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        return True, True, htype
    return False, True, htype
