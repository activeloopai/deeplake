from typing import Tuple, Optional
from hub.htype import DEFAULT_HTYPE, UNSPECIFIED, HTYPE_CONFIGURATIONS
from hub.util.exceptions import TensorMetaInvalidHtype


def parse_sequence_htype(htype: Optional[str]) -> Tuple[bool, str]:
    if htype in (None, UNSPECIFIED):
        return False, DEFAULT_HTYPE
    if htype.startswith("sequence"):
        if htype == "sequence":
            return True, DEFAULT_HTYPE
        if htype[len("sequence")] != "[" or htype[-1] != "]":
            raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))
        htype = htype.split("[", 1)[1][:-1]
        if not htype:  # sequence[]
            htype = DEFAULT_HTYPE
        return True, htype
    return False, htype
