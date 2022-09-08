# type: ignore
from typing import Tuple, Optional
from hub.htype import htype as HTYPE, HTYPE_CONFIGURATIONS, UNSPECIFIED
from hub.util.exceptions import TensorMetaInvalidHtype


def _raise(htype):
    raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS))


def parse_complex_htype(htype: Optional[str]) -> Tuple[bool, bool, bool, str]:
    if not htype or not htype.replace(" ", ""):
        htype = HTYPE.DEFAULT
    wrappers = {"sequence": False, "link": False, "tag": False}
    orig_htype = htype
    while htype[-1] == "]":
        sp = htype.split("[", 1)
        if len(sp) == 1:
            _raise(orig_htype)
        wrapper = sp[0]
        if wrapper not in wrappers:
            _raise(orig_htype)
        wrappers[wrapper] = True
        htype = sp[1][:-1]
    if htype in wrappers:
        wrappers[htype] = True
        htype = HTYPE.DEFAULT
    if wrappers["link"] and htype == HTYPE.DEFAULT:
        if wrappers["sequence"]:
            raise ValueError(
                "Can't create a linked tensor with a generic htype, you need to specify htype, for example sequence[link[image]] or link[sequence[image]]"
            )
        raise ValueError(
            "Can't create a linked tensor with a generic htype, you need to specify htype, for example link[image]"
        )
    if htype != UNSPECIFIED and htype not in HTYPE_CONFIGURATIONS:
        _raise(orig_htype)
    return (*wrappers.values(), htype)

def remove_wrapper_from_htype(htype:str, wrapper: str):
    if htype == wrapper:
        return HTYPE.DEFAULT
    htype = htype.replace("f{wrapper}[", "")[:-1]
    if not htype:
        return HTYPE.DEFAULT
    return htype
