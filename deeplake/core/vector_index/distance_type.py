from enum import Enum


class DistanceType(Enum):
    L1_NORM = "L1_NORM"
    L2_NORM = "L2_NORM"
    LINF_NORM = "LINF_NORM"
    COSINE_SIMILARITY = "COSINE_SIMILARITY"
