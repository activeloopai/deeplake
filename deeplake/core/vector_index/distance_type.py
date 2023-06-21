from enum import Enum


class DistanceType(Enum):
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    LINF_NORM = "linf_norm"
    COSINE_SIMILARITY = "cosine_similarity"
