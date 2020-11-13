import hub
from hub.utils import batch
from hub.compute.transform import Transform

try:
    import pathos
except:
    pass


class PathosTransform(Transform):
    pass