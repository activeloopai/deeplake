from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS
from mmengine.registry import Registry

TRANSFORMS = Registry(
    "transform",
    parent=MMSEG_TRANSFORMS,
    locations=["deeplake.integrations.mmlab.mmseg.mmseg_"],
)
