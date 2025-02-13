from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS  # type: ignore
from mmengine.registry import Registry  # type: ignore

TRANSFORMS = Registry(
    "transform",
    parent=MMSEG_TRANSFORMS,
    locations=["deeplake.integrations.mmlab.mmseg.mmseg_"],
)
