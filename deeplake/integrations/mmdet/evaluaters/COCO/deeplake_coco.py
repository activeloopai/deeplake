import warnings
import pycocotools  # type: ignore

from .coco import _COCO


class DeeplakeCOCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(
        self,
        deeplake_dataset=None,
        imgs=None,
        masks=None,
        bboxes=None,
        labels=None,
        iscrowds=None,
        class_names=None,
        bbox_format=("LTRB", "pixel"),
    ):
        if getattr(pycocotools, "__version__", "0") >= "12.0.2":
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning,
            )
        super().__init__(
            deeplake_dataset=deeplake_dataset,
            imgs=imgs,
            masks=masks,
            labels=labels,
            bboxes=bboxes,
            iscrowds=iscrowds,
            class_names=class_names,
            bbox_format=bbox_format,
        )
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)
