from collections import defaultdict
from typing import List

import copy
import itertools
import json
import time

import numpy as np

from deeplake.util.warnings import always_warn
from pycocotools import coco as pycocotools_coco  # type: ignore
from pycocotools import mask as _mask
import pycocotools.mask as maskUtils  # type: ignore

from tqdm import tqdm  # type: ignore


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class _COCO(pycocotools_coco.COCO):
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
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.masks = masks
        self.bboxes = bboxes
        self.labels = labels
        self.imgs_orig = imgs
        self.iscrowds = iscrowds
        self.class_names = class_names
        self.bbox_format = bbox_format

        # load dataset
        self.anns, self.cats, self.imgs = dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        print("loading annotations into memory...")
        self.dataset = deeplake_dataset
        if self.dataset is not None:
            self.createDeeplakeIndex()

    def createDeeplakeIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        absolute_id = 0
        all_categories = self.labels
        all_bboxes = self.bboxes
        all_masks = self.masks
        all_imgs = self.imgs_orig
        all_iscrowds = self.iscrowds

        for row_index, row in tqdm(
            enumerate(self.dataset),
            desc="loading annotations",
            total=len(self.dataset),
        ):
            if all_imgs[row_index].size == 0:
                always_warn(
                    "found empty image, skipping it. Please verify that your dataset is not corrupted."
                )
                continue
            categories = all_categories[row_index]  # make referencig custom
            bboxes = all_bboxes[row_index]
            if all_masks != [] and all_masks is not None:
                masks = all_masks[row_index]
            else:
                masks = None
            if all_iscrowds is not None:
                is_crowds = all_iscrowds[row_index]
            else:
                is_crowds = np.zeros_like(categories)
            img = {
                "id": row_index,
                "height": all_imgs[row_index].shape[0],
                "width": all_imgs[row_index].shape[1],
            }
            imgs[row_index] = img
            for bbox_index, bbox in enumerate(bboxes):
                if self.masks is not None and self.masks != []:
                    if self.masks.htype == "binary_mask":
                        if masks.size == 0:
                            mask = _mask.encode(np.asfortranarray(masks.numpy()))
                        else:
                            mask = _mask.encode(
                                np.asfortranarray(masks[..., bbox_index].numpy())
                            )
                    elif self.masks.htype == "polygon":
                        mask = convert_poly_to_coco_format(masks.numpy()[bbox_index])
                    else:
                        raise Exception(f"{type(self.masks)} is not supported yet.")
                ann = {
                    "image_id": row_index,
                    "id": absolute_id,
                    "category_id": categories[bbox_index],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "segmentation": mask
                    if masks is not None
                    else None,  # optimize here
                    "iscrowd": int(is_crowds[bbox_index]),
                }

                imgToAnns[row_index].append(ann)
                anns[absolute_id] = ann
                absolute_id += 1

        category_names = self.class_names  # TO DO: add super category names
        category_names = [
            {"id": cat_id, "name": name} for cat_id, name in enumerate(category_names)
        ]

        for idx, category_name in enumerate(category_names):
            cats[idx] = category_name

        for ann in anns.values():
            catToImgs[ann["category_id"]].append(ann["image_id"])

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        print("create index done!")

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = list(self.anns.values())
        else:
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = list(self.anns.values())
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )
        if not iscrowd == None:
            ids = [ann["id"] for ann in anns.values() if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def getCatIds(self, catNms: List = [], supNms: List = [], catIds: List = []):
        """Filtering parameters.

        Args:
            catNms (List): get cats for given cat names
            supNms (List): get classes for given supercategory names
            catIds (List): get cats for given cat ids

        Returns:
            ids (List[int]): integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = list(self.cats.values())
        else:
            cats = list(self.cats.values())
            cats = (
                cats
                if len(catNms) == 0
                else [cat for cat in cats if cat["name"] in catNms]
            )
            cats = (
                cats
                if len(supNms) == 0
                else [cat for cat in cats if cat["supercategory"] in supNms]
            )
            cats = (
                cats
                if len(catIds) == 0
                else [cat for cat in cats if cat["id"] in catIds]
            )
        ids = [cat["id"] for cat in cats]
        return ids

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = _COCO()
        res.dataset = {}
        res.dataset["images"] = [img for img in list(self.imgs.values())]

        print("Loading and preparing results...")
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (
            set(annsImgIds) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
                [ann["image_id"] for ann in anns]
            )
            res.dataset["images"] = [
                img for img in res.dataset["images"] if img["id"] in imgIds
            ]
            for id, ann in enumerate(anns):
                ann["id"] = id + 1
        elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
            res.dataset["categories"] = copy.deepcopy(list(self.cats.values()))
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not "segmentation" in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(list(self.cats.values()))
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if not "bbox" in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(list(self.cats.values()))
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res