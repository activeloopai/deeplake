import time
import warnings
import pycocotools  # type: ignore
import numpy as np
import copy
import itertools
import pycocotools.mask as maskUtils  # type: ignore
from pycocotools import coco as pycocotools_coco  # type: ignore
from pycocotools import mask as _mask
from collections import defaultdict
import sys
from typing import Union, Dict, List

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve  # type: ignore
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve
from mmdet.datasets import coco as mmdet_coco  # type: ignore
from mmdet.datasets import pipelines
from deeplake.util.warnings import always_warn
import json
import mmcv  # type: ignore
import math
from tqdm import tqdm


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


class COCODatasetEvaluater(mmdet_coco.CocoDataset):
    def __init__(
        self,
        pipeline,
        deeplake_dataset=None,
        classes=None,
        img_prefix="",
        seg_prefix=None,
        seg_suffix=".png",
        proposal_file=None,
        test_mode=True,
        filter_empty_gt=True,
        file_client_args=dict(backend="disk"),
        imgs=None,
        masks=None,
        bboxes=None,
        labels=None,
        iscrowds=None,
        bbox_format=None,
        batch_size=1,
        num_gpus=1,
    ):
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.seg_suffix = seg_suffix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = classes
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self.data_infos = self.load_annotations(
            deeplake_dataset,
            imgs=imgs,
            labels=labels,
            masks=masks,
            bboxes=bboxes,
            iscrowds=iscrowds,
            class_names=self.CLASSES,
            bbox_format=bbox_format,
        )
        self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline

    def pipeline(self, x):
        return x

    def __len__(self):
        length = super().__len__()
        per_gpu_length = math.floor(length / (self.batch_size * self.num_gpus))
        total_length = per_gpu_length * self.num_gpus
        return total_length

    def load_annotations(
        self,
        deeplake_dataset,
        imgs=None,
        labels=None,
        masks=None,
        bboxes=None,
        iscrowds=None,
        class_names=None,
        bbox_format=None,
    ):
        """Load annotation from COCO style annotation file.

        Args:
            deeplake_dataset (dp.Dataset): Deeplake dataset object.
            imgs (dp.Tensor): image deeplake tensor.
            labels (List[numpy]): List of labels for every every detection for each image in numpy format.
            masks (List[numpy]): List of masks for every every detection for each image in numpy format.
            bboxes (List[numpy]): List of bboxes for every every detection for each image in numpy.
            iscrowds (List[numpy]): List of iscrowds for every every detection for each image in numpy format.
            class_names (List[str]): List of class names for every every detection for each image.
            bbox_format (Dict[Dict[str, str]]): Dictionary contatining bbox format information.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = DeeplakeCOCO(
            deeplake_dataset,
            imgs=imgs,
            labels=labels,
            bboxes=bboxes,
            masks=masks,
            iscrowds=iscrowds,
            class_names=class_names,
            bbox_format=bbox_format,
        )
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids)
        return data_infos


def convert_poly_to_coco_format(masks):
    if isinstance(masks, np.ndarray):
        px = masks[..., 0]
        py = masks[..., 1]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [[float(p) for x in poly for p in x]]
        return poly
    poly = []
    for mask in masks:
        poly_i = convert_poly_to_coco_format(mask)
        poly.append([np.array(poly_i[0])])
    return poly


def check_unsupported_functionalities(cfg):
    check_unused_dataset_fields(cfg)
    check_unsupported_train_pipeline_fields(cfg, mode="train")
    check_unsupported_train_pipeline_fields(cfg, mode="val")
    check_dataset_augmentation_formats(cfg)


def check_unused_dataset_fields(cfg):
    if cfg.get("dataset_type"):
        always_warn(
            "The deeplake mmdet integration does not use dataset_type to work with the data and compute metrics. All deeplake datasets are in the same deeplake format. To specify a metrics format, you should deeplake_metrics_format "
        )

    if cfg.get("data_root"):
        always_warn(
            "The deeplake mmdet integration does not use data_root, this input will be ignored"
        )


def check_unsupported_train_pipeline_fields(cfg, mode="train"):
    transforms = cfg.data[mode].pipeline

    for transform in transforms:
        transform_type = transform.get("type")

        if transform_type == "LoadImageFromFile":
            always_warn(
                "LoadImageFromFile is going to be skipped because deeplake mmdet integration does not use it"
            )

        if transform_type == "LoadAnnotations":
            always_warn(
                "LoadAnnotations is going to be skipped because deeplake mmdet integration does not use it"
            )

        if transform_type == "Corrupt":
            raise Exception("Corrupt augmentation is not supported yet.")

        elif transform_type == "CopyPaste":  # TO DO: @adolkhan resolve this
            raise Exception("CopyPaste augmentation is not supported yet")

        elif transform_type == "CutOut":  # TO DO: @adolkhan resolve this
            raise Exception("CutOut augmentation is not supported yet")

        elif transform_type == "Mosaic":  # TO DO: @adolkhan resolve this
            raise Exception("Mosaic augmentation is not supported yet")


def check_dataset_augmentation_formats(cfg):
    if cfg.get("train_dataset"):
        always_warn(
            "train_dataset is going to be unused. Datset types like: ConcatDataset, RepeatDataset, ClassBalancedDataset, MultiImageMixDataset are not supported."
        )
