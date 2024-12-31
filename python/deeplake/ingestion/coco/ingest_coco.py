from typing import Union, Optional, List, Dict
import pathlib
from deeplake.ingestion.coco.exceptions import CocoAnnotationMissingError
import deeplake as dp
import numpy as np
from tqdm import tqdm
import os

COCO_REQUIRED_KEYS = ["instances", "keypoints", "stuff"]
MASKS_NOTE = "All segmentation polygons and RLEs were converted to stacked binary masks"


def convert_pathlib_to_string_if_needed(path: Union[str, pathlib.Path]) -> str:
    if isinstance(path, pathlib.Path):
        path = str(path)
    return path


def verify_coco_annotation_dict(
    annotation_files: Dict[str, Union[str, pathlib.Path]] = {}
):
    if all(key in annotation_files for key in COCO_REQUIRED_KEYS):
        return {
            key: convert_pathlib_to_string_if_needed(value)
            for key, value in annotation_files.items()
        }
    else:
        raise CocoAnnotationMissingError(
            list(COCO_REQUIRED_KEYS - annotation_files.keys())
        )


class COCOStructuredDataset:
    def __init__(
        self,
        dataset: dp.Dataset = None,
        images_directory: Union[str, pathlib.Path] = None,
        annotation_files: Dict[str, Union[str, pathlib.Path]] = {},
    ):
        from pycocotools.coco import COCO

        self.dataset = dataset
        self.images_directory = images_directory
        self.annotation_files = annotation_files

        self.coco = COCO(self.annotation_files["instances"])
        self.coco_kp = COCO(self.annotation_files["keypoints"])
        self.coco_stuff = COCO(self.annotation_files["stuff"])

        self.category_info = self.coco.loadCats(self.coco.getCatIds())
        self.category_info_kp = self.coco_kp.loadCats(self.coco_kp.getCatIds())
        self.category_info_stuff = self.coco_stuff.loadCats(self.coco_stuff.getCatIds())
        self.img_ids = sorted(self.coco.getImgIds())  # Image ids for uploading

        self.cat_names = [category["name"] for category in self.category_info]
        self.super_cat_names = list(
            set([category["supercategory"] for category in self.category_info])
        )
        self.cat_names_kp = [category["name"] for category in self.category_info_kp]
        self.super_cat_names_kp = list(
            set([category["supercategory"] for category in self.category_info_kp])
        )
        self.cat_names_stuff = [
            category["name"] for category in self.category_info_stuff
        ]
        self.super_cat_names_stuff = list(
            set([category["supercategory"] for category in self.category_info_stuff])
        )

    def get_kp_group_data(self, height, width, anns_kp):
        # Iterate through keypoints and parse each
        categories_kp = np.zeros((len(anns_kp)))
        supercats_kp = np.zeros((len(anns_kp)))
        masks_kp = np.zeros((height, width, len(anns_kp)))
        boxes_kp = np.zeros((len(anns_kp), 4))
        keypoints_kp = np.zeros((51, len(anns_kp)))

        for j, ann_kp in enumerate(anns_kp):
            categories_kp[j] = self.cat_names_kp.index(
                [
                    self.category_info_kp[i]["name"]
                    for i in range(len(self.category_info_kp))
                    if self.category_info_kp[i]["id"] == ann_kp["category_id"]
                ][0]
            )
            supercats_kp[j] = self.super_cat_names_kp.index(
                [
                    self.category_info_kp[i]["supercategory"]
                    for i in range(len(self.category_info_kp))
                    if self.category_info_kp[i]["id"] == ann_kp["category_id"]
                ][0]
            )
            mask_kp = self.coco.annToMask(ann_kp)  # Convert annotation to mask
            masks_kp[:, :, j] = mask_kp
            boxes_kp[j, :] = ann_kp["bbox"]
            keypoints_kp[:, j] = np.array(ann_kp["keypoints"])

        return categories_kp, supercats_kp, masks_kp, boxes_kp, keypoints_kp

    def get_stuff_group_data(self, height, width, ann, anns_stuff):
        # Iterate through stuff and parse each
        masks_stuff = np.zeros((height, width, len(anns_stuff)))
        boxes_stuff = np.zeros((len(anns_stuff), 4))
        categories_stuff = np.zeros((len(anns_stuff)))
        areas_stuff = np.zeros((len(anns_stuff)))
        iscrowds_stuff = np.zeros((len(anns_stuff)))
        supercats_stuff = np.zeros((len(anns_stuff)))

        for k, ann_stuff in enumerate(anns_stuff):
            mask_stuff = self.coco.annToMask(ann_stuff)  # Convert annotation to mask
            masks_stuff[:, :, k] = mask_stuff
            boxes_stuff[k, :] = ann["bbox"]

            # Do a brute force search and make no assumptions between order of relationship of category ids
            categories_stuff[k] = self.cat_names_stuff.index(
                [
                    self.category_info_stuff[i]["name"]
                    for i in range(len(self.category_info_stuff))
                    if self.category_info_stuff[i]["id"] == ann_stuff["category_id"]
                ][0]
            )
            supercats_stuff[k] = self.super_cat_names_stuff.index(
                [
                    self.category_info_stuff[i]["supercategory"]
                    for i in range(len(self.category_info_stuff))
                    if self.category_info_stuff[i]["id"] == ann_stuff["category_id"]
                ][0]
            )

            areas_stuff[k] = ann_stuff["area"]
            iscrowds_stuff[k] = ann_stuff["iscrowd"]

            if "segmentation" not in ann_stuff:
                print("----No segmentation found. Exiting.------")
                print("Annotation length: {}".format(len(anns_stuff)))
                print("----image id: {}----".format(img_id))
                print("----Exiting.------")

        return (
            masks_stuff,
            boxes_stuff,
            categories_stuff,
            areas_stuff,
            iscrowds_stuff,
            supercats_stuff,
        )

    def create_structure(self):
        self.dataset.add_column(
            "images", dp.types.Image(dp.types.UInt8(), sample_compression="jpg")
        )
        self.dataset.add_column("masks", dp.types.BinaryMask(sample_compression="lz4"))
        self.dataset.add_column(
            "boxes", dp.types.BoundingBox(dp.types.Float32(), "ltrb", "pixel")
        )
        self.dataset.add_column(
            "categories", dp.types.ClassLabel(dp.types.Array("uint32", 1))
        )
        self.dataset["categories"].metadata["class_names"] = self.cat_names
        self.dataset.add_column(
            "super_categories", dp.types.ClassLabel(dp.types.Array("uint32", 1))
        )
        self.dataset["super_categories"].metadata["class_names"] = self.super_cat_names
        self.dataset.add_column("areas", dp.types.Array("uint32", 1))
        self.dataset.add_column("iscrowds", dp.types.Array("bool", 1))
        self.dataset.add_column("images_meta", dp.types.Dict())

        # Pose
        self.dataset.add_column(
            "pose/categories", dp.types.ClassLabel(dp.types.Array("uint32", 1))
        )
        self.dataset["pose/categories"].metadata["class_names"] = self.cat_names_kp
        self.dataset.add_column(
            "pose/super_categories", dp.types.ClassLabel(dp.types.Array("uint32", 1))
        )
        self.dataset["pose/super_categories"].metadata[
            "class_names"
        ] = self.super_cat_names_kp
        self.dataset.add_column(
            "pose/boxes", dp.types.BoundingBox(dp.types.Float32(), "LTWH", "pixel")
        )
        self.dataset.add_column(
            "pose/keypoints", dp.types.Array("int32", 2)
        )  # htype="keypoints_coco"
        self.dataset.add_column(
            "pose/masks", dp.types.BinaryMask(sample_compression="lz4")
        )

        # Stuff
        self.dataset.add_column(
            "stuff/masks", dp.types.BinaryMask(sample_compression="lz4")
        )
        self.dataset.add_column(
            "stuff/boxes", dp.types.BoundingBox(dp.types.Float32(), "LTWH", "pixel")
        )
        self.dataset.add_column(
            "stuff/categories", dp.types.ClassLabel(dp.types.Array("uint32", 1))
        )
        self.dataset["stuff/categories"].metadata["class_names"] = self.cat_names_stuff
        self.dataset.add_column(
            "stuff/super_categories", dp.types.ClassLabel(dp.types.Array("uint32", 1))
        )
        self.dataset["stuff/super_categories"].metadata[
            "class_names"
        ] = self.super_cat_names_stuff
        self.dataset.add_column("stuff/areas", dp.types.Array("uint32", 1))
        self.dataset.add_column("stuff/iscrowds", dp.types.Array("bool", 1))

        # update metadatas
        self.dataset["categories"].metadata["category_info"] = self.category_info
        self.dataset["categories"].metadata[
            "notes"
        ] = "Numeric labels for categories represent the position of the class in the ds[categories].medatata['class_names'] list, and not the COCO category id."
        self.dataset["super_categories"].metadata["category_info"] = self.category_info
        self.dataset["super_categories"].metadata[
            "notes"
        ] = "Numeric labels for categories represent the position of the class in the ds[super_categories].medatata['class_names'] list, and not the COCO category id."

        self.dataset["masks"].metadata["notes"] = MASKS_NOTE
        self.dataset["pose/masks"].metadata["category_info"] = self.category_info_kp
        self.dataset["pose/masks"].metadata["notes"] = MASKS_NOTE
        self.dataset["pose/keypoints"].metadata["keypoints"] = [
            category["keypoints"] for category in self.category_info_kp
        ][0]
        self.dataset["pose/keypoints"].metadata["connections"] = [
            category["skeleton"] for category in self.category_info_kp
        ][0]

        self.dataset["stuff/masks"].metadata["category_info"] = self.category_info_stuff
        self.dataset["stuff/masks"].metadata["notes"] = MASKS_NOTE

    def ingest_columns(self):
        for ii, img_id in enumerate(tqdm(self.img_ids), start=1):
            ann_ids = self.coco.getAnnIds(img_id)
            ann_ids_kp = self.coco_kp.getAnnIds(img_id)
            ann_ids_stuff = self.coco_stuff.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            anns_kp = self.coco_kp.loadAnns(ann_ids_kp)
            anns_stuff = self.coco_stuff.loadAnns(ann_ids_stuff)

            img_coco = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_directory, img_coco["file_name"])
            with open(img_path, "rb") as file:
                image_bytes = file.read()
            (height, width) = (img_coco["height"], img_coco["width"])
            masks = np.zeros((height, width, len(anns)))
            boxes = np.zeros((len(anns), 4))
            categories = np.zeros((len(anns)))
            areas = np.zeros((len(anns)))
            iscrowds = np.zeros((len(anns)))
            supercats = np.zeros((len(anns)))

            for i, ann in enumerate(anns):
                mask = self.coco.annToMask(ann)
                masks[:, :, i] = mask
                boxes[i, :] = ann["bbox"]

                categories[i] = self.cat_names.index(
                    [
                        self.category_info[i]["name"]
                        for i in range(len(self.category_info))
                        if self.category_info[i]["id"] == ann["category_id"]
                    ][0]
                )
                supercats[i] = self.super_cat_names.index(
                    [
                        self.category_info[i]["supercategory"]
                        for i in range(len(self.category_info))
                        if self.category_info[i]["id"] == ann["category_id"]
                    ][0]
                )

                areas[i] = ann["area"]
                iscrowds[i] = ann["iscrowd"]

                if "segmentation" not in ann:
                    print("----No segmentation found. Exiting.------")
                    print("Annotation length: {}".format(len(anns)))
                    print("----image id: {}----".format(img_id))
                    print("----Exiting.------")

            (categories_kp, supercats_kp, masks_kp, boxes_kp, keypoints_kp) = (
                self.get_kp_group_data(height, width, anns_kp)
            )

            (
                masks_stuff,
                boxes_stuff,
                categories_stuff,
                areas_stuff,
                iscrowds_stuff,
                supercats_stuff,
            ) = self.get_stuff_group_data(height, width, ann, anns_stuff)

            in_dict = {
                "images": [image_bytes],
                "images_meta": [img_coco],
                "masks": [masks.astype("bool")],
                "boxes": [boxes.astype("float32")],
                "categories": [categories.astype("uint32")],
                "super_categories": [supercats.astype("uint32")],
                "areas": [areas.astype("uint32")],
                "iscrowds": [iscrowds.astype("bool")],
                "pose/categories": [categories_kp.astype("uint32")],
                "pose/super_categories": [supercats_kp.astype("uint32")],
                "pose/boxes": [boxes_kp.astype("float32")],
                "pose/masks": [masks_kp.astype("bool")],
                "pose/keypoints": [keypoints_kp.astype("int32")],
                "stuff/masks": [masks_stuff.astype("bool")],
                "stuff/boxes": [boxes_stuff.astype("float32")],
                "stuff/categories": [categories_stuff.astype("uint32")],
                "stuff/super_categories": [supercats_stuff.astype("uint32")],
                "stuff/areas": [areas_stuff.astype("uint32")],
                "stuff/iscrowds": [iscrowds_stuff.astype("bool")],
            }
            self.dataset.append(in_dict)
        self.dataset.commit("Finished ingestion")

    def structure(self):
        self.create_structure()
        self.ingest_columns()


def ingest_coco(
    images_directory: Union[str, pathlib.Path],
    annotation_files: Dict[str, Union[str, pathlib.Path]],
    dest: Union[str, pathlib.Path],
    dest_creds: Optional[Dict[str, str]] = None,
):
    """Ingest images and annotations in COCO format to a Deep Lake Dataset. The source data can be stored locally or in the cloud.

    Args:
        images_directory (str, pathlib.Path): The path to the directory containing images.
        annotation_files Dict(str, Union[str, pathlib.Path]): dictionary from key to path to JSON annotation file in COCO format.
            - the required keys are the following `instances`, `keypoints` and `stuff`
        dest (str, pathlib.Path):
            - The full path to the dataset. Can be:
            - a Deep Lake cloud path of the form ``al://org_id/datasetname``. To write to Deep Lake cloud datasets, ensure that you are authenticated to Deep Lake (pass in a token using the 'token' parameter).
            - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
            - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
            - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
        dest_creds (Optional[Dict[str, str]]): The dictionary containing credentials used to access the destination path of the dataset.

    Returns:
        Dataset: The Dataset created from images and COCO annotations.

    Raises:
        CocoAnnotationMissingError: If one or many annotation key is missing from file.
    """

    dest = convert_pathlib_to_string_if_needed(dest)
    images_directory = convert_pathlib_to_string_if_needed(images_directory)

    annotation_files = verify_coco_annotation_dict(annotation_files)

    dist_ds = dp.create(dest, dict(dest_creds) if dest_creds is not None else {})

    unstructured = COCOStructuredDataset(
        dataset=dist_ds,
        images_directory=images_directory,
        annotation_files=annotation_files,
    )

    unstructured.structure()

    return dist_ds
