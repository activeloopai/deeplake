from typing import Union, Optional, List, Dict
import pathlib
from deeplake.ingestion.coco.exceptions import CocoAnnotationMissingError
import deeplake as dp
import numpy as np
import os

try:
    from tqdm import tqdm as progress_bar
except ImportError:

    def progress_bar(iterable, *args, **kwargs):
        return iterable


MASKS_NOTE = "All segmentation polygons and RLEs were converted to stacked binary masks"

DEFAULT_KEY_COLUMN_MAPPING = {
    "masks": "masks",
    "boxes": "boxes",
    "categories": "categories",
    "super_categories": "super_categories",
    "areas": "areas",
    "iscrowds": "iscrowds",
    "keypoints": "keypoints",
}

DEFAULT_FILE_GROUP_MAPPING = {
    "instances": "",  # Root group
    "keypoints": "pose",
    "stuff": "stuff",
}


def standartize_path(path: Union[str, pathlib.Path]) -> str:
    if isinstance(path, pathlib.Path):
        path = str(path)

    if path.startswith("~"):
        path = os.path.expanduser(path)

    return path


def convert_pathlib_to_string_if_needed(path: Union[str, pathlib.Path]) -> str:
    if isinstance(path, pathlib.Path):
        path = str(path)
    return path


def verify_coco_annotation_dict(
    annotation_files: Dict[str, Union[str, pathlib.Path]] = {}
):
    return {
        key: convert_pathlib_to_string_if_needed(value)
        for key, value in annotation_files.items()
    }


class COCOStructuredDataset:
    def __init__(
        self,
        dataset: dp.Dataset = None,
        images_directory: Union[str, pathlib.Path] = None,
        annotation_files: Dict[str, Union[str, pathlib.Path]] = {},
        key_to_column_mapping: Optional[Dict] = None,
        file_to_group_mapping: Optional[Dict] = None,
    ):
        from pycocotools.coco import COCO

        self.dataset = dataset
        self.images_directory = images_directory
        self.annotation_files = annotation_files
        self.key_to_column_mapping = key_to_column_mapping or DEFAULT_KEY_COLUMN_MAPPING
        self.file_to_group_mapping = file_to_group_mapping or DEFAULT_FILE_GROUP_MAPPING

        self.keypoints_group = {}
        self.coco_instances = {}
        self.category_info = {}
        self.cat_names = {}
        self.super_cat_names = {}

        for file_key, file_path in self.annotation_files.items():
            self.coco_instances[file_key] = COCO(standartize_path(file_path))
            coco = self.coco_instances[file_key]

            self.category_info[file_key] = coco.loadCats(coco.getCatIds())
            self.cat_names[file_key] = [
                category["name"] for category in self.category_info[file_key]
            ]
            self.super_cat_names[file_key] = list(
                set(
                    [
                        category["supercategory"]
                        for category in self.category_info[file_key]
                    ]
                )
            )

        # Here the assumption is that all the annotation files have the s+ame image ids
        self.first_key = next(iter(self.annotation_files))
        self.img_ids = sorted(self.coco_instances[self.first_key].getImgIds())

    def has_keypoints(self, group_name: str):
        """Check if the annotations have keypoints"""
        return self.keypoints_group.get(group_name, False)

    def get_group_data(
        self,
        height: int,
        width: int,
        anns: List,
        file_key: str,
        has_keypoints: bool = False,
    ):
        """Generic function to process annotations for any group"""
        n_anns = len(anns)

        masks = np.zeros((height, width, n_anns))
        boxes = np.zeros((n_anns, 4))
        categories = np.zeros((n_anns))
        areas = np.zeros((n_anns))
        iscrowds = np.zeros((n_anns))
        supercats = np.zeros((n_anns))
        keypoints = None

        if has_keypoints:
            keypoints = np.zeros((51, n_anns))

        coco = self.coco_instances[file_key]
        cat_info = self.category_info[file_key]
        cat_names = self.cat_names[file_key]
        supercat_names = self.super_cat_names[file_key]

        for i, ann in enumerate(anns):
            masks[:, :, i] = coco.annToMask(ann)
            boxes[i, :] = ann["bbox"]

            cat_name = next(
                cat_info[j]["name"]
                for j in range(len(cat_info))
                if cat_info[j]["id"] == ann["category_id"]
            )
            supercat_name = next(
                cat_info[j]["supercategory"]
                for j in range(len(cat_info))
                if cat_info[j]["id"] == ann["category_id"]
            )

            categories[i] = cat_names.index(cat_name)
            supercats[i] = supercat_names.index(supercat_name)

            areas[i] = ann.get("area", 0)
            iscrowds[i] = ann.get("iscrowd", 0)

            if keypoints is not None and "keypoints" in ann:
                keypoints[:, i] = np.array(ann["keypoints"])

        result = {
            "masks": masks.astype("bool"),
            "boxes": boxes.astype("float32"),
            "categories": categories.astype("uint32"),
            "super_categories": supercats.astype("uint32"),
            "areas": areas.astype("uint32"),
            "iscrowds": iscrowds.astype("bool"),
        }

        if keypoints is not None:
            result["keypoints"] = keypoints.astype("int32")

        return result

    def create_structure_for_group(self, group_prefix: str, file_key: str):
        """Create dataset structure for a specific annotation group"""
        tensor_prefix = f"{group_prefix}/" if group_prefix else ""

        self.dataset.add_column(
            f"{tensor_prefix}{self.key_to_column_mapping['categories']}",
            dp.types.ClassLabel(dp.types.Array("uint32", 1)),
        )
        self.dataset.add_column(
            f"{tensor_prefix}{self.key_to_column_mapping['super_categories']}",
            dp.types.ClassLabel(dp.types.Array("uint32", 1)),
        )
        self.dataset.add_column(
            f"{tensor_prefix}{self.key_to_column_mapping['boxes']}",
            dp.types.BoundingBox(dp.types.Float32(), "LTWH", "pixel"),
        )
        self.dataset.add_column(
            f"{tensor_prefix}{self.key_to_column_mapping['masks']}",
            dp.types.BinaryMask(sample_compression="lz4"),
        )

        cat_path = f"{tensor_prefix}{self.key_to_column_mapping['categories']}"
        supercat_path = (
            f"{tensor_prefix}{self.key_to_column_mapping['super_categories']}"
        )

        self.dataset[cat_path].metadata["class_names"] = self.cat_names[file_key]
        self.dataset[supercat_path].metadata["class_names"] = self.super_cat_names[
            file_key
        ]

        sample_ann = self.coco_instances[file_key].loadAnns(
            self.coco_instances[file_key].getAnnIds(self.img_ids[0])
        )[0]

        if "area" in sample_ann:
            self.dataset.add_column(
                f"{tensor_prefix}{self.key_to_column_mapping['areas']}",
                dp.types.Array("uint32", 1),
            )

        if "iscrowd" in sample_ann:
            self.dataset.add_column(
                f"{tensor_prefix}{self.key_to_column_mapping['iscrowds']}",
                dp.types.Array("bool", 1),
            )

        if "keypoints" in sample_ann:
            self.keypoints_group[group_prefix] = True
            self.dataset.add_column(
                f"{tensor_prefix}{self.key_to_column_mapping['keypoints']}",
                dp.types.Array("int32", 2),
            )

            if "keypoints" in self.category_info[file_key][0]:
                self.dataset[
                    f"{tensor_prefix}{self.key_to_column_mapping['keypoints']}"
                ].metadata["keypoints"] = [
                    category["keypoints"] for category in self.category_info[file_key]
                ][
                    0
                ]

    def create_structure(self):
        """Create the complete dataset structure"""
        self.dataset.add_column(
            "images", dp.types.Image(dp.types.UInt8(), sample_compression="jpg")
        )
        self.dataset.add_column("images_meta", dp.types.Dict())
        for file_key, group_name in self.file_to_group_mapping.items():
            self.create_structure_for_group(group_name, file_key)

    def ingest_columns(self):
        """Ingest all data into the dataset"""
        for img_id in progress_bar(self.img_ids):
            img_coco = self.coco_instances[self.first_key].loadImgs(img_id)[0]
            img_path = os.path.join(self.images_directory, img_coco["file_name"])

            with open(img_path, "rb") as file:
                image_bytes = file.read()

            in_dict = {
                "images": [image_bytes],
                "images_meta": [img_coco],
            }

            for file_key, group_name in self.file_to_group_mapping.items():
                if file_key not in self.annotation_files:
                    continue

                coco = self.coco_instances[file_key]
                ann_ids = coco.getAnnIds(img_id)
                anns = coco.loadAnns(ann_ids)

                height, width = img_coco["height"], img_coco["width"]

                group_data = self.get_group_data(
                    height,
                    width,
                    anns,
                    file_key,
                    has_keypoints=self.has_keypoints(group_name),
                )

                prefix = f"{group_name}/" if group_name else ""
                for key, value in group_data.items():
                    tensor_name = self.key_to_column_mapping.get(key, key)
                    in_dict[f"{prefix}{tensor_name}"] = [value]

            self.dataset.append(in_dict)

        self.dataset.commit("Finished ingestion")

    def structure(self):
        self.create_structure()
        self.ingest_columns()


def from_coco(
    images_directory: Union[str, pathlib.Path],
    annotation_files: Dict[str, Union[str, pathlib.Path]],
    dest: Union[str, pathlib.Path],
    dest_creds: Optional[Dict[str, str]] = None,
    key_to_column_mapping: Optional[Dict] = None,
    file_to_group_mapping: Optional[Dict] = None,
) -> dp.Dataset:
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
        key_to_column_mapping (Optional[Dict]): A one-to-one mapping between COCO keys and Dataset column names.
        file_to_group_mapping (Optional[Dict]): A one-to-one mapping between COCO annotation file names and Dataset group names.

    Returns:
        Dataset: The Dataset created from images and COCO annotations.

    Raises:
        CocoAnnotationMissingError: If one or many annotation key is missing from file.
    """

    dest = convert_pathlib_to_string_if_needed(dest)
    images_directory = standartize_path(
        convert_pathlib_to_string_if_needed(images_directory)
    )

    annotation_files = verify_coco_annotation_dict(annotation_files)

    dist_ds = dp.create(dest, dict(dest_creds) if dest_creds is not None else {})

    unstructured = COCOStructuredDataset(
        dataset=dist_ds,
        images_directory=images_directory,
        annotation_files=annotation_files,
        key_to_column_mapping=key_to_column_mapping,
        file_to_group_mapping=file_to_group_mapping,
    )

    unstructured.structure()

    return dist_ds
