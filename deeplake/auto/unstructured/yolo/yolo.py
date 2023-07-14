import deeplake

from pathlib import Path
from typing import Dict, Optional, Union

from deeplake.core.dataset import Dataset
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger

from ..base import UnstructuredDataset
from ..util import DatasetStructure, TensorStructure
from .utils import YoloData

import numpy as np

from random import shuffle as rshuffle

from .constants import (
    DEFAULT_YOLO_COORDINATES_TENSOR_PARAMS,
    DEFAULT_YOLO_LABEL_TENSOR_PARAMS,
    DEFAULT_IMAGE_TENSOR_PARAMS,
)


class YoloDataset(UnstructuredDataset):
    def __init__(
        self,
        data_directory: str,
        class_names_file: Optional[str] = None,
        annotations_directory: Optional[str] = None,
        image_params: Optional[Dict] = None,
        label_params: Optional[Dict] = None,
        coordinates_params: Optional[Dict] = None,
        allow_no_annotation: Optional[bool] = False,
        verify_class_names: Optional[bool] = True,
        inspect_limit: Optional[int] = 1000,
        creds: Optional[Union[str, Dict]] = None,
        image_creds_key: Optional[str] = None,
    ):
        """Container for access to Yolo Data, parsing of key information, and conversions to a Deep Lake dataset"""

        super().__init__(data_directory)

        self.class_names_file = class_names_file
        self.data_directory = data_directory
        self.annotations_directory = annotations_directory

        self.allow_no_annotation = allow_no_annotation
        self.verify_class_names = verify_class_names
        self.creds = creds
        self.image_creds_key = image_creds_key
        self.inspect_limit = inspect_limit

        self.data = YoloData(
            self.data_directory,
            creds,
            self.annotations_directory,
            self.class_names_file,
        )
        self._validate_data()

        # Create a separate list of tuples with the intestion data (img_fn, annotation_fn).
        # We do this in advance so missing files are discovered before the ingestion process.
        self._create_ingestion_list()

        self._validate_ingestion_data()

        self._initialize_params(
            image_params or {}, label_params or {}, coordinates_params or {}
        )
        self._validate_image_params()

    def _parse_coordinates_type(self):
        """Function inspects up to inspect_limit annotation files in order to infer whether they are polygons or bounding boxes"""

        # If the htype or name of the coordinates is not specified (htype could be bbox or polygon), auto-infer it by reading some of the annotation files
        if (
            "htype" not in self.coordinates_params.keys()
            or "name" not in self.coordinates_params.keys()
        ):
            # Read the annotation files assuming they are polygons and check if there are any non-empty annotations without 4 coordinates
            coordinates_htype = "bbox"  # Initialize to bbox and change if contradicted
            coordinates_name = "boxes"  # Initialize to boxes and change if contradicted
            count = 0
            while count < min(self.inspect_limit, len(self.ingestion_data)):
                fn = self.ingestion_data[count][1]
                if fn is not None:
                    _, coordinates = self.data.read_yolo_coordinates(fn, is_box=False)
                    for c in coordinates:
                        coord_size = c.size
                        if coord_size > 0 and coord_size != 4:
                            coordinates_htype = "polygon"
                            coordinates_name = "polygons"

                            count = (
                                self.inspect_limit + 1
                            )  # Set this to exit the while loop
                            break

                        ## TODO: Add fancier math to see whether even coordinates with 4 elements could be polygons
                count += 1

        if "htype" not in self.coordinates_params.keys():
            self.coordinates_params["htype"] = coordinates_htype

        if "name" not in self.coordinates_params.keys():
            self.coordinates_params["name"] = coordinates_name

    def _initialize_params(self, image_params, label_params, coordinates_params):
        self.image_params = {
            **DEFAULT_IMAGE_TENSOR_PARAMS,
            **image_params,
        }

        self.coordinates_params = {
            **DEFAULT_YOLO_COORDINATES_TENSOR_PARAMS,
            **coordinates_params,
        }

        self.label_params = {
            **DEFAULT_YOLO_LABEL_TENSOR_PARAMS,
            **label_params,
        }

        self._parse_coordinates_type()

    def _create_ingestion_list(self):
        """Function creates a list of tuples (image_filename, annotation_filename) that is passed to a deeplake.compute ingestion function"""

        ingestion_data = []
        for img_fn in self.data.supported_images:
            base_name = Path(img_fn).stem
            if base_name + ".txt" in self.data.supported_annotations:
                ingestion_data.append((img_fn, base_name + ".txt"))
            else:
                if self.allow_no_annotation:
                    logger.warning(
                        f"Annotation was not found for {img_fn}. Empty annotation data will be appended for this image."
                    )

                else:
                    raise IngestionError(
                        f"Annotation was not found for {img_fn}. Please add an annotation for this image, of specify allow_no_annotation=True, which will automatically append an empty annotation to the Deep Lake dataset."
                    )
                ingestion_data.append((img_fn, None))

        self.ingestion_data = ingestion_data

    def prepare_structure(self) -> DatasetStructure:
        structure = DatasetStructure(ignore_one_group=True)
        self._add_annotation_tensors(structure)
        self._add_images_tensor(structure)

        return structure

    def _validate_data(self):
        if (
            len(self.data.supported_images) != len(self.data.supported_annotations)
            and self.allow_no_annotation == False
        ):
            raise IngestionError(
                "The number of supported images and annotations in the input data is not equal. Please ensure that each image has a corresponding annotation, or set allow_no_annotation = True"
            )

        if len(self.data.supported_images) == 0:
            raise IngestionError(
                "There are no supported images in the input data. Please verify the source directory."
            )

    def _validate_ingestion_data(self):
        if len(self.ingestion_data) == 0:
            raise IngestionError(
                "The data parser was not able to find any annotations corresponding to the images. Please check your directories, filename, and extenstions, or consider setting allow_no_annotation = True in order to upload empty annotations."
            )

    def _validate_image_params(self):
        if "name" not in self.image_params:
            raise IngestionError(
                "Image params must contain a name for the image tensor."
            )

    def _add_annotation_tensors(
        self,
        structure: DatasetStructure,
    ):
        structure.add_first_level_tensor(
            TensorStructure(
                name=self.label_params["name"],
                params={
                    i: self.label_params[i] for i in self.label_params if i != "name"
                },
            )
        )

        structure.add_first_level_tensor(
            TensorStructure(
                name=self.coordinates_params["name"],
                params={
                    i: self.coordinates_params[i]
                    for i in self.coordinates_params
                    if i != "name"
                },
            )
        )

    def _add_images_tensor(self, structure: DatasetStructure):
        img_params = self.image_params.copy()

        img_params["sample_compression"] = self.image_params.get(
            "sample_compression", self.data.most_frequent_image_extension
        )
        name = self.image_params.get("name")

        structure.add_first_level_tensor(
            TensorStructure(
                name=name,
                params={i: img_params[i] for i in img_params if i != "name"},
            )
        )

    def _ingest_data(self, ds: Dataset, progressbar: bool = True, num_workers: int = 0):
        """Functions appends the data to the dataset object using deeplake.compute"""

        if self.image_creds_key is not None:
            ds.add_creds_key(self.image_creds_key, managed=True)

        # Wrap tensor data needed by the deeplake.compute function into a net dict.
        tensor_meta = {
            "images": ds[self.image_params["name"]].meta,
            "labels": ds[self.label_params["name"]].meta,
            "coordinates": ds[self.coordinates_params["name"]].meta,
        }

        @deeplake.compute
        def append_data_bbox(data, sample_out, tensor_meta: Dict = tensor_meta):
            # If the ingestion data is None, create empty annotations corresponding to the file
            if data[1]:
                yolo_labels, yolo_coordinates = self.data.read_yolo_coordinates(
                    data[1], is_box=True
                )
            else:
                yolo_labels = np.zeros((0))
                yolo_coordinates = np.zeros((4, 0))

            sample_out.append(
                {
                    self.image_params["name"]: self.data.get_image(
                        data[0],
                        tensor_meta["images"].is_link,
                        self.image_creds_key,
                    ),
                    self.label_params["name"]: yolo_labels.astype(
                        tensor_meta["labels"].dtype
                    ),
                    self.coordinates_params["name"]: yolo_coordinates.astype(
                        tensor_meta["coordinates"].dtype
                    ),
                }
            )

        @deeplake.compute
        def append_data_polygon(data, sample_out, tensor_meta: Dict = tensor_meta):
            # If the ingestion data is None, create empty annotations corresponding to the file
            if data[1]:
                yolo_labels, yolo_coordinates = self.data.read_yolo_coordinates(
                    data[1], is_box=False
                )
            else:
                yolo_labels = np.zeros((0))
                yolo_coordinates = []

            sample_out.append(
                {
                    self.image_params["name"]: self.data.get_image(
                        data[0],
                        tensor_meta["images"].is_link,
                        self.image_creds_key,
                    ),
                    self.label_params["name"]: yolo_labels.astype(
                        tensor_meta["labels"].dtype
                    ),
                    self.coordinates_params["name"]: yolo_coordinates,
                }
            )

        if tensor_meta["coordinates"].htype == "bbox":
            append_data_bbox(tensor_meta=tensor_meta).eval(
                self.ingestion_data,
                ds,
                progressbar=progressbar,
                num_workers=num_workers,
            )
        else:
            append_data_polygon(tensor_meta=tensor_meta).eval(
                self.ingestion_data,
                ds,
                progressbar=progressbar,
                num_workers=num_workers,
            )

    def structure(self, ds: Dataset, progressbar: bool = True, num_workers: int = 0, shuffle: bool = True):  # type: ignore
        # Set class names in the dataset
        if self.data.class_names:
            ds[self.label_params["name"]].info["class_names"] = self.data.class_names

        # Set bounding box format in the dataset
        if ds[self.coordinates_params["name"]].meta.htype == "bbox":
            ds[self.coordinates_params["name"]].info["coords"] = {
                "type": "fractional",
                "mode": "CCWH",
            }

        if shuffle:
            rshuffle(self.ingestion_data)

        self._ingest_data(ds, progressbar, num_workers)

        if self.verify_class_names and self.data.class_names:
            labels = ds[self.label_params.get("name")].numpy(aslist=True)

            max_label = max(
                [l.max(initial=0) for l in labels]
            )  # Assume a label is 0 if array is empty. This is technically incorrect, but it's highly unlikely that all labels are empty

            if max_label != len(ds[self.label_params.get("name")].info.class_names) - 1:
                raise IngestionError(
                    "Dataset has been created but the largest numeric label in the annotations is inconsistent with the number of classes in the classes file."
                )
