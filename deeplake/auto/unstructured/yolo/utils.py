import os

import pathlib

from collections import defaultdict
from typing import Tuple, List, Union, Optional, DefaultDict

import deeplake
from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger
from deeplake.util.storage import storage_provider_from_path
from deeplake.util.path import convert_pathlib_to_string_if_needed

import numpy as np


class YoloData:
    def __init__(
        self,
        data_directory: Union[str, pathlib.Path],
        creds,
        annotations_directory: Optional[Union[str, pathlib.Path]] = None,
        class_names_file: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """Annotations can either be in hte data_directory, or in a separate annotations_directory"""

        self.root = convert_pathlib_to_string_if_needed(data_directory)
        self.provider = storage_provider_from_path(self.root, creds=creds)

        if annotations_directory:
            self.separate_annotations = True
            self.root_annotations = convert_pathlib_to_string_if_needed(
                annotations_directory
            )
            self.provider_annotations = storage_provider_from_path(
                self.root_annotations, creds=creds
            )
        else:
            self.separate_annotations = False
            self.root_annotations = self.root
            self.provider_annotations = self.provider

        if class_names_file:
            class_names_file = convert_pathlib_to_string_if_needed(class_names_file)
            self.root_class_names = os.path.dirname(class_names_file)
            self.provider_class_names = storage_provider_from_path(
                self.root_class_names, creds=creds
            )
            self.class_names = (
                self.provider_class_names.get_bytes(os.path.basename(class_names_file))
                .decode()
                .splitlines()
            )
        else:
            self.class_names = None

        (
            self.supported_images,
            self.supported_annotations,
            self.invalid_files,
            self.image_extensions,
            self.most_frequent_image_extension,
        ) = self.parse_data()

    def parse_data(
        self,
    ) -> Tuple[List[str], List[str], List[str], List[str], Optional[str]]:
        """Parses the given directory to generate a list of image and annotation paths.
        Returns:
            A tuple with, respectively, list of supported images, list of encountered invalid files, list of encountered extensions and the most frequent extension
        """
        supported_image_extensions = tuple(
            "." + fmt for fmt in HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"]
        )
        supported_images = []
        supported_annotations = []
        invalid_files = []
        image_extensions: DefaultDict[str, int] = defaultdict(int)

        if self.separate_annotations:
            for file in self.provider_annotations:
                if file.lower().endswith(".txt"):
                    supported_annotations.append(file)
                else:
                    invalid_files.append(file)

            for file in self.provider:
                if file.endswith(supported_image_extensions):
                    supported_images.append(file)
                    ext = file.rsplit(".", 1)[1]
                    image_extensions[ext] += 1
                else:
                    invalid_files.append(file)

        else:
            for file in self.provider:
                if file.endswith(".txt"):
                    supported_annotations.append(file)
                elif file.endswith(supported_image_extensions):
                    supported_images.append(file)
                    ext = pathlib.Path(file).suffix[
                        1:
                    ]  # Get extension without the . symbol
                    image_extensions[ext] += 1
                else:
                    invalid_files.append(file)

        if len(invalid_files) > 0:
            logger.warning(
                f"Encountered {len(invalid_files)} unsupported files in the data folders and annotation folders (if specified)."
                + "\nUp to first 10 invalid files are:\n"
                + "\n".join(invalid_files[0:10])
            )

        most_frequent_image_extension = max(
            image_extensions, key=lambda k: image_extensions[k], default=None
        )

        return (
            supported_images,
            supported_annotations,
            invalid_files,
            list(image_extensions.keys()),
            most_frequent_image_extension,
        )

    def read_yolo_file(self, file_name: str, is_box: bool = True):
        """
        Function reads a label.txt YOLO file and returns a numpy array of labels,
        and an object containing the coordinates. If is_box is True, the coordinates
        object is an (Nx4) array, where N is the number of bounding boxes in the annotation
        file. If is_box is Fales, we assume the coordinates represent a polygon, so the coordinates
        object is as list of length N, where each element is an Mx2 array, where M is the number
        points in each polygon, and N is the number of ploygons in the annotation file.
        """

        ann = self.get_annotation(file_name)

        return read_yolo_coordinates(ann, is_box=is_box, file_name=file_name)

    def get_full_path_image(self, image_name: str) -> str:
        return os.path.join(self.root, image_name)

    def get_image(
        self,
        image: str,
        is_link: Optional[bool] = False,
        creds_key: Optional[str] = None,
    ):
        if is_link:
            return deeplake.link(self.get_full_path_image(image), creds_key=creds_key)

        return deeplake.read(self.get_full_path_image(image), storage=self.provider)

    def get_annotation(self, annotation: str):
        return self.provider_annotations.get_bytes(annotation).decode()


def read_yolo_coordinates(ann_text: str, is_box: bool = True, file_name=None):
    """
    Function reads the yolo text annotation and returns a numpy array of labels,
    and an object containing the coordinates. If is_box is True, the coordinates
    object is an (Nx4) array, where N is the number of bounding boxes in the annotation
    file. If is_box is Fales, we assume the coordinates represent a polygon, so the coordinates
    object is as list of length N, where each element is an Mx2 array, where M is the number
    points in each polygon, and N is the number of ploygons in the annotation file.

    file_name is an option input for error handling purposes.
    """

    lines_split = ann_text.splitlines()

    yolo_labels = np.zeros(len(lines_split))

    # Initialize box and polygon coordinates in order to mypy to pass, since types are different. This is computationally negligible.
    yolo_coordinates_box = np.zeros((len(lines_split), 4))
    yolo_coordinates_poly = []

    # Go through each line and parse data
    for l, line in enumerate(lines_split):
        line_split = line.split()

        if is_box:
            yolo_coordinates_box[l, :] = np.array(
                (
                    float(line_split[1]),
                    float(line_split[2]),
                    float(line_split[3]),
                    float(line_split[4]),
                )
            )
        else:  # Assume it's a polygon
            coordinates = np.array([float(item) for item in line_split[1:]])

            if coordinates.size % 2 != 0:
                raise IngestionError(
                    f"Error in annotation filename: {file_name}. Polygons must have an even number of points."
                )

            yolo_coordinates_poly.append(
                coordinates.reshape((int(coordinates.size / 2), 2))
            )

        yolo_labels[l] = int(line_split[0])

    if is_box:
        return yolo_labels, yolo_coordinates_box
    else:
        return yolo_labels, yolo_coordinates_poly
