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


class YoloData:
    def __init__(
        self,
        data_directory: Union[str, pathlib.Path],
        annotations_directory: Union[str, pathlib.Path],
        class_names_file: Union[str, pathlib.Path],
        creds,
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

        (
            self.supported_images,
            self.supported_annotations,
            self.invalid_files,
            self.image_extensions,
            self.most_frequent_image_extension,
        ) = self.parse_data()

        self.class_names = (
            self.parse_class_names(class_names_file) if class_names_file else None
        )

    def parse_class_names(self, class_names_file: Union[str, pathlib.Path]):
        """Parses the file with class names into a list of strings"""
        names = self.get_text_file(class_names_file)

        return names.splitlines()

    def parse_data(self) -> Tuple[List[str], List[str], List[str], Optional[str]]:
        """Parses the given directory to generate a list of image and annotation paths.
        Returns:
            A tuple with, respectively, list of supported images, list of encountered invalid files, list of encountered extensions and the most frequent extension
        """
        supported_image_extensions = tuple(
            HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"]
        )
        supported_images = []
        supported_annotations = []
        invalid_files = []
        image_extensions: DefaultDict[str, int] = defaultdict(int)

        if self.separate_annotations:
            for file in self.provider_annotations:
                if file.endswith(".txt"):
                    supported_annotations.append(file)
                else:
                    invalid_files.append(file)

            for file in self.provider:
                if file.endswith(supported_image_extensions):
                    supported_images.append(file)
                    ext = pathlib.Path(file).suffix[
                        1:
                    ]  # Get extension without the . symbol
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
                f"Encountered {len(invalid_files)} unsupported files the data folders and annotation folders (if specified)."
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

    def read_yolo_coordinates(self, file_name: str, is_box: bool = True):
        """
        Function reads a label.txt YOLO file and returns a numpy array of labels,
        and an object containing the coordinates. If is_box is True, the coordinates
        object is an (Nx4) array, where N is the number of bounding boxes in the annotation
        file. If is_box is Fales, we assume the coordinates represent a polygon, so the coordinates
        object is as list of length N, where each element is an Mx2 array, where M is the number
        points in each polygon, and N is the number of ploygons in the annotation file.
        """

        ann = self.get_text_file(file_name)
        lines_split = ann.splitlines()

        yolo_labels = np.zeros(len(lines_split))

        if is_box:
            yolo_coordinates = np.zeros((len(lines_split), 4))
        else:
            yolo_coordinates = []

        # Go through each line and parse data
        for l, line in enumerate(lines_split):
            line_split = line.split()

            if is_box:
                yolo_coordinates[l, :] = np.array(
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
                        f"Error ih annotation {fn}. Polygons must have an even number of points."
                    )

                yolo_coordinates.append(
                    coordinates.reshape((int(coordinates.size / 2), 2))
                )

            yolo_labels[l] = int(line_split[0])

        return yolo_labels, yolo_coordinates

    def get_full_path_image(self, image_name: str) -> str:
        return os.path.join(self.root, image_name)

    def get_full_path_annotation(self, annotation_name: str) -> str:
        return os.path.join(self.root_annotations, annotation_name)

    def get_image(self, image: str, is_link: bool, creds_key: str):
        if is_link:
            return deeplake.link(self.get_full_path_image(image), creds_key=creds_key)

        return deeplake.read(self.get_full_path_image(image), storage=self.provider)

    def get_text_file(self, file_name: str):
        return self.provider_annotations.get_bytes(
            self.get_full_path_annotation(file_name)
        ).decode()
