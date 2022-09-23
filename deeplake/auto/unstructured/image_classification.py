import warnings
import numpy as np
from pathlib import Path
import os
import glob
import PIL  # type: ignore
from typing import Dict, List, Sequence, Tuple, Union

from hub.util.auto import ingestion_summary
from hub.util.exceptions import (
    InvalidPathException,
    TensorInvalidSampleShapeError,
)
from hub.core.dataset import Dataset

from tqdm import tqdm  # type: ignore

from .base import UnstructuredDataset

import hub

IMAGES_TENSOR_NAME = "images"
LABELS_TENSOR_NAME = "labels"


def _get_file_paths(directory: Path, relative_to: Union[str, Path] = "") -> List[Path]:
    g = glob.glob(os.path.join(directory, "**"), recursive=True)
    file_paths = []
    for path_str in g:
        if os.path.isfile(path_str):
            path = Path(path_str)
            if relative_to:
                relative_path = Path(path).relative_to(directory)
            else:
                relative_path = path
            file_paths.append(relative_path)
    return file_paths


def _class_name_from_path(path: Path) -> str:
    return path.parts[-2]


def _set_name_from_path(path: Path) -> str:
    return path.parts[-3]


class ImageClassification(UnstructuredDataset):
    def __init__(self, source: str):
        """Convert an unstructured dataset to a structured dataset.

        Note:
            Currently only supports computer vision (image) datasets.

        Args:
            source (str): The full path to the dataset.
                Can be a Hub cloud path of the form hub://username/datasetname. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                Can be a s3 path of the form s3://bucketname/path/to/dataset. Credentials are required in either the environment or passed to the creds argument.
                Can be a local file system path of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
                Can be a memory path of the form mem://path/to/dataset which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.

        Raises:
            InvalidPathException: If source is invalid.
        """

        super().__init__(source)

        self._abs_file_paths = _get_file_paths(self.source)
        self._rel_file_paths = _get_file_paths(self.source, relative_to=self.source)
        if len(self._abs_file_paths) == 0:
            raise InvalidPathException(
                f"No files found in {self.source}. Please ensure that the source path is correct."
            )

        self.set_names = self.get_set_names()
        self.class_names = self.get_class_names()

    # TODO: make lazy/memoized property
    def get_set_names(self) -> Tuple[str, ...]:
        # TODO: move outside class
        set_names = set()
        for file_path in self._abs_file_paths:
            set_names.add(_set_name_from_path(file_path))
        return tuple(sorted(set_names))  # TODO: lexicographical sorting

    # TODO: make lazy/memoized property
    def get_class_names(self) -> Tuple[str, ...]:
        # TODO: move outside class
        class_names = set()
        for file_path in self._abs_file_paths:
            class_names.add(_class_name_from_path(file_path))
        return tuple(sorted(class_names))  # TODO: lexicographical sorting

    def structure(  # type: ignore
        self,
        ds: Dataset,
        use_progress_bar: bool = True,
        generate_summary: bool = True,
        image_tensor_args: dict = {},
    ) -> Dataset:
        """Create a structured dataset.

        Args:
            ds (Dataset) : A Hub dataset object.
            use_progress_bar (bool): Defines if the method uses a progress bar. Defaults to True.
            generate_summary (bool): Defines if the method generates ingestion summary. Defaults to True.
            image_tensor_args (dict): Defines the sample compression of the dataset (jpeg or png).

        Returns:
            A hub dataset.

        """

        images_tensor_map = {}
        labels_tensor_map = {}

        use_set_prefix = len(self.set_names) > 1

        for set_name in self.set_names:
            if not use_set_prefix:
                set_name = ""

            images_tensor_name = os.path.join(set_name, IMAGES_TENSOR_NAME)
            labels_tensor_name = os.path.join(set_name, LABELS_TENSOR_NAME)
            images_tensor_map[set_name] = images_tensor_name.replace("\\", "/")
            labels_tensor_map[set_name] = labels_tensor_name.replace("\\", "/")

            # TODO: infer sample_compression
            ds.create_tensor(
                images_tensor_name.replace("\\", "/"),
                htype="image",
                **image_tensor_args,
            )
            ds.create_tensor(
                labels_tensor_name.replace("\\", "/"),
                htype="class_label",
                class_names=self.class_names,
            )

            paths = self._abs_file_paths
            skipped_files: list = []

            iterator = tqdm(
                paths,
                desc='Ingesting "%s" (%i files skipped)'
                % (self.source.name, len(skipped_files)),
                total=len(paths),
                disable=not use_progress_bar,
            )

        with ds, iterator:
            for file_path in iterator:
                image = hub.read(file_path)

                class_name = _class_name_from_path(file_path)

                label = np.uint32(self.class_names.index(class_name))

                set_name = _set_name_from_path(file_path) if use_set_prefix else ""

                # TODO: try to get all len(shape)s to match.
                # if appending fails because of a shape mismatch, expand dims (might also fail)
                try:
                    ds[images_tensor_map[set_name]].append(image)

                except TensorInvalidSampleShapeError:
                    im = image.array
                    reshaped_image = np.expand_dims(im, -1)
                    ds[images_tensor_map[set_name]].append(reshaped_image)

                except Exception:
                    skipped_files.append(file_path.name)
                    iterator.set_description(
                        'Ingesting "%s" (%i files skipped)'
                        % (self.source.name, len(skipped_files))
                    )
                    continue

                ds[labels_tensor_map[set_name]].append(label)

            if generate_summary:
                ingestion_summary(str(self.source), skipped_files)
            return ds
