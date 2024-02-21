import numpy as np
from pathlib import Path
from random import shuffle as rshuffle
import os
import glob
from typing import List, Tuple, Union

from deeplake.util.auto import ingestion_summary
from deeplake.util.exceptions import (
    InvalidPathException,
    TensorInvalidSampleShapeError,
)
from deeplake.core.dataset import Dataset


from .base import UnstructuredDataset

import deeplake

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
                Can be a Deep Lake cloud path of the form hub://username/datasetname. To write to Deep Lake cloud datasets, ensure that you are authenticated to Deep Lake (pass in a token using the 'token' parameter).
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
    def get_class_names(self) -> List[str]:
        # TODO: move outside class
        class_names = set()
        for file_path in self._abs_file_paths:
            class_names.add(_class_name_from_path(file_path))
        return list(sorted(class_names))  # TODO: lexicographical sorting

    def structure(  # type: ignore
        self,
        ds: Dataset,
        progressbar: bool = True,
        generate_summary: bool = True,
        shuffle: bool = True,
        image_tensor_args: dict = {},
        label_tensor_args: dict = {},
        num_workers: int = 0,
    ) -> Dataset:
        """Create a structured dataset.

        Args:
            ds (Dataset): A Deep Lake dataset object.
            progressbar (bool): Defines if the method uses a progress bar. Defaults to True.
            generate_summary (bool): Defines if the method generates ingestion summary. Defaults to True.
            shuffle (bool): Defines if the file paths should be shuffled prior to ingestion. Defaults to True.
            image_tensor_args (dict): Defines the parameters for the images tensor.
            label_tensor_args (dict): Defines the parameters for the class_labels tensor.
            num_workers (int): The number of workers passed to compute.

        Returns:
            A Deep Lake dataset.

        """

        images_tensor_map = {}
        labels_tensor_map = {}

        use_set_prefix = len(self.set_names) > 1

        for set_name in self.set_names:
            if not use_set_prefix:
                set_name = ""

            images_tensor_name = os.path.join(
                set_name, image_tensor_args.pop("name", IMAGES_TENSOR_NAME)
            )
            labels_tensor_name = os.path.join(
                set_name, label_tensor_args.pop("name", LABELS_TENSOR_NAME)
            )
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
                **label_tensor_args,
            )

        paths = self._abs_file_paths
        if shuffle:
            rshuffle(paths)

        skipped_files: List[str] = []

        @deeplake.compute
        def ingest_classification(file_path: Path, ds: Dataset):
            image = deeplake.read(file_path)
            class_name = _class_name_from_path(file_path)
            set_name = _set_name_from_path(file_path) if use_set_prefix else ""

            # if appending fails because of a shape mismatch, expand dims (might also fail)
            try:
                ds[images_tensor_map[set_name]].append(image)
            except TensorInvalidSampleShapeError:
                im = image.array
                reshaped_image = np.expand_dims(im, -1)
                ds[images_tensor_map[set_name]].append(reshaped_image)
            except Exception:
                skipped_files.append(file_path.name)
                ds[images_tensor_map[set_name]].append(None)

            ds[labels_tensor_map[set_name]].append(class_name)

        ingest_classification().eval(
            paths,
            ds,
            skip_ok=True,
            progressbar=progressbar,
            num_workers=num_workers,
        )

        if generate_summary:
            ingestion_summary(str(self.source), skipped_files)
        return ds
