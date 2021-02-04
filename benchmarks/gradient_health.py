"""mimiciii_cxr dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
import os
import posixpath
from typing import Dict, Any

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from fsspec import AbstractFileSystem
from s3fs import S3FileSystem
import argparse
import pandas as pd
import pydicom
import numpy as np

import hub
from hub import schema
from hub.utils import Timer
from ee.backend.synchronizer import RedisSynchronizer, ProcessSynchronizer

_CITATION = """
@article{Johnson2019,
abstract = {Chest radiography is an extremely powerful imaging modality, allowing for a detailed inspection of a patient's chest, but requires specialized training for proper interpretation. With the advent of high performance general purpose computer vision algorithms, the accurate automated analysis of chest radiographs is becoming increasingly of interest to researchers. Here we describe MIMIC-CXR, a large dataset of 227,835 imaging studies for 65,379 patients presenting to the Beth Israel Deaconess Medical Center Emergency Department between 2011-2016. Each imaging study can contain one or more images, usually a frontal view and a lateral view. A total of 377,110 images are available in the dataset. Studies are made available with a semi-structured free-text radiology report that describes the radiological findings of the images, written by a practicing radiologist contemporaneously during routine clinical care. All images and reports have been de-identified to protect patient privacy. The dataset is made freely available to facilitate and encourage a wide range of research in computer vision, natural language processing, and clinical data mining.},
author = {Johnson, Alistair E.W. and Pollard, Tom J. and Berkowitz, Seth J. and Greenbaum, Nathaniel R. and Lungren, Matthew P. and Deng, Chih Ying and Mark, Roger G. and Horng, Steven},
doi = {10.1038/s41597-019-0322-0},
file = {:Users/zl190/Downloads/s41597-019-0322-0.pdf:pdf},
issn = {20524463},
journal = {Scientific data},
number = {1},
pages = {317},
pmid = {31831740},
publisher = {Springer US},
title = {{MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports}},
url = {http://dx.doi.org/10.1038/s41597-019-0322-0},
volume = {6},
year = {2019}
}
"""

_DESCRIPTION = """
The MIMIC Chest X-ray (MIMIC-CXR) Database v2.0.0 is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. The dataset is intended to support a wide body of research in medicine including image understanding, natural language processing, and decision support.

The MIMIC-CXR dataset must be downloaded separately after reading and agreeing 
to a Research Use Agreement. To do so, please follow the instructions on the 
website, https://physionet.org/content/mimic-cxr/2.0.0/
"""

_LABELS = {
    -1.0: "uncertain",
    1.0: "positive",
    0.0: "negative",
    99.0: "unmentioned",
}

MAX_TEXT_LEN = 2000
MY_TEXT = schema.Text(max_shape=(MAX_TEXT_LEN,), dtype="uint8")


class MimiciiiCxr:
    def __init__(self, image_size=512):
        self._image_size = image_size

    def _info(self):
        image_size = self._image_size
        MAX_IMAGE_COUNT = 30

        return {
            "subject_id": MY_TEXT,
            "study_id": MY_TEXT,
            "study_date": MY_TEXT,
            "study_time": MY_TEXT,
            "report": MY_TEXT,
            "label_chexpert": schema.Sequence(
                shape=14, dtype=schema.ClassLabel(names=list(_LABELS.values()))
            ),
            "label_negbio": schema.Sequence(
                shape=14, dtype=schema.ClassLabel(names=list(_LABELS.values()))
            ),
            "image": schema.Tensor(
                shape=(None, image_size, image_size, 1),
                max_shape=(MAX_IMAGE_COUNT, image_size, image_size, 1),
                dtype="uint16",
            ),
            # "dicom_id": schema.Tensor(max_shape=(MAX_IMAGE_COUNT,), dtype="object"),
            # "columns": schema.Tensor(max_shape=(MAX_IMAGE_COUNT,), dtype="object"),
            # "viewPosition": schema.Tensor(max_shape=(MAX_IMAGE_COUNT,), dtype="object"),
            # "viewCodeSequence_CodeMeaning": schema.Tensor(
            #     max_shape=(MAX_IMAGE_COUNT,), dtype="object"
            # ),
            # "patientOrientationCodeSequence_CodeMeaning": schema.Tensor(
            #     max_shape=(MAX_IMAGE_COUNT,), dtype="object"
            # ),
            # "procedureCodeSequence_CodeMeaning": schema.Tensor(
            #     max_shape=(MAX_IMAGE_COUNT,), dtype="object"
            # ),
            # "performedProcedureStepDescription": schema.Tensor(
            #     max_shape=(MAX_IMAGE_COUNT,), dtype="object"
            # ),
        }

    def _intermitidate_schema(self):
        return {"row": MY_TEXT}

    def _build_pcollection(
        self,
        filepath: str,
        manual_dir: str,
        fs: AbstractFileSystem,
        output_dir: str,
        args: Dict[str, Any],
    ):
        # beam = tfds.core.lazy_imports.apache_beam
        # pydicom = tfds.core.lazy_imports.pydicom
        # pd = tfds.core.lazy_imports.pandas
        image_size = self._image_size
        # image_size = 512 if self.builder_config.name is "512" else 2048

        def _right_size(row):
            row = row["row"]
            data = row.split(",")
            if len(data) == 11:
                return [{"row": row}]
            else:
                return []

        result = fs.cat_file(os.path.join(manual_dir, "mimic-cxr-2.0.0-chexpert.csv"))
        with BytesIO(result) as f:
            chexpert_df = pd.read_csv(f)
        result = fs.cat_file(os.path.join(manual_dir, "mimic-cxr-2.0.0-negbio.csv"))
        with BytesIO(result) as f:
            negbio_df = pd.read_csv(f)
        chexpert_df = chexpert_df.fillna(99.0)
        negbio_df = negbio_df.fillna(99.0)

        def _check_files(row):
            row = row["row"]
            (
                study_id,
                subject_id,
                split,
                dicom_id,
                performedProcedureStepDescription,
                ViewPosition,
                StudyDate,
                StudyTime,
                procedureCodeSequence_CodeMeaning,
                ViewCodeSequence_CodeMeaning,
                patientOrientationCodeSequence_CodeMeaning,
            ) = row.split(",")
            basepath = "{}/files/p{}/p{}/s{}".format(
                manual_dir, subject_id[0:2], subject_id, study_id
            )
            paths = ["{}/{}.dcm".format(basepath, d) for d in dicom_id.split("|")]
            paths.append(basepath + ".txt")
            for path in paths:
                if not fs.exists(path):
                    return []

            # Job Graph Too Large

            try:
                negbio_values = negbio_df[
                    (negbio_df["subject_id"] == int(subject_id))
                    & (negbio_df["study_id"] == int(study_id))
                ].values.tolist()[0][2:]
                chexpert_values = chexpert_df[
                    (chexpert_df["subject_id"] == int(subject_id))
                    & (chexpert_df["study_id"] == int(study_id))
                ].values.tolist()[0][2:]
                negbio_values = [_LABELS[v] for v in negbio_values]
                chexpert_values = [_LABELS[v] for v in chexpert_values]
            except Exception as e:
                print(subject_id)
                print(study_id)
                return []

            return [{"row": row}]

        def _process_example(row):
            def fast_histogram_equalize(image):
                """histogram for integer based images"""
                image = image - tf.reduce_min(image)
                image = tf.cast(image, tf.int32)
                histogram = tf.math.bincount(image)
                cdf = tf.cast(tf.math.cumsum(histogram), tf.float32)
                cdf = cdf / cdf[-1]
                return tf.gather(params=cdf, indices=image)

            row = row["row"]
            (
                study_id,
                subject_id,
                split,
                dicom_id,
                performedProcedureStepDescription,
                ViewPosition,
                StudyDate,
                StudyTime,
                procedureCodeSequence_CodeMeaning,
                ViewCodeSequence_CodeMeaning,
                patientOrientationCodeSequence_CodeMeaning,
            ) = row.split(",")

            # Job Graph Too Large
            result = fs.cat_file(
                os.path.join(manual_dir, "mimic-cxr-2.0.0-chexpert.csv")
            )
            with BytesIO(result) as f:
                chexpert_df = pd.read_csv(f)
            result = fs.cat_file(os.path.join(manual_dir, "mimic-cxr-2.0.0-negbio.csv"))
            with BytesIO(result) as f:
                negbio_df = pd.read_csv(f)
            chexpert_df = chexpert_df.fillna(99.0)
            negbio_df = negbio_df.fillna(99.0)

            dicom_id = dicom_id.split("|")
            ViewPosition = ViewPosition.split("|")
            ViewCodeSequence_CodeMeaning = ViewCodeSequence_CodeMeaning.split("|")
            patientOrientationCodeSequence_CodeMeaning = (
                patientOrientationCodeSequence_CodeMeaning.split("|")
            )
            procedureCodeSequence_CodeMeaning = procedureCodeSequence_CodeMeaning.split(
                "|"
            )
            performedProcedureStepDescription = performedProcedureStepDescription.split(
                "|"
            )
            basepath = "{}/files/p{}/p{}/s{}".format(
                manual_dir, subject_id[0:2], subject_id, study_id
            )

            dicom_paths = ["{}/{}.dcm".format(basepath, d) for d in dicom_id]
            images = []
            rows = []
            columns = []
            for dicom_path in dicom_paths:
                result = fs.cat_file(dicom_path)
                with BytesIO(result) as d:
                    ds = pydicom.dcmread(d)
                    image = tf.squeeze(tf.constant(ds.pixel_array))[..., None]
                    row, col, channel = image.shape
                    image = fast_histogram_equalize(image)
                    images.append(
                        tf.cast(
                            tf.round(
                                tf.image.resize_with_pad(image, image_size, image_size)
                            ),
                            tf.uint16,
                        ).numpy()
                    )
                    rows.append(row)
                    columns.append(col)

            negbio_values = negbio_df[
                (negbio_df["subject_id"] == int(subject_id))
                & (negbio_df["study_id"] == int(study_id))
            ].values.tolist()[0][2:]
            chexpert_values = chexpert_df[
                (chexpert_df["subject_id"] == int(subject_id))
                & (chexpert_df["study_id"] == int(study_id))
            ].values.tolist()[0][2:]
            negbio_values = [_LABELS[v] for v in negbio_values]
            chexpert_values = [_LABELS[v] for v in chexpert_values]

            images = np.array(images)

            return {
                "subject_id": subject_id,
                "study_id": study_id,
                "study_date": StudyDate.split("|")[0],
                "study_time": StudyTime.split("|")[0],
                "report": fs.cat_file(basepath + ".txt").decode("utf-8"),
                "label_chexpert": chexpert_values,
                "label_negbio": negbio_values,
                "image": images,
                # "rows": rows,
                # "columns": columns,
                # "dicom_id": dicom_id[0],
                # "viewPosition": ViewPosition,
                # "viewCodeSequence_CodeMeaning": ViewCodeSequence_CodeMeaning,
                # "patientOrientationCodeSequence_CodeMeaning": patientOrientationCodeSequence_CodeMeaning,
                # "procedureCodeSequence_CodeMeaning": procedureCodeSequence_CodeMeaning,
                # "performedProcedureStepDescription": performedProcedureStepDescription,
            }

        result = fs.cat_file(filepath)
        with BytesIO(result) as f:
            lines = [{"row": line.decode("utf-8")} for line in f.readlines()[1:]]

        schema_ = self._info()
        schemai = self._intermitidate_schema()
        print("Number of samples: ", len(lines))
        lines = lines[:400]
        if args.redisurl:
            sync = RedisSynchronizer(host=args.redisurl, password="5241590000000000")
        elif args.scheduler == "processed":
            sync = ProcessSynchronizer("./data/process_sync")
        else:
            sync = None

        ds1 = hub.transform(
            schemai, scheduler=args.scheduler, workers=args.workers, synchronizer=sync
        )(_right_size)(lines)
        ds1 = ds1.store(f"{output_dir}/ds1")
        ds2 = hub.transform(
            schemai, scheduler=args.scheduler, workers=args.workers, synchronizer=sync
        )(_check_files)(ds1)
        ds2 = ds2.store(f"{output_dir}/ds2")
        print("LEN:", len(ds2))
        ds3 = hub.transform(
            schema_, scheduler=args.scheduler, workers=args.workers, synchronizer=sync
        )(_process_example)(ds2)
        ds3.store(f"{output_dir}/ds3")
        print("Success")


def main():
    DEFAULT_WORKERS = 3
    DEFAULT_SCHEDULER = "ray_generator"
    if DEFAULT_SCHEDULER == "ray_generator":
        DEFAULT_REDIS_URL = (
            os.environ["RAY_HEAD_IP"] if "RAY_HEAD_IP" in os.environ else "localhost"
        )
    else:
        DEFAULT_REDIS_URL = False
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", default="s3://snark-gradient-raw-data/mimic-cxr-2.0.0"
    )
    parser.add_argument("-o", "--output", default="s3://snark-gradient-raw-data/output")
    parser.add_argument("-w", "--workers", default=DEFAULT_WORKERS)
    parser.add_argument("-s", "--scheduler", default=DEFAULT_SCHEDULER)
    parser.add_argument("-r", "--redisurl", default=DEFAULT_REDIS_URL)
    args = parser.parse_args()
    handle = MimiciiiCxr()
    fs = S3FileSystem(default_block_size=2 ** 26)
    manual_dir = args.input
    filepath = posixpath.join(manual_dir, "train-split.csv")
    handle._build_pcollection(filepath, manual_dir, fs, args.output, args)


if __name__ == "__main__":
    main()