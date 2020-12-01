import argparse
import os
import pickle
import json
import time

import numpy as np
import psutil
from PIL import Image

import hub
from hub.collections import dataset, tensor
from hub.log import logger


class CocoGenerator(dataset.DatasetGenerator):
    def __init__(self, args, tag):
        self._args = args
        self._tag = tag

    def meta(self):
        return {
            "segmentation": {"shape": (1,), "dtype": "uint32", "chunksize": 1000},
            "area": {"shape": (1,), "dtype": "uint32", "chunksize": 1000},
            "iscrowd": {"shape": (1,), "dtype": "uint8", "chunksize": 1000},
            "image_id": {"shape": (1,), "dtype": "int64"},
            "bbox": {"shape": (1,), "dtype": "uint16", "chunksize": 1000},
            "category_id": {"shape": (1,), "dtype": "uint32", "chunksize": 1000},
            "id": {"shape": (1,), "dtype": "uint32", "chunksize": 1000},
            "image": {"shape": (1,), "dtype": "uint32", "chunksize": 100},
        }

    def __call__(self, input):
        try:
            ds = {}
            # print(f"Image id: {input['image_id']}")
            ds["image_id"] = input["image_id"]
            info = input["info"]
            ds["image"] = np.empty(1, "uint32")
            ds["image"][0] = np.array(
                Image.open(
                    os.path.join(
                        self._args.dataset_path,
                        get_image_name(self._args, self._tag, input["image_id"]),
                    )
                ),
                dtype="uint32",
            )
            ds["segmentation"] = np.empty(1, "uint32")
            ds["area"] = np.empty(1, "uint32")
            ds["iscrowd"] = np.empty(1, "uint8")
            ds["bbox"] = np.empty(1, "uint16")
            ds["category_id"] = np.empty(1, "uint32")
            ds["id"] = np.empty(1, "uint32")

            ds["segmentation"][0] = [anno["segmentation"] for anno in info]
            ds["area"][0] = [anno["area"] for anno in info]
            ds["iscrowd"][0] = [anno["iscrowd"] for anno in info]
            ds["bbox"][0] = [anno["bbox"] for anno in info]
            ds["category_id"][0] = [anno["category_id"] for anno in info]
            ds["id"][0] = [anno["id"] for anno in info]

            logger.info(f"Tag: {self._tag}, Index: {input['index']}")
            return ds
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)


def get_image_name(args, tag, id):
    if args.year == "2014":
        return f"{tag}/COCO_{tag}2014_{str(id).zfill(12)}.jpg"
    elif args.year == "2017":
        return f"{tag}/{str(id).zfill(12)}.jpg"
    else:
        raise Exception("Invalid COCO year")


def load_dataset(args, tag):
    with open(
        os.path.join(args.dataset_path, f"annotations/instances_{tag}{args.year}.json"),
    ) as f:
        instances = json.load(f)
    # print(instances.keys())
    # print(f"Image sample: {instances['images'][0]}")
    print(f"Annotation sample: {instances['annotations'][0]}")
    # print(f"Categories sample: {instances['categories'][0]}")
    # categories = {cat["id"]: cat for cat in instances["categories"]}
    images = {image["id"]: [] for image in instances["images"]}
    print("Images loaded")
    for anno in instances["annotations"]:
        # anno["name"] = categories[anno["category_id"]]["name"]
        # anno["supercategory"] = categories[anno["category_id"]]["supercategory"]
        images[anno["image_id"]] += [anno]
    print("Annotations loaded")
    ids = [
        f
        for f in sorted(images.keys())
        if os.path.exists(os.path.join(args.dataset_path, get_image_name(args, tag, f)))
    ]
    print("Image ids selected")
    images = [
        {"image_id": image_id, "info": images[image_id], "index": index}
        for index, image_id in enumerate(ids)
    ]
    # print(f"Images selected by ids, sample {images[0]}")
    # images = images[:100000]
    # print(ids[:1000][-10:])
    # print("First 200 images slices")
    ds = dataset.generate(CocoGenerator(args, tag), images)
    return ds


def main():
    t1 = time.time()
    hub.init(processes=True, n_workers=psutil.cpu_count(), memory_limit=55e9)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        metavar="P",
        type=str,
        help="Path to coco2017 dataset",
        default="./data/COCOdataset2017",
    )
    parser.add_argument(
        "output_path",
        metavar="N",
        type=str,
        help="Dataset output path",
        default="COCOdataset2017",
    )
    parser.add_argument("year", metavar="Y", type=str, default="2017")
    args = parser.parse_args()
    tags = ["train", "val"]
    ds = {tag: load_dataset(args, tag) for tag in tags}
    for tag in ds:
        print(f"{tag}: {len(ds[tag])} samples")
    ds = dataset.concat([ds[tag] for tag in tags])
    # ds = ds["train"]
    ds.store(f"{args.output_path}")
    t2 = time.time()
    logger.info(f"Pipeline took {(t2 - t1) / 60} minutes")


if __name__ == "__main__":
    main()
