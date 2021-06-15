import hub_v1
from hub_v1.utils import Timer
import tensorflow_datasets as tfds


def benchmark_coco(num=5):
    with tfds.testing.mock_data(num_examples=num):
        ds = hub_v1.Dataset.from_tfds("coco", num=num)

        res_ds = ds.store(
            "./data/test_tfds/coco", length=num
        )  # mock data doesn't have length, so explicitly provided


if __name__ == "__main__":
    nums = [5, 100, 1000, 10000, 100000]
    for num in nums:
        with Timer("Coco " + str(num) + " samples"):
            benchmark_coco(num)
