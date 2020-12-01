import hub
from hub.utils import Timer
from hub import dev_mode

dev_mode()

if __name__ == "__main__":
    # path = "s3://snark-test/coco_dataset"
    path = "./data/test/coco"
    with Timer("Eurosat TFDS"):
        out_ds = hub.Dataset.from_tfds("coco", num=1000)

        res_ds = out_ds.store(path)
        ds = hub.load(path)
