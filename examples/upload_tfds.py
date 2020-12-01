import hub
from hub.utils import Timer

if __name__ == "__main__":
    path = "./data/test/tfds_new/coco"
    with Timer("Eurosat TFDS"):
        out_ds = hub.Dataset.from_tfds("coco", num=10000)

        res_ds = out_ds.store(path)
        ds = hub.load(path)
