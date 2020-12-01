import hub
from hub.utils import Timer

if __name__ == "__main__":
    path = "./data/test/tfds_new/coco"
    with Timer("Eurosat TFDS"):
        out_ds = hub.Dataset.from_tfds("coco", num=100)

        ds = hub.Dataset(
            "./data/test/tfds_new/coco2", schema=out_ds.schema, shape=(10000,), mode="w"
        )
        print(out_ds.schema)
        for key in ds.keys:
            print(ds[key].chunksize)
        exit()
        res_ds = out_ds.store(path)
        ds = hub.load(path)
        print(ds)
        print(ds["image", 0].compute())
