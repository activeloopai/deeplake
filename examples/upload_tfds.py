import hub
from hub.utils import Timer

if __name__ == "__main__":
    with Timer("Eurosat TFDS"):
        out_ds = hub.Dataset.from_tfds('eurosat')
        res_ds = out_ds.store("./data/test/tfds_new/eurosat")