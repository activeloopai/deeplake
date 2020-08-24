import tensorflow as tf
import tensorflow_datasets as tfds
from hub.collections import dataset, tensor


def is_equal(lds, rds):
    output = True
    for lex, rex in zip(lds, rds):
        if lex.keys() != rex.keys():
            return False
        for key in lex.keys():
            comparsion = lex[key].numpy() == rex[key].numpy()
            output *= comparsion.all()
    return output


def check_one(name):
    ds_tf, info_tf = tfds.load(
        name,
        as_supervised=True,
        split="train",
        shuffle_files=True,
        with_info=True
    )
    features = info_tf.features.keys()
    ds_hb = dataset.from_tensorflow(ds_tf, features, 100)
    stored = ds_hb.store(f"./tmp/{name}")
    stored_tf = stored.to_tensorflow()
    stored_pt = stored.to_pytorch()
    return is_equal(stored_tf, stored_pt)


def check_many(names):
    output = []
    for name in names:
        output += [check_one(name)]
    return output


def test_from_tensorflow():
    names = ['mnist']
    assert check_many(names)