import hub
from hub.schema import Image
import numpy as np

my_schema = {
    "abc": "uint32",
    "img": Image((1000, 1000, 3), dtype="uint16"),
}
ds = hub.Dataset(
    "./data/test_versioning/eg_3", shape=(100,), schema=my_schema, mode="w"
)
for i in range(100):
    ds["img", i] = i * np.ones((1000, 1000, 3))
a = ds.commit("first")

# chunk 7.0.0.0 gets rewritten
ds["img", 21] = 2 * ds["img", 21].compute()

# the rest part of the chunk stays intact
assert (ds["img", 21].compute() == 2 * 21 * np.ones((1000, 1000, 3))).all()
assert (ds["img", 22].compute() == 22 * np.ones((1000, 1000, 3))).all()
assert (ds["img", 23].compute() == 23 * np.ones((1000, 1000, 3))).all()

# other chunks are still accessed from original chunk, for eg chunk 11 that contains 35th sample has single copy
assert (ds["img", 35].compute() == 35 * np.ones((1000, 1000, 3))).all()

b = ds.commit("second")

# going back to first commit
ds.checkout(a)

# sanity check
assert (ds["img", 21].compute() == 21 * np.ones((1000, 1000, 3))).all()

ds.checkout("another", create=True)

ds["img", 21] = 3 * ds["img", 21].compute()
assert (
    ds["img", 21].compute() == 3 * 21 * np.ones((1000, 1000, 3))
).all()  # and not 6 * 21 as it would have been, had we checked out from b

ds.commit("first2")
ds.log()

ds.checkout("master")
ds.log()
assert (ds["img", 21].compute() == 2 * 21 * np.ones((1000, 1000, 3))).all()
