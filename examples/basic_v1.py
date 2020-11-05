from hub import Dataset, features
import numpy as np
from pprint import pprint
import time

if False:
    t1 = time.time()
    ds = Dataset(
        "davit/basic6",
        shape=(4,),
        schema={
            "image": features.Tensor((512, 512), dtype="float"),
            "label": features.Tensor((512, 512), dtype="float"),
        },
    )
    t2 = time.time()
    ds["image"][:] = np.zeros((4, 512, 512))
    ds["label"][:] = np.zeros((4, 512, 512))
    t3 = time.time()
    ds.commit()

t4 = time.time()
ds = Dataset(
    "./data/test/davit/basic10",
    shape=(4,),
    mode="a",
    schema={
        "image": features.Tensor((512, 512), dtype="float"),
        "label": features.Tensor((512, 512), dtype="float"),
    },
)

ds["image"][0] = 1

ds = Dataset("./data/test/davit/basic10", mode="a")
t5 = time.time()
print(ds["image"][0].compute())
t6 = time.time()
exit()

pprint(
    {
        "dataset creating": t2 - t1,
        "assigning": t3 - t2,
        "commting": t4 - t3,
        "loading dataset": t5 - t4,
        "accessing": t6 - t5,
    }
)
