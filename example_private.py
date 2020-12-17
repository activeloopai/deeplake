import hub
import numpy as np
schema = {
    "abc" : "int32"
}
ds = hub.Dataset("abhinav/try_pvt21", schema=schema, mode="w", shape=(5,), public=False)
ds["abc", 0] = 7
ds.commit()
