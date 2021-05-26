from hub.api.dataset import Dataset
import numpy as np
from tqdm import tqdm
from hub.core.storage import MemoryProvider, S3Provider, LocalProvider
from hub.util.cache_chain import get_cache_chain
import time

s3 = S3Provider("s3://snark-test/abc-large-3")
local = LocalProvider("./yuo")
prov = get_cache_chain([MemoryProvider("aty"), local], [256 * 1024 * 1024,])
prov["mnop"] = b"123"
ds = Dataset(provider=prov)
start = time.time()
ds["image"] = np.ones((1000, 500, 500))
ds.provider.flush()
end = time.time()
print(end-start)


# ds = Dataset("s3://snark-test/abc-large-3")


# ptds = ds.to_pytorch(workers=108)
# for item in tqdm(ptds):
#     continue