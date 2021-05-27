from hub.api.dataset import Dataset
import numpy as np
from tqdm import tqdm
from hub.core.storage import MemoryProvider, S3Provider, LocalProvider
from hub.util.cache_chain import get_cache_chain
import torch

# s3 = S3Provider("s3://snark-test/abc-large-3")
# local = LocalProvider("./yuo")
# prov = get_cache_chain([MemoryProvider("aty"), s3], [256 * 1024 * 1024,])
# prov["mnop"] = b"123"
# ds = Dataset(provider=prov)
# ds["image"] = np.ones((10000, 500, 500))
# ds.provider.flush()

ds = Dataset("s3://snark-test/abc-large-3")


ptds = ds.to_pytorch(workers=108)
dl = torch.utils.data.DataLoader(
    ptds,
    batch_size=1,
    num_workers=0,
)
for item in tqdm(ptds):
    # print(item)
    # print(item["image"][0][0])
    continue
    # item["image"][0][0]
    # 640 Mbps
    # item
    # print(item["image"][0][0])
