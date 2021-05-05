from hub.core.storage.s3.s3fs import S3Provider
import s3fs
import time
from hub.core.storage.local import LocalProvider
from tqdm import tqdm

l = LocalProvider("/home/ubuntu/Hub/")
vid = l["zoom_1.mp4"]
print(len(vid))
start = time.time()

# s3fs.S3FileSystem
m = S3Provider("snark-test/benchmarks/vid_s3fs/")
# m["abcdef.txt"] = b"12345"
# for i in range(100):
#     m[f"vid_{i}.mp4"] = vid

for i in tqdm(range(100)):
    m[f"vid_{i}.mp4"]

# print(m["abcdef.txt"])

end = time.time()

print("Time taken is", end-start)
