from hub.core.storage.s3.s3_metaflow.s3 import S3
import s3fs
import time
from hub.core.storage.local import LocalProvider
from tqdm import tqdm

l = LocalProvider("/home/ubuntu/Hub/")
vid = l["zoom_1.mp4"]
print(len(vid))
start = time.time()
# s3fs.S3FileSystem
m = S3(s3root="s3://snark-test/benchmarks/vid_s3fs/")
# m["abcdef.txt"] = b"12345"
# for i in range(100):
#     m[f"vid_{i}.mp4"] = vid
ls = [f"vid_{i}.mp4" for i in range(100)]
# for i in tqdm(range(100)):
m.get_many(ls)

# print(m["abcdef.txt"])

end = time.time()

print("Time taken is", end-start)
