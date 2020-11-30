import os
import time
from helper import generate_dataset, report

BUCKET = "s3://snark-hub/benchmark"
# BUCKET = "/tmp"


def aws_cli_copy(samples=30, chunksize=None, name="aws cli"):
    """
    Uploads dataset into S3 and then downlods using aws cli
    """
    path = "/tmp/create_s3"
    ds = generate_dataset(
        [(samples, 256, 256), (samples, 256, 256)], chunksize=chunksize
    )
    t0 = time.time()
    ds = ds.store(path)
    t1 = time.time()
    os.system(" ".join(["aws", "s3", "cp", path, f"{BUCKET}/{path}", "--recursive"]))
    t2 = time.time()
    os.system(" ".join(["aws", "s3", "cp", f"{BUCKET}/{path}", path, "--recursive"]))
    t3 = time.time()
    return {
        "name": name,
        "upload": t2 - t1,
        "download": t3 - t2,
        "write_to_fs": t1 - t0,
    }


def upload_and_download(samples=30, chunksize=None, name="hub"):
    """
    Uploads dataset into S3 and then downlods using hub package
    """
    ds = generate_dataset([(samples, 256, 256), (samples, 256, 256)], chunksize=1)
    t1 = time.time()
    ds = ds.store(f"{BUCKET}/transfer/upload")
    t2 = time.time()
    ds.store("/tmp/download")
    t3 = time.time()
    return {"name": name, "upload": t2 - t1, "download": t3 - t2}


if __name__ == "__main__":
    samples = 64
    chunksize = None
    import hub

    hub.init(processes=True, n_workers=8, threads_per_worker=1)

    r1 = upload_and_download(samples, chunksize=chunksize)
    r2 = aws_cli_copy(samples, chunksize=chunksize)

    report([r1, r2])
