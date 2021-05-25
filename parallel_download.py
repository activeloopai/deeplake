from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os
from hub.core.storage import S3Provider


import boto3
import tqdm

AWS_BUCKET = "snark-test"
OUTPUT_DIR = "downloads"

def download_one_file(bucket: str, output: str, client: boto3.client, s3_file: str):
    """
    Download a single file from S3
    Args:
        bucket (str): S3 bucket where images are hosted
        output (str): Dir to store the images
        client (boto3.client): S3 client
        s3_file (str): S3 object name
    """

    client.download_file(
        Bucket=bucket, Key=s3_file, Filename=os.path.join(output, s3_file.split("/")[-1])
    )
    # s3://snark-test/abc-small/image/chunks/02a0d588-bd4f-11eb-aeda-166ff600252b
    # self.client.get_object(
    #         Bucket=self.bucket,
    #         Key=path,
    #     )


files_to_download = S3Provider("s3://snark-test/abc-small/image/chunks")._list_keys()
files_to_download = [f"abc-small/image/chunks/{file}" for file in files_to_download]
# exit()
# Creating only one session and one client
session = boto3.Session()
client = session.client("s3")
# The client is shared between threads
func = partial(download_one_file, AWS_BUCKET, OUTPUT_DIR, client)

# List for storing possible failed downloads to retry later
failed_downloads = []

with tqdm.tqdm(desc="Downloading images from S3", total=len(files_to_download)) as pbar:
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Using a dict for preserving the downloaded file for each future, to store it as a failure if we need that
        futures = {
            executor.submit(func, file_to_download): file_to_download for file_to_download in files_to_download
        }
        for future in as_completed(futures):
            if future.exception():
                failed_downloads.append(futures[future])
            pbar.update(1)
if len(failed_downloads) > 0:
    print("Some downloads have failed. Saving ids to csv")
    with open(
        os.path.join(OUTPUT_DIR, "failed_downloads.csv"), "w", newline=""
    ) as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(failed_downloads)