import os
import sys
import boto3
from botocore.client import Config

R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')

if not all([R2_BUCKET_NAME, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    print("Missing environment variables. Please set R2_BUCKET_NAME, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.")
    sys.exit(1)

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <directory_to_upload>")
    sys.exit(1)

directory_to_upload = sys.argv[1]

if not os.path.isdir(directory_to_upload):
    print(f"The specified path '{directory_to_upload}' is not a valid directory.")
    sys.exit(1)

s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)


def upload_directory(local_directory, bucket_name):
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, local_directory)

            try:
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f"Uploaded: '{local_path}' → '{s3_path}'")
            except Exception as e:
                print(f"Failed to upload '{local_path}': {e}")


print(f"Starting upload of '{directory_to_upload}' to R2 bucket '{R2_BUCKET_NAME}'...")
upload_directory(directory_to_upload, R2_BUCKET_NAME)
print("Upload completed successfully!")
