#!/usr/bin/env python3

import sys
import os
import boto3

RUN_ID = os.environ.get("RUN_ID")
RUNNER_TYPE = os.environ.get("TYPE")
PYTHON = os.environ.get("PYTHON")

KEY = "indra/wheels"

if sys.argv[1] == "deeplake-v4":
    W_PATH = f"python/dist/centos_wheels_{RUN_ID}_{RUNNER_TYPE}_{PYTHON}.zip"
    os.system(f"cd python/dist; zip centos_wheels_{RUN_ID}_{RUNNER_TYPE}_{PYTHON}.zip *.whl")
elif sys.argv[1] == "libdeeplake":
    W_PATH = f"python-v3/dist/centos_wheels_{RUN_ID}_{RUNNER_TYPE}_{PYTHON}.zip"
    os.system(f"cd python-v3/dist; zip centos_wheels_{RUN_ID}_{RUNNER_TYPE}_{PYTHON}.zip *.whl")
elif sys.argv[1] == "libdeeplake-dev":
    KEY = "indra/dev-wheels/latest"
    W_PATH = os.environ.get("W_PATH")
else:
    print("wrong preset")
    sys.exit(1)

S3_CONSOLE_URL = f"https://us-east-1.console.aws.amazon.com/s3/object/activeloop-platform-tests?region=us-east-1&bucketType=general&prefix={KEY}/{W_PATH}"

client = boto3.client("s3")

with open(W_PATH, "rb") as file:
    client.upload_fileobj(
        Fileobj=file, Bucket="activeloop-platform-tests", Key=f"{KEY}/{W_PATH}"
    )

with open(os.environ.get("GITHUB_STEP_SUMMARY"), "a", encoding="utf-8") as file:
    file.write(f"|{PYTHON}_{RUNNER_TYPE}|\n")
    file.write("|-|\n")
    file.write(f"|[s3://activeloop-platform-tests/{KEY}/{W_PATH}]({S3_CONSOLE_URL})|\n")
