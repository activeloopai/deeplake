import hub
import pytest
import os

# Scenario 1. User has AWS credentials, no hub creds


def test_s3_right_creds():
    try:
        bucket = hub.s3("snark-test", aws_creds_filepath=".secrets/s3").connect()
        bucket.blob_set("temporary_blob.txt", bytes(1))
        bucket.blob_get("temporary_blob.txt")
    except:
        pytest.fail("Unexpected error.. could not connect to s3")


def test_s3_wrong_creds():
    with open("./data/cache/wrong_creds.txt", "w") as f:
        f.write("empty")
    with pytest.raises(Exception):
        bucket = hub.s3(
            "snark-test", aws_creds_filepath="./data/cache/wrong_creds.txt"
        ).connect()
        bucket.blob_set("temporary_blob.txt", bytes(1))
        bucket.blob_get("temporary_blob.txt")
