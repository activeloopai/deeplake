import hub
import pytest
import os

# Scenario 1. User has AWS credentials, no hub creds


def test_s3_right_creds():
    status = "connected"
    try:
        bucket = hub.s3('snark-test', aws_creds_filepath=".secrets/s3").connect()
        bucket.blob_set('temporary_blob.txt', bytes(1))
        bucket.blob_get('temporary_blob.txt')
    except:
        status == "could not conntect"
        pass
    assert status =="connected" 

def test_s3_wrong_creds():
    status = "connected"
    with open("./data/cache/wrong_creds.txt", "w") as f:
        f.write("empty") 
    try:
        bucket = hub.s3('snark-test', aws_creds_filepath="./data/cache/wrong_creds.txt").connect()
        bucket.blob_set('temporary_blob.txt', bytes(1))
        bucket.blob_get('temporary_blob.txt')
    except:
        status = "could not conntect"
        pass
    assert status =="could not conntect" 


if __name__ == "__main__":

    print('Running Basic Tests')
    test_s3_right_creds()
    test_s3_wrong_creds()
    # test_wo_aws_or_hub_creds()
    # test_public_access_no_creds()
