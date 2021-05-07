def s3_creds_exist():
    import boto3

    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except Exception:
        return False
    return True