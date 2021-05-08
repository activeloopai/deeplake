def s3_creds_exist():
    """Checks whether s3_creds exist in the environment"""
    import boto3

    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except Exception:
        return False
    return True
