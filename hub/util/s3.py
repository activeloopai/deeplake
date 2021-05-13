def has_s3_credentials():
    """Checks if s3 credentials are accessible to boto3."""
    import boto3

    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except Exception:
        return False
    return True
