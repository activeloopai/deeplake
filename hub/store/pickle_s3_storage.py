from hub.store.s3_storage import S3Storage
import copyreg
from hub.store.s3_storage import S3Storage


def save_s3_storage(obj):
    return obj.__class__, (
        obj.s3fs,
        obj.url,
        obj.public,
        None,
        None,
        None,
        obj.parallel,
        obj.endpoint_url,
    )


copyreg.pickle(S3Storage, save_s3_storage)
