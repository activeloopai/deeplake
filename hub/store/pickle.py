import copyreg

from hub.store.s3_file_system_replacement import S3FileSystemReplacement
from hub.api.dataset import Dataset


def save_s3_filesystem_replacement(obj: S3FileSystemReplacement):
    return obj.__class__, (None, None, None, obj.client_kwargs)


def save_dataset(obj: Dataset):
    mode = "a"
    obj.flush()
    return obj.__class__, (
        obj.url,
        mode,
        obj.shape,
        obj.schema,
        obj.token,
        obj._fs,
        obj.meta_information,
        obj.cache,
        obj.storage_cache,
        obj.lock_cache,
        obj.tokenizer,
        obj.lazy,
        obj._public,
    )


copyreg.pickle(S3FileSystemReplacement, save_s3_filesystem_replacement)
copyreg.pickle(Dataset, save_dataset)