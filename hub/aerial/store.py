import fsspec
import s3fs
import gcsfs
import zarr
import configparser


def _read_aws_creds(creds_path: str):
    parser = configparser.ConfigParser()
    parser.read(creds_path)
    return {section: dict(parser.items(section)) for section in parser.sections()}


def _get_fs_and_path(url: str, creds: dict = None):
    if url.startswith("s3://"):
        creds = creds or dict()
        creds = _read_aws_creds(creds) if isinstance(creds, str) else creds
        return (
            fsspec.filesystem(
                "s3",
                key=creds.get("aws_access_key_id"),
                secret=creds.get("aws_secret_access_key"),
            ),
            url[5:],
        )
    elif url.startswith("gcs://"):
        return gcsfs.GCSFileSystem(token=creds), url[6:]
    elif url.startswith("abs://"):
        # TODO: Azure
        raise NotImplementedError()
    elif (
        url.startswith("../")
        or url.startswith("./")
        or url.startswith("/")
        or url.startswith("~/")
    ):
        return fsspec.filesystem("file"), url
    else:
        raise NotImplementedError()


def _get_storage_map(url: str, creds: dict = None):
    fs, path = _get_fs_and_path(url, creds)
    fs_map = fs.get_mapper(path, check=False, create=True)
    return (
        fs,
        path,
        fs_map,
    )


def get_storage_map(url: str, creds: dict = None, memcache: float = None):
    fs, path, store = _get_storage_map(url, creds)
    # TODO: Make use that fs.listdir and store.get do not cache locally filenames,
    # because in that case if something is added to s3 or gcs it won't be notified by the program
    if (
        store.get(".zarray") is None
        and store.get(".zgroup") is None
        and len(fs.listdir(path)) > 0
    ):
        raise NotZarrFolderException(
            "This url is not empty but not zarr url either, for safety reasons refusing to overwrite this folder"
        )
    return store if not memcache else zarr.LRUStoreCache(store, memcache * (2 ** 20))


class NotZarrFolderException(Exception):
    pass
