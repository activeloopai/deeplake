from typing import *
import os, sys, time, random, json, itertools, uuid, traceback

from .bucket import Bucket
from . import storage
from .storage.retry_wrapper import RetryWrapper
from hub.utils.store_control import StoreControlClient
import configparser
from hub.exceptions import S3CredsParseException


class Base:
    def __init__(self):
        pass

    def connect(self) -> Bucket:
        return Bucket(RetryWrapper(self._create_storage()))

    @staticmethod
    def _s3(
        bucket: Optional[str] = None,
        aws_creds_filepath: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> "creds.Base":
        if aws_access_key_id is not None or aws_secret_access_key is not None:
            assert aws_access_key_id is not None
            assert aws_secret_access_key is not None
            assert bucket is not None
            assert aws_creds_filepath is None
            return S3(bucket, aws_access_key_id, aws_secret_access_key)
        elif aws_creds_filepath is not None:
            config = configparser.ConfigParser()
            path = os.path.expanduser(aws_creds_filepath)
            path = os.path.abspath(path)
            config.read(path)

            if len(config.sections()) > 0:
                default = config.sections()[0]
                aws_access_key_id = config[default]["aws_access_key_id"]
                aws_secret_access_key = config[default]["aws_secret_access_key"]
            else:
                with open(aws_creds_filepath, "r") as f:
                    j = json.loads(f.read())
                    if not bucket:
                        bucket = j["bucket"]
                    aws_access_key_id = j["aws_access_key_id"]
                    aws_secret_access_key = j["aws_secret_access_key"]
            return S3(bucket, aws_access_key_id, aws_secret_access_key)
        else:
            # Try load from IAM role
            try:
                return S3(bucket, aws_access_key_id, aws_secret_access_key)
            except Exception as e:
                print("using default creds")
            config = StoreControlClient.get_config()
            if bucket is None:
                bucket = config["BUCKET"]

            aws_access_key_id = config["AWS_ACCESS_KEY_ID"]
            aws_secret_access_key = config["AWS_SECRET_ACCESS_KEY"]

            return S3(bucket, aws_access_key_id, aws_secret_access_key)

    def s3(
        self,
        bucket: Optional[str] = None,
        aws_creds_filepath: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> "creds.Base":
        return Recursive(
            self,
            self.__class__._s3(
                bucket, aws_creds_filepath, aws_access_key_id, aws_secret_access_key
            ),
        )

    @staticmethod
    def _gs(bucket: str, creds_path: Optional[str] = None):
        return GS(bucket, creds_path)

    def gs(self, bucket: str, creds_path: Optional[str] = None) -> "creds.Base":
        return Recursive(self, self.__class__._gs(bucket, creds_path))

    @staticmethod
    def _fs(dir: str) -> "creds.Base":
        dir = os.path.expanduser(dir)

        if not os.path.exists(dir):
            os.makedirs(dir)

        return FS(dir)

    def fs(self, dir: str) -> "creds.Base":
        return Recursive(self, self.__class__._fs(dir))

    def _create_storage(self) -> storage.Base:
        raise NotImplementedError()


class Recursive(Base):
    _core: "creds.Base" = None
    _wrapper: "creds.Base" = None

    def __init__(self, core: "creds.Base", wrapper: "creds.Base"):
        super().__init__()
        self._core = core
        self._wrapper = wrapper

    def _create_storage(self):
        return storage.Recursive(
            self._core._create_storage(), self._wrapper._create_storage()
        )


class S3(Base):

    _bucket: str = None
    _aws_access_key_id: str = None
    _aws_secret_access_key: str = None

    def __init__(self, bucket: str, aws_access_key_id: str, aws_secret_access_key: str):
        super().__init__()
        self._bucket = bucket
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key

    def _create_storage(self):
        return storage.S3(
            self._bucket, self._aws_access_key_id, self._aws_secret_access_key
        )


class GS(Base):
    _creds_path: str = None
    _bucket: str = None

    def __init__(self, bucket: str, creds_path: str):
        self._bucket = bucket
        if creds_path is not None:
            self._creds_path = os.path.expanduser(creds_path)
        else:
            self._creds_path = None

    def _create_storage(self):
        return storage.GS(self._bucket, self._creds_path)


class FS(Base):
    _dir: str = None

    def __init__(self, dir: str):
        super().__init__()
        self._dir = dir

    def _create_storage(self):
        return storage.FS(self._dir)
