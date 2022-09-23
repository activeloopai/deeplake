import time
import os
from hub.core.storage.provider import StorageProvider
from io import BytesIO
import posixpath
import pickle
from typing import Dict, Optional, Union
from hub.util.hash import hash_inputs
import logging

try:
    from httplib2 import Http  # type: ignore
    from googleapiclient import discovery  # type: ignore
    from googleapiclient.http import (  # type: ignore
        MediaIoBaseDownload,
        MediaIoBaseUpload,
    )
    from googleapiclient.errors import HttpError  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from google.oauth2.credentials import Credentials  # type: ignore

    _GDRIVE_PACKAGES_INSTALLED = True
except ImportError:
    _GDRIVE_PACKAGES_INSTALLED = False


logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.install",
    "https://www.googleapis.com/auth/drive.readonly",
]
FOLDER = "application/vnd.google-apps.folder"
FILE = "application/octet-stream"


class GoogleDriveIDManager:
    """Class used to make google drive path to id maps"""

    def __init__(self, drive, root: str, root_id: Optional[str] = None):
        self.path_id_map: Dict[str, str] = {}
        self.drive = drive
        self.root_path = root
        self.root_id = root_id if root_id else self.find_id(root)

    def find_id(self, path):
        """Find google drive id given path of folder"""

        dirname, basename = posixpath.split(path)
        try:
            file_list = (
                self.drive.files()
                .list(
                    q=f"""'{self.find_id(dirname) if dirname else 'root'}' in parents and 
                        name = '{basename}' and 
                        trashed = false""",
                    supportsAllDrives="true",
                    includeItemsFromAllDrives="true",
                    spaces="drive",
                    fields="files(id)",
                )
                .execute()
            )
        except HttpError:
            file_list = {"files": []}

        if len(file_list["files"]) > 0:
            id = file_list["files"][0].get("id")
        else:
            id = None
        return id

    def makemap(self, root_id, root_path):
        """Make mapping from google drive paths to ids for all files and folders under root"""

        file_list = (
            self.drive.files()
            .list(
                q=f"'{root_id}' in parents and trashed = false",
                spaces="drive",
                fields="files(name, id)",
            )
            .execute()["files"]
        )
        prefix = "" if root_path == self.root_path else root_path
        for file in file_list:
            path = posixpath.join(prefix, file["name"])
            self.path_id_map[path] = file["id"]
            self.makemap(file["id"], path)
        return self.path_id_map


class GDriveProvider(StorageProvider):
    """Provider class for using Google Drive storage."""

    client_id = None
    client_secret = None
    refresh_token = None

    def __init__(
        self, root: str, token: Optional[Union[str, Dict]] = None, makemap: bool = True
    ):
        """Initializes the GDriveProvider

        Example:

            >>> gdrive_provider = GDriveProvider("gdrive://folder_name/folder_name")

        Args:
            root(str): The root of the provider. All read/write request keys will be appended to root.
            token(dict, str, optional): Google Drive token. Can be path to the token file or the actual credentials dictionary.
            makemap(bool): Creates path to id map if ``True``.

        Note:
            - Requires ``client_secrets.json`` in working directory if ``token`` is not provided.
            - Due to limits on requests per 100 seconds on google drive api, continuous requests such as uploading many small files can be slow.
            - Users can request to increse their quotas on their google cloud platform.
        """

        creds = None

        if not token:
            token = "gdrive_token.json"

        if isinstance(token, str):
            if os.path.exists(token):
                creds = Credentials.from_authorized_user_file(token, SCOPES)
        elif isinstance(token, dict):
            creds = Credentials.from_authorized_user_info(token, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if os.path.exists("client_secrets.json"):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        "client_secrets.json", SCOPES
                    )
                else:
                    OAUTH_CLIENT_ID = os.getenv("CLIENT_ID")
                    OAUTH_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
                    flow = InstalledAppFlow.from_client_config(
                        {
                            "installed": {
                                "client_id": OAUTH_CLIENT_ID,
                                "client_secret": OAUTH_CLIENT_SECRET,
                                "redirect_uris": [
                                    "urn:ietf:wg:oauth:2.0:oob",
                                    "http://localhost",
                                ],
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                            }
                        },
                        scopes=SCOPES,
                    )
                creds = flow.run_local_server(port=0)

            with open("gdrive_token.json", "w") as token_file:
                token_file.write(creds.to_json())

        self.root = root
        if creds:
            self.client_id = creds.client_id
            self.client_secret = creds.client_secret
            self.refresh_token = creds.refresh_token

        self._init_provider(creds, makemap=makemap)

    def _init_provider(self, creds, makemap=True):
        self.drive = discovery.build("drive", "v3", credentials=creds)
        self.root_path = self.root.replace("gdrive://", "")
        if self.root_path == "":
            self.root_id = "root"
        if hasattr(self, "root_id"):
            self.gid = GoogleDriveIDManager(self.drive, self.root_path, self.root_id)  # type: ignore
        else:
            self.gid = GoogleDriveIDManager(self.drive, self.root_path)
            self.root_id = self.gid.root_id
        if self.root_id is None:
            self.root_id = "root"
            root_dir = self.make_dir(self.root_path, find=True)
            self.root_id = self.gid.root_id = root_dir.get("id")
            for i in range(len(self.root_path.split("/"))):
                self.gid.path_id_map.pop(
                    self.root_path.split("/", i)[0], None
                )  # Remove root dir components from map
        if makemap:
            self.gid.makemap(self.root_id, self.root_path)

    def _init_from_state(self):
        token = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
        }
        creds = Credentials.from_authorized_user_info(token, SCOPES)
        self._init_provider(creds)

    def _get_id(self, path):
        return self.gid.path_id_map.get(path)

    def _pop_id(self, path):
        return self.gid.path_id_map.pop(path)

    def _set_id(self, path, id):
        self.gid.path_id_map[path] = id

    def make_dir(self, path, find=False):
        dirname, basename = posixpath.split(path)
        if dirname:
            if find:
                parent_id = self.gid.find_id(dirname)
            else:
                parent_id = self._get_id(dirname)
            if not parent_id:
                locked = self._lock_creation(dirname)
                if locked:
                    self.make_dir(dirname)
                    self._unlock_creation(dirname)
                parent_id = self._get_id(dirname)
            folder = self._create_file(basename, FOLDER, parent_id)
        else:
            folder = self._create_file(basename, FOLDER, self.root_id)

        self._set_id(path, folder.get("id"))
        return folder

    def _create_file(self, name, mimeType, parent=None, content=None):
        file_metadata = {
            "name": name,
            "mimeType": mimeType,
            "parents": [parent if parent else self.root_id],
        }

        if not content:
            file = self.drive.files().create(body=file_metadata, fields="id").execute()
        else:
            content = MediaIoBaseUpload(BytesIO(content), mimeType)
            file = (
                self.drive.files()
                .create(body=file_metadata, media_body=content, fields="id")
                .execute()
            )

        return file

    def _write_to_file(self, id, content):
        content = MediaIoBaseUpload(BytesIO(content), FILE)
        file = self.drive.files().update(media_body=content, fileId=id).execute()
        return file

    def _delete_file(self, id):
        file = self.drive.files().delete(fileId=id).execute()
        return file

    def sync(self):
        """Sync provider keys with actual storage"""
        self.gid.makemap(self.root_id, self.root_path)

    def get_object_by_id(self, id):
        request = self.drive.files().get_media(fileId=id)
        file = BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        file.seek(0)
        return file.read()

    def __getitem__(self, path):
        id = self._get_id(path)
        if not id:
            raise KeyError(path)

        return self.get_object_by_id(id)

    def get_object_from_full_url(self, url: str):
        url = url.replace("gdrive://", "")
        id = self.gid.find_id(url)
        return self.get_object_by_id(id)

    def _lock_creation(self, path):
        # lock creation of folder, otherwise multiple workers can create folders of the same name.
        lock_hash = "." + hash_inputs(self.root_id, path)
        try:
            lock_file = open(lock_hash, "x")
            lock_file.close()
            return True
        except FileExistsError:
            while os.path.exists(lock_hash):
                time.sleep(0.1)
            return False

    def _unlock_creation(self, path):
        lock_hash = "." + hash_inputs(self.root_id, path)
        os.remove(lock_hash)

    def __setitem__(self, path, content):
        self.check_readonly()
        id = self._get_id(path)
        if not id:
            dirname, basename = posixpath.split(path)
            if dirname:
                parent_id = self._get_id(dirname)
                if not parent_id:
                    locked = self._lock_creation(dirname)
                    if locked:
                        self.make_dir(dirname)
                        self._unlock_creation(dirname)
                    self.sync()
                    parent_id = self._get_id(dirname)
            else:
                parent_id = self.root_id
            file = self._create_file(basename, FILE, parent_id, content)
            self._set_id(path, file.get("id"))
            return

        self._write_to_file(id, content)
        return

    def __delitem__(self, path):
        self.check_readonly()
        id = self._pop_id(path)
        if not id:
            raise KeyError(path)
        self._delete_file(id)

    def __getstate__(self):
        return (
            self.root,
            self.root_id,
            self.client_id,
            self.client_secret,
            self.refresh_token,
        )

    def __setstate__(self, state):
        self.root = state[0]
        self.root_id = state[1]
        self.client_id = state[2]
        self.client_secret = state[3]
        self.refresh_token = state[4]
        self._init_from_state()

    def _all_keys(self):
        keys = set(self.gid.path_id_map.keys())
        return keys

    def __iter__(self):
        yield from self._all_keys()

    def __len__(self):
        return len(self._all_keys())

    def clear(self, prefix=""):
        self.check_readonly()
        for key in self._all_keys():
            if key.startswith(prefix):
                try:
                    del self[key]
                except:
                    pass
        if not prefix:
            self._delete_file(self.root_id)
