from hub.core.storage.provider import StorageProvider
from googleapiclient import discovery
from googleapiclient.http import (  # type: ignore
    MediaIoBaseDownload,
    MediaIoBaseUpload,
)
from httplib2 import Http  # type: ignore
from oauth2client import file, client, tools  # type: ignore
from io import BytesIO
import posixpath
import pickle
from typing import Dict

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.install",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]
FOLDER = "application/vnd.google-apps.folder"
FILE = "application/octet-stream"


class GoogleDriveIDManager:
    """Class used to make google drive path to id maps"""

    def __init__(self, drive: discovery.Resource, root: str):
        self.path_id_map: Dict[str, str] = {}
        self.drive = drive
        self.root_path = root
        self.root_id = self.find_id(root)
        self.makemap(self.root_id, self.root_path)

    def find_id(self, path):
        """Find google drive id given path of folder"""

        dirname, basename = posixpath.split(path)
        file_list = (
            self.drive.files()
            .list(
                q=f"""'{self.find_id(dirname, update_map=False) if dirname else 'root'}' in parents and 
                    name = '{basename}' and 
                    trashed = false""",
                spaces="drive",
                fields="files(id)",
            )
            .execute()
        )
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
        for file in file_list:
            prefix = "" if root_path == self.root_path else root_path
            path = posixpath.join(prefix, file["name"])
            self.path_id_map[path] = file["id"]
            self.makemap(file["id"], path)
        return self.path_id_map

    def save(self):
        with open(f".{self.root_id}_gdrive_ids", "wb") as f:
            pickle.dump(self.path_id_map, f)

    def load(self):
        try:
            with open(f".{self.root_id}_gdrive_ids", "rb") as f:
                self.path_id_map = pickle.load(f)
        except FileNotFoundError:
            pass


class GDriveProvider(StorageProvider):
    """Provider class for using Google Drive storage."""

    def __init__(self, root: str = "root"):
        """Initializes the GDriveProvider

        Example:
            gdrive_provider = GDriveProvider("gdrive://folder_name/folder_name")

        Args:
            root(str): The root of the provider. All read/write request keys will be appended to root.

        Note:
            Requires `client_secrets.json` in working directory
        """

        store = file.Storage("gdrive_creds.json")
        creds = store.get()

        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets("client_secrets.json", SCOPES)
            creds = tools.run_flow(flow, store)

        self.drive = discovery.build("drive", "v3", http=creds.authorize(Http()))
        self.root = root
        if root.startswith("gdrive://"):
            root = root.replace("gdrive://", "")
            self.root_path = root
            self.gid = GoogleDriveIDManager(self.drive, self.root_path)
        self.root_id = self.gid.root_id
        if not self.root_id:
            root_dir = self.make_dir(self.root_path)
            self.root_id = self.gid.root_id = root_dir.get("id")
            self.gid.path_id_map.pop(self.root_path)

    def _get_id(self, path):
        return self.gid.path_id_map.get(path)

    def _set_id(self, path, id):
        self.gid.path_id_map[path] = id

    def make_dir(self, path):
        dirname, basename = posixpath.split(path)
        if dirname:
            parent_id = self._get_id(dirname)
            if not parent_id:
                self.make_dir(dirname)
                parent_id = self._get_id(dirname)
            folder = self._create_file(basename, FOLDER, parent_id)
            self._set_id(path, folder.get("id"))
            return folder

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

    def __getitem__(self, path):
        id = self._get_id(path)
        if not id:
            raise KeyError(path)

        request = self.drive.files().get_media(fileId=id)
        file = BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        file.seek(0)
        return file.read()

    def __setitem__(self, path, content):
        self.check_readonly()
        id = self._get_id(path)
        if not id:
            dirname, basename = posixpath.split(path)
            if dirname:
                parent_id = self._get_id(dirname)
                if not parent_id:
                    self.make_dir(dirname)
                    parent_id = self._get_id(dirname)
            else:
                parent_id = self.root_id
            file = self._create_file(basename, FILE, parent_id, content)
            self._set_id(path, file.get("id"))
            return

        self._write_to_file(id, content)
        return

    def __delitem__(self, path):
        id = self._get_id(path)
        if not id:
            raise KeyError(path)
        self._delete_file(id)

    def _all_keys(self):
        keys = set(self.gid.path_id_map.keys())
        return keys

    def __iter__(self):
        yield from self._all_keys()

    def __len__(self):
        return len(self._all_keys())

    def clear(self):
        for key in self._all_keys():
            try:
                del self[key]
            except:
                pass
