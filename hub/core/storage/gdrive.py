from hub.core.storage.provider import StorageProvider
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from hub.core.storage.provider import StorageProvider
from io import BytesIO
import posixpath
import pickle
from googleapiclient.errors import HttpError
import os

CREDS_FILE = ".gdrive_creds"


class GDriveIDManager:
    def __init__(self, drive: GoogleDrive, root):
        self.path_id_map = {}
        self.drive = drive
        self.root_fname = root

    def makemap(self, root_id, root_fname):
        try:
            file_list = self.drive.ListFile(
                {"q": f"'{root_id}' in parents and trashed = false"}
            ).GetList()
        except HttpError:
            file_list = []

        self.path_id_map.update(
            {
                posixpath.join(
                    f"{'' if root_fname == self.root_fname else root_fname}",
                    file["title"],
                ): file["id"]
                for file in file_list
            }
        )
        for file in file_list:
            self.makemap(
                file["id"],
                posixpath.join(
                    f"{'' if root_fname == self.root_fname else root_fname}",
                    file["title"],
                ),
            )
        return self.path_id_map

    def save(self):
        with open(".gdrive_ids", "wb") as f:
            pickle.dump(self.path_id_map, f)

    def load(self):
        try:
            with open(".gdrive_ids", "rb") as f:
                self.path_id_map = pickle.load(f)
        except FileNotFoundError:
            pass


class GDriveProvider(StorageProvider):
    def __init__(self, root="root"):
        self.gauth = GoogleAuth()

        try:
            self.gauth.LoadCredentialsFile(CREDS_FILE)
        except:
            pass

        if self.gauth.credentials is None:
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            self.gauth.Refresh()
        else:
            self.gauth.Authorize()

        self.gauth.SaveCredentialsFile(CREDS_FILE)
        self.drive = GoogleDrive(self.gauth)
        self.root = root
        if root.startswith("gdrive://"):
            root = root.replace("gdrive://", "")
            root = posixpath.split(root)[-1]
        self.root_id = root
        self.root_fname = self.drive.CreateFile({"id": self.root_id})["title"]
        self.gid = GDriveIDManager(self.drive, self.root_fname)
        self.gid.makemap(self.root_id, self.root_fname)

    def _get_id(self, path):
        return self.gid.path_id_map.get(path)

    def _set_id(self, path, id):
        self.gid.path_id_map[path] = id

    def __setitem__(self, path, content):
        self.check_readonly()
        id = self._get_id(path)
        if not id:
            dirname, basename = posixpath.split(path)
            if dirname:
                dir_id = self._get_id(dirname)
                if not dir_id:
                    self.make_dir(dirname)
                    dir_id = self._get_id(dirname)
            else:
                dir_id = self.root_id

            file = self.drive.CreateFile(
                {
                    "title": basename,
                    "parents": [{"id": dir_id}],
                }
            )
            file.content = BytesIO(content)
            file.Upload()
            self._set_id(path, file["id"])
            return

        file = self.drive.CreateFile({"id": id})
        file.content = BytesIO(content)
        file.Upload()
        return

    def make_dir(self, path):
        dirname, basename = posixpath.split(path)
        if dirname:
            id = self._get_id(dirname)
            if not id:
                self.make_dir(dirname)
                id = self._get_id(dirname)
            folder = self.drive.CreateFile(
                {
                    "title": basename,
                    "parents": [{"id": id}],
                    "mimeType": "application/vnd.google-apps.folder",
                }
            )
            folder.Upload()
            self._set_id(path, folder["id"])
            return

        folder = self.drive.CreateFile(
            {
                "title": basename,
                "parents": [{"id": self.root_id}],
                "mimeType": "application/vnd.google-apps.folder",
            }
        )
        folder.Upload()
        self._set_id(basename, folder["id"])
        return

    def __getitem__(self, path):
        if self._get_id(path) is None:
            raise KeyError(path)
        file = self.drive.CreateFile({"id": self._get_id(path)})
        file.FetchContent()
        return file.content.getvalue()

    def __delitem__(self, path):
        self.check_readonly()
        file = self.drive.CreateFile({"id": self._get_id(path)})
        file.Delete()
        self.gid.path_id_map.pop(path)

    def _all_keys(self):
        keys = set(self.gid.path_id_map.keys())
        return keys

    def __iter__(self):
        yield from self._all_keys()

    def __len__(self):
        return len(self._all_keys())

    def refresh_ids(self):
        self.gid.makemap(self.root_id, self.root_fname)

    def clear(self):
        for key in self._all_keys:
            del self[key]
