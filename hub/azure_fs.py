from fsspec import AbstractFileSystem
import array
from collections.abc import MutableMapping

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    pass


class AzureBlobFileSystem(AbstractFileSystem):
    def __init__(
        self,
        account_name: str,
        account_key: str = None,
        connection_string: str = None,
        credential: str = None,
        sas_token: str = None,
        request_session=None,
        socket_timeout: int = None,
        client_id: str = None,
        client_secret: str = None,
        tenant_id: str = None,
    ):

        super().__init__()
        self.account_name = account_name
        self.account_key = account_key
        self.connection_string = connection_string
        self.credential = credential
        self.sas_token = sas_token
        self.request_session = request_session
        self.socket_timeout = socket_timeout
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        if (
            self.credential is None
            and self.account_key is None
            and self.sas_token is None
            and self.client_id is not None
        ):
            self.credential = self._get_credential_from_service_principal()
        self.do_connect()

    def _get_credential_from_service_principal(self):
        """
        Create a Credential for authentication.  This can include a TokenCredential
        client_id, client_secret and tenant_id

        Returns
        -------
        Credential
        """
        from azure.identity import ClientSecretCredential

        sp_token = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        return sp_token

    def do_connect(self):
        """Connect to the BlobServiceClient, using user-specified connection details.
        Tries credentials first, then connection string and finally account key

        Raises
        ------
        ValueError if none of the connection details are available
        """
        try:
            self.account_url: str = f"https://{self.account_name}.blob.core.windows.net"
            if self.credential is not None:
                self.service_client = BlobServiceClient(
                    account_url=self.account_url, credential=self.credential
                )
            elif self.connection_string is not None:
                self.service_client = BlobServiceClient.from_connection_string(
                    conn_str=self.connection_string
                )
            elif self.account_key is not None:
                self.service_client = BlobServiceClient(
                    account_url=self.account_url, credential=self.account_key
                )
            elif self.sas_token is not None:
                self.service_client = BlobServiceClient(
                    account_url=self.account_url + self.sas_token, credential=None
                )
            else:
                self.service_client = BlobServiceClient(account_url=self.account_url)

        except Exception as e:
            raise ValueError(f"unable to connect to account for {e}")

    def exists(self, path):
        spl = path.split('/')
        rest = "/".join(spl[1:])
        container = spl[0]
        container = self.service_client.get_container_client(container)
        it = container.list_blobs(name_starts_with=rest)
        return len(list(it)) > 0

    def find(self, path):
        spl = path.split('/')
        rest = "/".join(spl[1:])
        container = spl[0]
        container = self.service_client.get_container_client(container)
        it = container.list_blobs(name_starts_with=rest)
        return [item["name"] for item in it]

    def rm(self, path, recursive=False, maxdepth=None):
        spl = path.split('/')
        rest = "/".join(spl[1:])
        container = spl[0]
        container = self.service_client.get_container_client(container)
        it = container.list_blobs(name_starts_with=rest)
        for item in it:
            container.delete_blob(item)

    def makedirs(self, path, exist_ok=False):
        # in azure empty directories have no meaning, so makedirs not needed
        return

    def get_mapper(self, root, check=False, create=False):
        return FSMap(root, self)

    def upload(self, path, value):
        spl = path.split('/')
        rest = "/".join(spl[1:])
        container = spl[0]
        bc = self.service_client.get_blob_client(container, rest)
        bc.upload_blob(value, overwrite=True)

    def download(self, path):
        spl = path.split('/')
        rest = "/".join(spl[1:])
        container = spl[0]
        bc = self.service_client.get_blob_client(container, rest)
        return bc.download_blob().readall()


class FSMap(MutableMapping):
    def __init__(self, root, fs, check=False, create=False, missing_exceptions=None):
        self.fs = fs
        self.root = root
        if create:
            if not self.fs.exists(root):
                self.fs.mkdir(root)
        if check and not create:
            if not self.fs.exists(root):
                raise ValueError(
                    "Path %s does not exist. Create "
                    " with the ``create=True`` keyword" % root
                )

    def clear(self):
        """Remove all keys below root - empties out mapping"""
        self.fs.rm(self.root, True)

    def _key_to_str(self, key):
        """Generate full path for the key"""
        if isinstance(key, (tuple, list)):
            key = str(tuple(key))
        else:
            key = str(key)
        return "/".join([self.root, key]) if self.root else key

    def _str_to_key(self, s):
        """Strip path of to leave key name"""
        return s[len(self.root) :].lstrip("/")

    def __getitem__(self, key, default=None):
        """Retrieve data"""
        k = self._key_to_str(key)
        try:
            result = self.fs.download(k)
        except self.missing_exceptions:
            if default is not None:
                return default
            raise KeyError(key)
        return result

    def __setitem__(self, key, value):
        """Store value in key"""
        key = self._key_to_str(key)
        if isinstance(value, array.array):
            value = bytearray(value)
        self.fs.upload(key, value)

    def __iter__(self):
        return (self._str_to_key(x) for x in self.fs.find(self.root))

    def __len__(self):
        return len(self.fs.find(self.root))

    def __delitem__(self, key):
        """Remove key"""
        self.fs.rm(self._key_to_str(key))

    def __contains__(self, key):
        """Does key exist in mapping?"""
        path = self._key_to_str(key)
        return self.fs.exists(path)
