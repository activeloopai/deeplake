import deeplake as dp
from deeplake.client.client import DeepLakeBackendClient


class CfgObject:
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        self._init_credentials()
        self._init_token()
        self._init_uname()
        
    def _init_credentials(self):
        self.creds = self.cfg.get("deeplake_credentials", {})
        
    def _init_token(self):
        self.token = self.creds.get("token", None)
        
    def _init_uname(self):
        if self.token is None:
            uname = self.creds.get("username")
            if uname is not None:
                pword = self.creds["password"]
                client = DeepLakeBackendClient()
                self.token = client.request_auth_token(username=uname, password=pword)
    
    def load(self):
        ds_path = self.cfg.deeplake_path
        ds = dp.load(ds_path, token=self.token, read_only=True)
        self.load_commit(ds)
        return self.load_view_or_query(ds)
    
    def load_commit(self, ds):
        deeplake_commit = self.cfg.get("deeplake_commit")
        if deeplake_commit:
            ds.checkout(deeplake_commit)
            
    def load_view_or_query(self, ds):
        deeplake_view_id = self.cfg.get("deeplake_view_id")
        deeplake_query = self.cfg.get("deeplake_query")
        
        if deeplake_view_id and deeplake_query:
            raise Exception(
                "A query and view_id were specified simultaneously for a dataset in the config. Please specify either the deeplake_query or the deeplake_view_id."
            )

        if deeplake_view_id:
            return ds.load_view(id=deeplake_view_id)
        return ds.query(deeplake_query)