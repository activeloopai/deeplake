from aifc import Error
from distutils.log import error
import hub
import time
import json
import logging
import requests

from requests.exceptions import HTTPError
from typing import Optional, Set, Sequence
from hub.core.storage.provider import StorageProvider


logger = logging.getLogger("ipfsspec")

class IPFSProvider(StorageProvider):
    def __init__(
        self,
        coreurl:str='', # Core URL to use
        cid:str='',
        storage_type:str=None, # specify type of gateway (e.g. Infura, Estuary, Web3.Storage, local node...)
        api_key:str=None, # if applicable, api key for access to storage service
    ) -> None:
        """Initialize the object, assign credentials if required."""
        super().__init__()
        self.coreurl = coreurl
        self.cid = cid
        self.links = self.get_links()
        self.gw = IPFSGateway(url=self.coreurl)
        self.storage_type = storage_type
        self.api_key = api_key

    def __getitem__(self, path, **kwargs):
        """Gets the object present at the path."""
        try:
            query = next(i for i in self.links if i['Name'] == path)
            cid = query['Hash']['/']
            res, content = self.gw.get(cid)
            if res.status_code == 200:
                return content
            else:
                raise HTTPError (parse_error_message(res))
        except:
            print(f"Path requested: {path}")

    def __setitem__(self, path, value):
        """Sets the object present at the path with the value"""
        res = self.gw.apipost("add", path)
        self.cids = [r['Hash'] for r in res]


    def __delitem__(self, path):
        """Delete the object present at the path."""
        # test = f"/pinning/pins/{path}"

        params = {
            'arg': path,
        }

        response = requests.post(f'{self.coreurl}/pin/rm', params=params)
        return response

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.cids

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.cids)

    def _all_keys(self) -> Set[str]:
        """Generator function that iterates over the keys of the provider.

        Returns:
            set: set of all keys present at the root of the provider.
        """
        return self.cids

    def clear(self, prefix=""):
        """Delete the contents of the provider."""
        if self.cids:
            self.cids = None
        else:
            super().clear()

    def dag_get(self):
        """Get a DAG node from IPFS."""
        
        params = {}
        params['arg'] = self.cid
        params['output-codec'] = 'dag-json'
        
        response = requests.post(f'{self.coreurl}/dag/get', params=params)
        
        if response.status_code == 200:
            return response, parse_response(response)

        else:
            raise HTTPError (parse_error_message(response))

    def get_links(self):
        if (self.coreurl is not None) and (self.cid is not None):
            res = self.dag_get()
            links = res[1]['Links']
            return links
        else:
            raise Error("Coreurl or CID not defined.")

def parse_error_message(
    response, # Response object from requests
    ):
    """Parse error message for raising exceptions"""

    sc = response.status_code

    try:
        message = response.json()['Message']

    except:
        message = response.text

    return f"Response Status Code: {sc}; Error Message: {message}"

def parse_response(
        response, # Response object
    ):
    """Parse response object into JSON"""

    if response.text.split('\n')[-1] == "":
        try:
            return [json.loads(each) for each in response.text.split('\n')[:-1]]

        except:
            pass

    try:
        return response.json()

    except:
        return response.text


class IPFSGateway():
    def __init__(self, url):
        self.url = url

        if self.url in ['http://127.0.0.1:5001', 'https://ipfs.infura.io:5001', 'https://ipfs.infura.io:5001/api/v0']: # hard coded because of limited number of post gateways
            self.reqtype = 'post' # local node and infura works on `post` request while public on `get`
        else:
            self.reqtype = 'get'

        self.state = "unknown"
        self.min_backoff = 1e-9
        self.max_backoff = 5
        self.backoff_time = 0
        self.next_request_time = time.monotonic()
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get(self, 
        cid:str, # Path to the IPFS object
        **kwargs
    ):
        'Get a file/directory from IPFS'
        
        params = {}
        params['arg'] = cid
        params.update(kwargs)
        
        res = self.session.post(f'{self.url}/get', params=params)
            
        if res.status_code == 200:
            return res, parse_response(res)
        
        else:
            raise HTTPError (parse_error_message(res))

    def head(self, path, headers=None):
        logger.debug("head %s via %s", path, self.url, headers=headers or {})
        try:
            res = self.session.get(self.url + "/ipfs/" + path)
        except requests.ConnectionError as e:
            logger.debug("Connection Error: %s", e)
            self._backoff()
            return None
        if res.status_code == 429:  # too many requests
            self._backoff()
            return None
        elif res.status_code == 200:
            self._speedup()
        res.raise_for_status()
        return res.headers

    def apipost(self, call, **kwargs):
        logger.debug("post %s via %s", call, self.url)
        if 'data' in kwargs.keys():
            data = kwargs.pop('data')
            headers = kwargs.pop('headers')
        else: data, headers = None, None            

        try:
            if data is not None:
                res = self.session.post(self.url + "/api/v0/" + call, params=kwargs, data=data, headers=headers)
            else:
                res = self.session.post(self.url + "/api/v0/" + call, params=kwargs)
        
        except requests.ConnectionError:
            self._backoff()
            return None
        
        if res.status_code == 429:  # too many requests
            self._backoff()
            return None
        
        elif res.status_code == 200:
            self._speedup()
        
#         res.raise_for_status() # moving the exception to filesystem level 
        return res

    def _schedule_next(self):
        self.next_request_time = time.monotonic() + self.backoff_time

    def _backoff(self):
        self.backoff_time = min(max(self.min_backoff, self.backoff_time) * 2,
                                self.max_backoff)
        logger.debug("%s: backing off -> %f sec", self.url, self.backoff_time)
        self._schedule_next()

    def _speedup(self):
        self.backoff_time = max(self.min_backoff, self.backoff_time * 0.9)
        logger.debug("%s: speeding up -> %f sec", self.url, self.backoff_time)
        self._schedule_next()

    def _init_state(self):
        try:
            if self.reqtype == 'get':
                res = self.session.get(self.url + "/api/v0/version")
            else:
                res = self.session.post(self.url + "/api/v0/version")      
            if res.ok:
                self.state = "online"
            else:
                self.state = "offline"
        except requests.ConnectionError:
            self.state = "offline"

    def get_state(self):
        if self.state == "unknown":
            self._init_state()
        now = time.monotonic()
        if self.next_request_time > now:
            return ("backoff", self.next_request_time - now)
        else:
            return (self.state, None)


