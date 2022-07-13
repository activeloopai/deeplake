from aifc import Error
from distutils.log import error
import hub
import time
import json
import logging
import dag_cbor
import requests
import pandas as pd

from io import StringIO,BytesIO
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
        self.links = self._get_links(cid)
        self.ordered_links = self.get_hash(self.links, {})
        self.gw = IPFSGateway(url=self.coreurl)
        self.storage_type = storage_type
        self.api_key = api_key

    def __getitem__(self, path, **kwargs):
        """Gets the object present at the path."""
        try:
            query = next(i for i in self.links if i['Name'] == path)
            cid = query['Hash']['/']
            # res, content = self.gw.get(cid)
            res, content = self.gw.cat(cid)
            # content = self.read_json(cid)
            # assert 1==0, f'Content is {content}'
            # cont = pd.read_json(StringIO(content))
            b = bytes(content,'utf-8')
            if res.status_code == 200:                
                return b
            else:
                assert 1==0
                # raise HTTPError (parse_error_message(res))
        except StopIteration:
            print(f"Path requested: {path}")
        except FileNotFoundError:
            print(f"Path requested: {path}, filenotfound")
            raise KeyError(path)
        except AssertionError:
            print(f'Status code: {res.status_code}, Assertion error, content is {content}, path is {path}, cid query is {cid}, query is {query}, and in bytes we have {b}')
        except TypeError:
            print('Got type error.')
        # try:
        #     full_path = self._check_is_file(path)
        #     with open(full_path, "rb") as file:
        #         return file.read()
        # except DirectoryAtPathException:
        #     raise
        # except FileNotFoundError:
        #     raise KeyError(path)

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

    def dag_get(self, cid=None, url=None):
        """Get a DAG node from IPFS."""
        if (cid is not None) and (url is not None):
            params = {}
            params['arg'] = cid
            params['output-codec'] = 'dag-json'
            
            response = requests.post(f'{url}/dag/get', params=params)
            
            if response.status_code == 200:
                return response, parse_response(response)

            else:
                raise HTTPError (parse_error_message(response))
        else:
            params = {}
            params['arg'] = self.cid
            params['output-codec'] = 'dag-json'
            
            response = requests.post(f'{self.coreurl}/dag/get', params=params)
            
            if response.status_code == 200:
                return response, parse_response(response)

            else:
                raise HTTPError (parse_error_message(response))

    def read_json(self, 
        cid:str, 
    ):
        r, data = self.gw.cat(cid)      
        
        return pd.read_json(StringIO(data))

    def _file_or_dir(self,
        name
        ):
        return 'file' if len(name.split('.')) > 1 else 'dir'

    def _get_links(self,
        cid,
        fol=''
    ):
        root_struct = {}
        struct = {}
        _, content = self.dag_get(url=self.coreurl, cid=cid)
        links = content['Links']

        for link in links:
            name = f'{fol}/{link["Name"]}'
            hash_ = str(link['Hash']['/'])
            type_ = self._file_or_dir(name)

            if type_ == 'dir':
                details = self._get_links(hash_, name)

            else:
                details = {'Hash': hash_, 'type': type_}

            struct[name] = details

        root_struct[fol] = struct

        return root_struct

    def get_hash(self, links_dict, ordered_links):
        for link in links_dict:
            if 'Hash' in links_dict[link].keys():
                cid = links_dict[link]['Hash']
                ordered_links[link[1:]] = cid
            else:
                ordered_links = self.get_hash(links_dict[link], ordered_links)
        return ordered_links


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

    def cat(self, 
        cid:str, # Path to the IPFS object
        **kwargs
    ):
        'Read a file from IPFS'
        
        params = {}
        params['arg'] = cid
        params.update(kwargs)
        
        res = self.session.post(f'{self.url}/cat', params=params)
        
        if res.status_code == 200:
            return res, res.text
        
        else:
            if res.status_code == 500:
                raise TypeError (f"dag node {cid} is a directory; Provide a file CID")
            else:
                raise HTTPError (parse_error_message(res))


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


class DownloadDir:
    'Download a IPFS directory to your local disk'
    def __init__(self,
        coreurl:str,
        root_cid:str, # Root CID of the directory
        output_fol:str, # Path to save in your local disk
    ):

        self.url = coreurl
        self.root = root_cid
        self.output = output_fol
        self.full_structure = None

    


    def _save_links(self,
        links
    ):
        for k, v in links.items():
            if len(k.split('.')) < 2:
                if not os.path.exists(k): os.mkdir(k)
                self._save_links(v)

            else:
                data = cat_items(self.url, links[k]['Hash']).content

                with open(k, 'wb') as f:
                    f.write(data)

    def download(self
    ):
        self.full_structure = self._get_links(self.root, self.output)
        self._save_links(self.full_structure)