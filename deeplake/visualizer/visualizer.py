import json
from typing import Dict, Optional, Union
import uuid
from flask import Flask, request, Response
from deeplake.core.link_creds import LinkCreds  # type: ignore
from deeplake.core.storage.provider import StorageProvider
from deeplake.util.threading import terminate_thread
from deeplake.client.config import (
    USE_DEV_ENVIRONMENT,
    USE_STAGING_ENVIRONMENT,
    USE_LOCAL_HOST,
)
import logging
import re
import socketserver
import threading

from IPython.display import IFrame, display  # type: ignore

_SERVER_THREAD: Optional[threading.Thread] = None
_APP = Flask("dataset_visualizer")

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def _run_app(port: int):
    try:
        _APP.run(host="0.0.0.0", port=port, threaded=True)
    except Exception:
        pass


@_APP.after_request
def after_request(response):
    response.headers.add("Accept-Ranges", "bytes")
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    return response


class _Visualizer:
    """
    Visualizer class to manage visualization of the datasets.
    """

    _port: Optional[int] = None
    _storages: Dict = {}
    _link_creds_storage: Dict = {}

    def __init__(self):
        self.start_server()
        self._storages = {}

    def add(self, storage: StorageProvider) -> str:
        id = str(uuid.uuid4())
        self._storages[id] = storage
        return id

    def get(self, id: str) -> StorageProvider:
        return self._storages[id]

    def add_link_creds(self, link_creds: LinkCreds):
        id = str(uuid.uuid4())
        self._link_creds_storage[id] = link_creds
        return id

    def get_link_creds(self, id: str) -> LinkCreds:
        return self._link_creds_storage[id]

    @property
    def port(self):
        return self._port

    def get_free_port(self):
        with socketserver.TCPServer(("localhost", 0), None) as s:
            return s.server_address[1]

    def is_server_running(self) -> bool:
        return (_SERVER_THREAD is not None) and _SERVER_THREAD.is_alive()

    def start_server(self):
        global _SERVER_THREAD
        if self.is_server_running():
            return f"http://localhost:{self.port}/"
        self._port = self.get_free_port()

        def run_app():
            _run_app(port=self.port)

        _SERVER_THREAD = threading.Thread(target=run_app, daemon=True)
        _SERVER_THREAD.start()
        return f"http://localhost:{self.port}/"

    def stop_server(self):
        global _SERVER_THREAD
        if not self.is_server_running():
            return
        terminate_thread(_SERVER_THREAD)
        _SERVER_THREAD = None

    def __del__(self):
        self.stop_server()


visualizer = _Visualizer()


def _get_visualizer_backend_url():
    if USE_LOCAL_HOST:
        return "http://localhost:3000"
    elif USE_DEV_ENVIRONMENT:
        return "https://app-dev.activeloop.dev"
    elif USE_STAGING_ENVIRONMENT:
        return "https://app-staging.activeloop.dev"
    else:
        return "https://app.activeloop.ai"


def visualize(
    source: Union[StorageProvider, str],
    link_creds: Union[LinkCreds, None] = None,
    token: Union[str, None] = None,
    creds: Union[dict, None] = None,
    width: Union[int, str, None] = None,
    height: Union[int, str, None] = None,
):
    """
    Visualizes the given dataset in the Jupyter notebook.

    Args:
        source: Union[StorageProvider, str] The storage or the path of the dataset.
        link_creds: Union[LinkCreds, None] The link creds to serve visualizer frontend.
        token: Union[str, None] Optional token to use in the backend call.
        creds: Union[dict, None] Optional credentials dictionary.
        width: Union[int, str, None] Optional width of the visualizer canvas.
        height: Union[int, str, None] Optional height of the visualizer canvas.
    """
    if isinstance(source, StorageProvider):
        id = visualizer.add(source)
        params = f"url=http://localhost:{visualizer.port}/{id}/"
    else:
        params = f"url={source}"

    if token is not None:
        params += f"&token={token}"

    if creds is not None:
        params += f"&creds={json.dumps(creds)}"

    if link_creds is not None:
        link_creds_id = visualizer.add_link_creds(link_creds)
        params += (
            f"&link_creds_url=http://localhost:{visualizer.port}/creds/{link_creds_id}/"
        )

    iframe = IFrame(
        f"{_get_visualizer_backend_url()}/visualizer/hub?{params}",
        width=width or "90%",
        height=height or 800,
    )
    display(iframe)


@_APP.route("/creds/<path:path>")
def access_creds(path: str):
    paths = path.split("/", 1)
    id = paths[0]
    creds_key = paths[1]
    if creds_key in visualizer.get_link_creds(id).creds_keys:
        return visualizer.get_link_creds(id).get_creds(creds_key)

    return Response("", 404)


@_APP.route("/<path:path>")
def access_data(path):
    try:
        paths = path.split("/", 1)
        range_header = request.headers.get("Range", None)
        start, end = 0, None
        storage: StorageProvider = visualizer.get(paths[0])
        if request.method == "HEAD":
            if paths[1] in storage.keys():
                return Response("OK", 200)
            else:
                return Response("", 404)
        if range_header:
            match = re.search(r"(\d+)-(\d*)", range_header)
            groups = match.groups()

            if groups[0]:
                start = int(groups[0])
            if groups[1]:
                end = int(groups[1]) + 1

        c = storage.get_bytes(paths[1], start, end)
        if isinstance(c, memoryview):
            c = c.tobytes()
        resp = Response(
            c,
            206,
            content_type="application/octet-stream",
        )
        resp.headers.add("Connection", "keep-alive")
        resp.headers.add("Accept-Ranges", "bytes")
        resp.headers.add("Content-Range", "bytes {0}-{1}".format(start, end))
        return resp

    except Exception as e:
        return Response(
            "Not Found",
            404,
            content_type="application/octet-stream",
        )
