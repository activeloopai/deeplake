from typing import Dict, Optional, Union
import uuid
from flask import Flask, request, Response  # type: ignore
from hub.core.storage.provider import StorageProvider
from hub.util.threading import terminate_thread
from hub.client.config import (
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

    def __init__(self):
        self.start_server()
        self._storages = {}

    def add(self, storage: StorageProvider) -> str:
        id = str(uuid.uuid4())
        self._storages[id] = storage
        return id

    def get(self, id: str) -> StorageProvider:
        return self._storages[id]

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
    token: Union[str, None] = None,
    width: Union[int, str, None] = None,
    height: Union[int, str, None] = None,
):
    """
    Visualizes the given dataset in the Jupyter notebook.

    Args:
        source: Union[StorageProvider, str] The storage or the path of the dataset.
        token: Union[str, None] Optional token to use in the backend call.
        width: Union[int, str, None] Optional width of the visualizer canvas.
        height: Union[int, str, None] Optional height of the visualizer canvas.
    """
    if isinstance(source, StorageProvider):
        id = visualizer.add(source)
        params = f"url=http://localhost:{visualizer.port}/{id}/"
    elif token is None:
        params = f"url={source}"
    else:
        params = f"url={source}&token={token}"
    iframe = IFrame(
        f"{_get_visualizer_backend_url()}/visualizer/hub?{params}",
        width=width or "90%",
        height=height or 800,
    )
    display(iframe)


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
