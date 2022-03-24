from typing import Optional
import uuid
from flask import Flask, request, Response
from numpy import disp
from hub.core.dataset.dataset import Dataset
from hub.core import dataset
from hub.core.storage.provider import StorageProvider
from hub.util.threading import terminate_thread
import logging
import re
import socketserver
import threading

from IPython.display import IFrame, display

from hub.visualizer.visual_context import VisualContext

visualizer = None

_PORT: Optional[int] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_APP = Flask("dataset_visualizer")

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

def _run_app():
    try:
        _APP.run(host="0.0.0.0", port=_PORT, threaded=True)
    except Exception:
        pass

@_APP.after_request
def after_request(response):
    response.headers.add("Accept-Ranges", "bytes")
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    return response

@_APP.route('/<path:path>')
def access_data(path):
    try:
        paths = path.split('/', 1)
        range_header = request.headers.get("Range", None)
        start, end = 0, None
        ds: Dataset = visualizer.get(paths[0])
        storage: StorageProvider = ds.storage
        if request.method == "HEAD":
            if paths[1] in storage.keys:
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

class _Visualizer:
    """
    Visualizer class to manage visualization of the datasets.
    """

    def __init__(self):
        self.start_server()
        self._datasets = {}

    def add(self, ds: dataset) -> str:
        id = str(uuid.uuid4())
        self._datasets[id] = ds
        return id

    def get(self, id: str) -> dataset:
        return self._datasets[id]

    def get_free_port(self):
        with socketserver.TCPServer(("localhost", 0), None) as s:
            return s.server_address[1]


    def is_server_running(self) -> bool:
        return _SERVER_THREAD and _SERVER_THREAD.is_alive()

    def start_server(self):
        global _PORT
        global _SERVER_THREAD
        if self.is_server_running():
            return
        _PORT = self.get_free_port()
        _SERVER_THREAD = threading.Thread(target=_run_app, daemon=True)
        _SERVER_THREAD.start()
        return f"http://localhost:{_PORT}/"

    def stop_server(self):
        global _SERVER_THREAD
        if not self.is_server_running():
            return
        terminate_thread(_SERVER_THREAD)
        _SERVER_THREAD = None

visualizer = _Visualizer()

def visualize(ds: dataset):
    """
    Visualizes the given dataset in the Jupyter notebook and returns the corresponding context.

    Args:
        ds: dataset The dataset to visualize.
    
    Returns:
        VisualContext: The corresponding context to use later to interact with the visualizer.
    """
    global visualizer
    id = visualizer.add(ds)
    url = f"http://localhost:{_PORT}/{id}/"
    iframe = IFrame(f"https://app.dev.activeloop.ai/visualizer/hub?url={url}", width="100%", height=600)
    display(iframe)
    return VisualContext(id, ds, iframe)
