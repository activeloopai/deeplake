# Streams video from a chunk

from typing import Optional
from flask import Flask, request, Response
from numpy import disp
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core import dataset
from hub.util.threading import terminate_thread
import logging
import re
import socketserver
import threading

from IPython.display import IFrame, display


_PORT: Optional[int] = None
_SERVER_THREAD: Optional[threading.Thread] = None

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

ds = None

def visualize(dataset: dataset):
    global ds
    ds = dataset
    if _is_server_running:
        _stop_server()
    url = _start_server()
    print(f"Using PORT {_PORT}")
    display(IFrame(f"http://localhost:3000/iframe?url={url}", width="100%", height=600))


def _get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def _is_server_running() -> bool:
    return _SERVER_THREAD and _SERVER_THREAD.is_alive()


def _run_app():
    try:
        _APP.run(host="0.0.0.0", port=_PORT, threaded=True)
    except Exception:  # thread termination
        pass

def _start_server():
    global _PORT
    global _SERVER_THREAD
    if _is_server_running():
        return
    _PORT = _get_free_port()
    _SERVER_THREAD = threading.Thread(target=_run_app, daemon=True)
    _SERVER_THREAD.start()
    return f"http://localhost:{_PORT}/"

def _stop_server():
    global _SERVER_THREAD
    if not _is_server_running():
        return
    terminate_thread(_SERVER_THREAD)
    _SERVER_THREAD = None


_APP = Flask("dataset_visualizer")

@_APP.after_request
def after_request(response):
    response.headers.add("Accept-Ranges", "bytes")
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    return response

@_APP.route('/<path:path>')
def access_data(path):
    try:
        range_header = request.headers.get("Range", None)
        start, end = 0, None
        if range_header:
            match = re.search(r"(\d+)-(\d*)", range_header)
            groups = match.groups()

            if groups[0]:
                start = int(groups[0])
            if groups[1]:
                end = int(groups[1])

        c = ds.storage[path]
        if isinstance(c, HubMemoryObject):
            c = c.tobytes()
        if start == None:
            start = 0
        if end == None:
            end = len(c) - 1

        resp = Response(
            c[start:end + 1],
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