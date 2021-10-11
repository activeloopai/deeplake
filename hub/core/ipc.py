import hub
from hub.util.threading import terminate_thread
import socketserver
from typing import Optional, Callable, Dict
import inspect
import threading
import queue
import requests
import responder
import atexit


def _get_free_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]


class Server(object):
    def __init__(self, callback):
        self.port = _get_free_port()
        self.callback = callback
        self._args = inspect.getargspec(callback).args
        self.api = responder.API()
        self.api.route("/")(self._respond)
        atexit.register(self.stop)
        self.start()

    def _loop(self):
        try:
            self.api.run(port=self.port)
        except Exception:  # Thread termination
            pass

    async def _respond(self, req, resp, *args):
        self.callback(**{k: req.params.get(k) for k in self._args})
        resp.status_code = 200

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread:
            terminate_thread(self._thread)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"


class Client(object):
    def __init__(self, port):
        self.port = port
        self._queue = queue.Queue()
        self._url = f"http://localhost:{port}/"

    def send(self, stuff: Dict):
        threading.Thread(
            target=requests.get,
            args=(self._url,),
            kwargs={"params": stuff},
            daemon=True,
        ).start()
