import hub
from hub.util.threading import terminate_thread
import socketserver
from typing import Optional, Callable, Dict
import inspect
import threading
import queue
import multiprocessing.connection
import atexit
import time


def _get_free_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]


class Server(object):
    def __init__(self, callback):
        self.callback = callback
        self.start()
        atexit.register(self.stop)

    def start(self):
        if getattr(self, "_connect_thread", None):
            return
        self.port = _get_free_port()
        self._listener = multiprocessing.connection.Listener(("localhost", self.port))
        self._connections = []
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()

    def _connect_loop(self):
        try:
            while True:
                try:
                    connection = self._listener.accept()
                    thread = threading.Thread(
                        target=self._receive_loop, args=(connection,)
                    )
                    self._connections.append((connection, thread))
                    thread.start()
                except Exception:
                    time.sleep(0.1)
        except Exception:
            pass  # Thread termination

    def _receive_loop(self, connection):
        try:
            while True:
                self.callback(connection.recv())
        except Exception:
            pass  # Thread termination

    def stop(self):
        if self._connect_thread:
            terminate_thread(self._connect_thread)
            self._connect_thread = None
        while self._connections:
            connection, thread = self._connections.pop()
            terminate_thread(thread)
            connection.close()
        self._listener.close()

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"


class Client(object):
    def __init__(self, port):
        self.port = port
        self._buffer = []
        self._client = None
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        while True:
            try:
                self._client = multiprocessing.connection.Client(
                    ("localhost", self.port)
                )
                for stuff in self._buffer:
                    self._client.send(stuff)
                self._buffer.clear()
                return
            except Exception:
                time.sleep(1)

    def send(self, stuff):
        if self._client:
            self._client.send(stuff)
        else:
            self._buffer.append(stuff)
