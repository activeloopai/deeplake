from time import sleep
from uuid import uuid4
import threading
import multiprocessing.connection
from socketserver import TCPServer
from atexit import register as atexit_register
from typing import Optional, Callable, Dict

from hub.util.threading import terminate_thread


_DISCONNECT_MESSAGE = "!@_dIsCoNNect"


def _get_free_port() -> int:
    with TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]


class Server(object):
    def __init__(self, callback):
        self.callback = callback
        self.start()
        atexit_register(self.stop)

    def start(self):
        if getattr(self, "_connect_thread", None):
            return
        self.port = _get_free_port()
        self._listener = multiprocessing.connection.Listener(("localhost", self.port))
        self._connections = {}
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()

    def _connect_loop(self):
        try:
            while True:
                try:
                    connection = self._listener.accept()
                    key = str(uuid4())
                    thread = threading.Thread(target=self._receive_loop, args=(key,))
                    self._connections[key] = (connection, thread)
                    thread.start()
                except Exception:
                    sleep(0.1)
        except Exception:
            pass  # Thread termination

    def _receive_loop(self, key):
        try:
            while True:
                connection = self._connections[key][0]
                try:
                    msg = connection.recv()
                except ConnectionAbortedError:
                    return  # Required to avoid pytest.PytestUnhandledThreadExceptionWarning
                if msg == _DISCONNECT_MESSAGE:
                    self._connections.pop(key)
                    connection.close()
                    return
                self.callback(msg)
        except Exception:
            pass  # Thread termination

    def stop(self):
        if self._connect_thread:
            terminate_thread(self._connect_thread)  # Do not accept anymore connections
            self._connect_thread = None
        timer = 0
        while self._connections:  # wait for clients to disconnect
            if timer >= 5:
                # clients taking too long, force shutdown
                for connection, thread in self._connections.values():
                    terminate_thread(thread)
                    connection.close()
                self._connections.clear()
            else:
                timer += 1
                sleep(1)
        self._listener.close()

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"


class Client(object):
    def __init__(self, port):
        self.port = port
        self._buffer = []
        self._client = None
        self._closed = False
        threading.Thread(target=self._connect, daemon=True).start()
        atexit_register(self.close)

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
                sleep(1)

    def send(self, stuff):
        if self._client:
            try:
                self._client.send(stuff)
            except Exception:  # Server shutdown
                pass
        else:
            self._buffer.append(stuff)

    def close(self):
        if self._closed:
            return
        try:
            while not self._client:
                sleep(0.5)
            for stuff in self._buffer:
                self._client.send(stuff)
            self._client.send(_DISCONNECT_MESSAGE)
            self._client.close()
            self._client = None
            self._closed = True
        except Exception as e:
            pass
