# Streams video from a chunk

from typing import Optional, Callable, Tuple
from flask import Flask, request, Response
from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder
from deeplake.core.storage import StorageProvider
from deeplake.core.meta.encode.byte_positions import BytePositionsEncoder
from deeplake.util.keys import get_chunk_key
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.threading import terminate_thread
from deeplake.util.hash import hash_inputs
from deeplake.constants import MB
import socketserver
import deeplake
import threading
import struct
import numpy as np
import re

import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


_PORT: Optional[int] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_STREAMS = {}
_VIDEO_STREAM_CHUNK_SIZE = 1 * MB


def _get_chunk_key(tensor, index: int) -> str:
    chunk_id = tensor.chunk_engine.chunk_id_encoder[index][0]  # videos are not tiled
    chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
    chunk_commit_id, key = tensor.chunk_engine.get_chunk_commit(chunk_name)
    chunk_key = get_chunk_key(key, chunk_name, chunk_commit_id)
    return chunk_key


def _get_stream_key(storage: StorageProvider, chunk_key: str):
    return hash_inputs(storage.root, chunk_key)


def get_video_stream_url(tensor, index: int) -> str:
    chunk_key = _get_chunk_key(tensor, index)
    storage = get_base_storage(tensor.storage)
    stream_key = _get_stream_key(storage, chunk_key)
    if stream_key not in _STREAMS:
        _STREAMS[stream_key] = _VideoStream(storage, chunk_key)
    _start_server()
    index = tensor.chunk_engine.chunk_id_encoder.translate_index_relative_to_chunks(
        index
    )
    return f"http://localhost:{_PORT}/video/{stream_key}/{index}"


class _LazyByteStream:
    MIN_READ_SIZE = 100  # bytes

    def __init__(self, read_f: Callable):
        self.read_f = read_f
        self.buffer = bytearray()

    def read(self, n: int) -> bytes:
        rem = n - len(self.buffer)
        if rem > 0:
            new_bts = self.read_f(max(self.MIN_READ_SIZE, rem))
            ret = self.buffer + new_bts[:rem]
            self.buffer = bytearray(new_bts[rem:])
            return ret
        else:
            ret = self.buffer[:n]
            self.buffer = self.buffer[n:]
            return ret


class _VideoStream:
    __slots__ = [
        "storage",
        "chunk_key",
        "byte_positions_encoder",
        "header_size",
        "chunk_size",
    ]

    def __init__(self, storage: StorageProvider, chunk_key: str):
        self.storage: StorageProvider = storage
        self.chunk_key: str = chunk_key
        self._read_header()

    def _read_header(self):
        offset = {"value": 0}

        def _read(n: int) -> bytes:
            o = offset["value"]
            ret = self.storage.get_bytes(self.chunk_key, o, o + n)
            offset["value"] += n
            return ret

        lazy = _LazyByteStream(_read)
        n_ver = lazy.read(1)[0]
        lazy.read(n_ver)
        r, c = struct.unpack("<ii", lazy.read(8))
        enc_dtype = np.dtype(deeplake.constants.ENCODING_DTYPE)
        isize = enc_dtype.itemsize
        sh_enc_size = r * c * isize
        lazy.read(sh_enc_size)
        r = int.from_bytes(lazy.read(4), "little")
        bp_enc_size = r * 3 * isize
        self.byte_positions_encoder = BytePositionsEncoder(
            np.frombuffer(lazy.read(bp_enc_size), dtype=enc_dtype).reshape(r, 3).copy()
        )
        self.header_size = 13 + n_ver + sh_enc_size + bp_enc_size
        self.chunk_size = self.storage.get_object_size(self.chunk_key)

    def read(
        self, index: int, start_byte: int, end_byte: int
    ) -> Tuple[bytes, int, int, int]:
        ret_start = start_byte
        sample_start_index, sample_end_index = self.byte_positions_encoder[index]
        offset = self.header_size + sample_start_index
        start_byte += offset
        limit = sample_end_index + self.header_size
        if start_byte >= limit:
            start_byte = offset
            ret_start = 0
        if end_byte is None:
            end_byte = limit
        else:
            end_byte = min(end_byte + offset + 1, limit)

        # Chunking
        chunk_size = end_byte - start_byte
        chunk_size = min(chunk_size, _VIDEO_STREAM_CHUNK_SIZE)
        end_byte = start_byte + chunk_size

        byts = bytes(self.storage.get_bytes(self.chunk_key, start_byte, end_byte))
        return byts, ret_start, len(byts), sample_end_index - sample_start_index


def _get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def _is_server_running() -> bool:
    return (_SERVER_THREAD is not None) and _SERVER_THREAD.is_alive()


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


def _stop_server():
    global _SERVER_THREAD
    if not _is_server_running():
        return
    request.environ.get("werkzeug.server.shutdown", lambda: None)()
    terminate_thread(_SERVER_THREAD)
    _SERVER_THREAD = None
    _STREAMS.clear()


_APP = Flask("video_stream")


@_APP.after_request
def after_request(response):
    response.headers.add("Accept-Ranges", "bytes")
    return response


@_APP.route("/video/<chunk_id>/<sample_id>")
def stream_video(chunk_id, sample_id):
    range_header = request.headers.get("Range", None)
    start, end = 0, None
    if range_header:
        match = re.search(r"(\d+)-(\d*)", range_header)
        groups = match.groups()

        if groups[0]:
            start = int(groups[0])
        if groups[1]:
            end = int(groups[1])

    chunk, start, length, file_size = _STREAMS[chunk_id].read(
        int(sample_id), start, end
    )
    assert len(chunk) == length

    resp = Response(
        chunk,
        206,
        mimetype="video/mp4",
        content_type="video/mp4",
    )
    resp.headers.add(
        "Connection",
        "keep-alive",
    )
    resp.headers.add("Accept-Ranges", "bytes")
    resp.headers.add(
        "Content-Range",
        "bytes {0}-{1}/{2}".format(start, start + length - 1, file_size),
    )
    return resp
