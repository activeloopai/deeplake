
from hub.core.compression.compressions import AUDIO_COMPRESSIONS
from hub.core.compression.compressor import BaseCompressor
from typing import Union, List
import os

if os.name == "nt":
    _FFMPEG_BINARY = "ffmpeg.exe"
    _FFPROBE_BINARY = "ffprobe.exe"
else:
    _FFMPEG_BINARY = "ffmpeg"
    _FFPROBE_BINARY = "ffprobe"

_FFMPEG_EXISTS = None


def ffmpeg_exists():
    global _FFMPEG_EXISTS
    if _FFMPEG_EXISTS is None:
        _FFMPEG_EXISTS = True
        try:
            retval = sp.run(
                [_FFMPEG_BINARY, "-h"], stdout=sp.PIPE, stderr=sp.PIPE
            ).returncode
        except FileNotFoundError as e:
            _FFMPEG_EXISTS = False
    return _FFMPEG_EXISTS


def ffmpeg_binary():
    if ffmpeg_exists():
        return _FFMPEG_BINARY
    raise FileNotFoundError(
        "FFMPEG not found. Install FFMPEG to use hub's video features"
    )


def ffprobe_binary():
    if ffmpeg_exists():
        return _FFPROBE_BINARY
    raise FileNotFoundError(
        "FFMPEG not found. Install FFMPEG to use hub's video features"
    )


def _read_video_shape(file: Union[bytes, memoryview, str]) -> Tuple[int, ...]:
    info = _get_video_info(file)
    if info["duration"] is None:
        nframes = -1
    else:
        nframes = math.floor(info["duration"] * info["rate"])
    return (nframes, info["height"], info["width"], 3)


def _get_video_info(file: Union[bytes, memoryview, str]) -> dict:
    duration = None
    command = [
        ffprobe_binary(),
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1",
        "pipe:",
    ]

    if isinstance(file, str):
        command[-1] = file
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 5)
        raw_info = pipe.stdout.read()  # type: ignore
        raw_err = pipe.stderr.read()  # type: ignore
        pipe.communicate()
        duration = bytes.decode(re.search(DURATION_RE, raw_err).groups()[0])  # type: ignore
        duration = _to_seconds(duration)
    else:
        if file[: len(_HUB_MKV_HEADER)] == _HUB_MKV_HEADER:
            mv = memoryview(file)
            n = len(_HUB_MKV_HEADER) + 2
            duration = struct.unpack("f", mv[n : n + 4])[0]
            file = mv[n + 4 :]
        pipe = sp.Popen(
            command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 5
        )
        raw_info = pipe.communicate(input=file)[0]
    ret = dict(
        map(lambda kv: (bytes.decode(kv[0]), kv[1]), re.findall(INFO_RE, raw_info))
    )
    ret["width"] = int(ret["width"])
    ret["height"] = int(ret["height"])
    if "duration" in ret:
        ret["duration"] = float(ret["duration"])
    else:
        ret["duration"] = duration
    ret["rate"] = float(eval(ret["rate"]))
    return ret


DURATION_RE = re.compile(rb"Duration: ([0-9:.]+),")


def _to_seconds(time):
    return sum([60 ** i * float(j) for (i, j) in enumerate(time.split(":")[::-1])])


def _to_hub_mkv(file: str):
    command = [
        ffmpeg_binary(),
        "-i",
        file,
        "-codec",
        "copy",
        "-f",
        "matroska",
        "pipe:",
    ]
    pipe = sp.Popen(
        command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 5
    )
    mkv, raw_info = pipe.communicate()
    duration = bytes.decode(re.search(DURATION_RE, raw_info).groups()[0])  # type: ignore
    duration = _to_seconds(duration)
    mkv = _HUB_MKV_HEADER + struct.pack("<Hf", 4, duration) + mkv
    return mkv

class VideoCompressor(BaseCompressor):
    supported_compressions = AUDIO_COMPRESSIONS

    def __init__(self, compression):
        super(VideoCompressor, self).__init__(compression=compression)

    def compress(self, data: np.ndarray) -> bytes:
        raise NotImplementedError(
            "In order to store video data, you should use `hub.read(path_to_file)`. Compressing raw data is not yet supported."
        )

    def decompress(self, compressed: Union[bytes, memoryview, str]) -> np.ndarray:
        file = compressed

        shape = _read_video_shape(file)

        command = [
            ffmpeg_binary(),
            "-i",
            "pipe:",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        if isinstance(file, str):
            command[2] = file
            pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 8)
            raw_video = pipe.communicate()[0]
        else:
            file = _strip_hub_mp4_header(file)
            pipe = sp.Popen(
                command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 8
            )
            raw_video = pipe.communicate(input=file)[0]  # type: ignore
        return np.frombuffer(raw_video[: int(np.prod(shape))], dtype=np.uint8).reshape(
            shape
        )
