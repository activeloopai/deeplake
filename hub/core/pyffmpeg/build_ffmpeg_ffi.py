from cffi import FFI  # type: ignore
import posixpath
import os

from hub.core import pyffmpeg

ffibuilder = FFI()

pyffmpeg_include_dirs = [
    posixpath.split(__file__)[0],
    os.getcwd(),
]

ffibuilder.cdef(
    """
    int getVideoShape(unsigned char *file, int size, int ioBufferSize, int *shape, int isBytes);
    int decompressVideo(unsigned char *file, int size, int ioBufferSize, unsigned char *decompressed, int isBytes, int nbytes);
    """
)

ffibuilder.set_source(
    "hub.core.pyffmpeg._pyffmpeg",
    """
    #include "avcodec.h"
    #include "avformat.h"
    #include "swscale.h"
    #include "hub/core/pyffmpeg/pyffmpeg.h"
    """,
    include_dirs=pyffmpeg_include_dirs,
    sources=["hub/core/pyffmpeg/pyffmpeg.c"],
    libraries=["avcodec", "avformat", "swscale"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
