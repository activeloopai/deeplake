from cffi import FFI
import os

from hub.core import pyffmpeg

ffibuilder = FFI()

pyffmpeg_include_dir = os.getcwd()

ffibuilder.cdef(
    """
    int getVideoShape(unsigned char *file, int ioBufferSize, int *shape, int isBytes);
    int decompressVideo(unsigned char *file, int ioBufferSize, unsigned char *decompressed, int isBytes, int n_packets);
    """
)

ffibuilder.set_source(
    "hub.core.pyffmpeg._pyffmpeg",
    """
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include "hub/core/pyffmpeg/pyffmpeg.h"
    """,
    sources=["hub/core/pyffmpeg/pyffmpeg.c"],
    include_dirs=[pyffmpeg_include_dir],
    libraries=["avcodec", "avformat", "swscale"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
