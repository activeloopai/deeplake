from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    int getVideoShape(unsigned char *file, int ioBufferSize, int *shape, int isBytes);
    int decompressVideo(unsigned char *file, int ioBufferSize, unsigned char *decompressed, int isBytes, int n_packets);
    """
)

ffibuilder.set_source(
    "_pyffmpeg",
    """
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include "pyffmpeg.h"
    """,
    sources=["pyffmpeg.c"],
    libraries=["avcodec", "avformat", "swscale"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
