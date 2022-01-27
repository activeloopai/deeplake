int getVideoShape(unsigned char *file, int size, int ioBufferSize, int *shape, int isBytes);
int decompressVideo(unsigned char *file, int size, int ioBufferSize, int start_frame, unsigned char *decompressed, int isBytes, int nbytes);
