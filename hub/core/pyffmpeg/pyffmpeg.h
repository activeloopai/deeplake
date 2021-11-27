int getVideoShape(unsigned char *file, int ioBufferSize, int *shape, int isBytes);
int decompressVideo(unsigned char *file, int ioBufferSize, unsigned char *decompressed, int isBytes, int n_packets);
