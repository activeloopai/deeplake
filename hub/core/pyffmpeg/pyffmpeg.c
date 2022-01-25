#include "avcodec.h"
#include "avformat.h"
#include "swscale.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>

static void logging(const char *fmt, ...);
static int decode_video_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, unsigned char **decompressed, struct SwsContext **sws_context, int *bufpos);
int readFunc(void *opaque, uint8_t *buf, int buf_size);

struct buffer_data
{
    uint8_t *ptr;
    size_t size;
};

int getVideoShape(unsigned char *file, int size, int ioBufferSize, int *shape, int isBytes)
{
    AVFormatContext *pFormatContext = NULL;
    AVIOContext *pioContext = NULL;
    unsigned char *ioBuffer;
    pFormatContext = avformat_alloc_context();
    struct buffer_data bd = {0};

    int ret;

    if (!pFormatContext)
    {
        logging("ERROR could not allocate memory for Format Context");
        return -1;
    }

    if (isBytes == 1)
    {
        bd.ptr = file;
        bd.size = size;
        ioBuffer = av_malloc(ioBufferSize);
        pioContext = avio_alloc_context(ioBuffer, ioBufferSize, 0, &bd, &readFunc, NULL, NULL);
        pFormatContext->pb = pioContext;
        ret = avformat_open_input(&pFormatContext, NULL, NULL, NULL);
    }
    else
    {
        ret = avformat_open_input(&pFormatContext, (const char *)file, NULL, NULL);
    }

    if (ret != 0)
    {
        logging("ERROR could not open the file");
        return -1;
    }

    if (avformat_find_stream_info(pFormatContext, NULL) < 0)
    {
        logging("ERROR could not get the stream info");
        return -1;
    }

    for (unsigned int i = 0; i < pFormatContext->nb_streams; i++)
    {

        AVCodecParameters *pLocalCodecParameters = NULL;
        pLocalCodecParameters = pFormatContext->streams[i]->codecpar;

        if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            float fps = (float)pFormatContext->streams[i]->avg_frame_rate.num / (float)pFormatContext->streams[i]->avg_frame_rate.den;
            float timebase = (float)pFormatContext->streams[i]->time_base.num / (float)pFormatContext->streams[i]->time_base.den;
            float duration = (float)pFormatContext->streams[i]->duration * timebase;
            if (duration < 0)
            {
                duration = (float)pFormatContext->duration / AV_TIME_BASE;
            }
            int n_frames = (int)duration * (int)fps;
            int width = pLocalCodecParameters->width;
            int height = pLocalCodecParameters->height;
            shape[0] = n_frames;
            shape[1] = height;
            shape[2] = width;
            break;
        }
    }
    avformat_close_input(&pFormatContext);
    if (isBytes == 1)
    {
        if (pioContext)
        {
            av_freep(&pioContext->buffer);
            av_freep(&pioContext);
        }
    }
    return 0;
}

int decompressVideo(unsigned char *file, int size, int ioBufferSize, unsigned char *decompressed, int isBytes, int nbytes)
{
    AVFormatContext *pFormatContext = NULL;
    AVIOContext *pioContext = NULL;
    unsigned char *ioBuffer;
    pFormatContext = avformat_alloc_context();
    struct buffer_data bd = {0};

    int ret;

    if (!pFormatContext)
    {
        logging("ERROR could not allocate memory for Format Context");
        return -1;
    }

    if (isBytes == 1)
    {
        bd.ptr = file;
        bd.size = size;
        ioBuffer = av_malloc(ioBufferSize);
        pioContext = avio_alloc_context(ioBuffer, ioBufferSize, 0, &bd, &readFunc, NULL, NULL);
        pFormatContext->pb = pioContext;
        ret = avformat_open_input(&pFormatContext, NULL, NULL, NULL);
    }
    else
    {
        ret = avformat_open_input(&pFormatContext, (const char *)file, NULL, NULL);
    }

    if (ret != 0)
    {
        logging("ERROR could not open the file");
        return -1;
    }

    if (avformat_find_stream_info(pFormatContext, NULL) < 0)
    {
        logging("ERROR could not get the stream info");
        return -1;
    }

    int video_stream_index = -1;
    AVCodec *pCodec = NULL;
    AVCodecParameters *pCodecParameters = NULL;

    for (unsigned int i = 0; i < pFormatContext->nb_streams; i++)
    {

        AVCodecParameters *pLocalCodecParameters = NULL;
        pLocalCodecParameters = pFormatContext->streams[i]->codecpar;
        AVCodec *pLocalCodec = NULL;
        pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);

        if (pLocalCodec == NULL)
        {
            logging("ERROR unsupported codec!");
            continue;
        }

        if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_index = i;
            pCodec = pLocalCodec;
            pCodecParameters = pLocalCodecParameters;
            break;
        }
    }

    if (video_stream_index == -1)
    {
        logging("Input does not contain a video stream!");
        return -1;
    }

    AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);

    if (!pCodecContext)
    {
        logging("failed to allocated memory for AVCodecContext");
        return -1;
    }

    if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0)
    {
        logging("failed to copy codec params to codec context");
        return -1;
    }

    if (avcodec_open2(pCodecContext, pCodec, NULL) < 0)
    {
        logging("failed to open codec through avcodec_open2");
        return -1;
    }

    AVFrame *pFrame = av_frame_alloc();

    if (!pFrame)
    {
        logging("failed to allocated memory for AVFrame");
        return -1;
    }

    AVPacket *pPacket = av_packet_alloc();

    if (!pPacket)
    {
        logging("failed to allocated memory for AVPacket");
        return -1;
    }

    struct SwsContext *sws_context = NULL;

    int response = 0;
    int bufpos = 0;
    unsigned char *start = decompressed;
    while (av_read_frame(pFormatContext, pPacket) >= 0)
    {
        if (pPacket->stream_index == video_stream_index)
        {
            response = decode_video_packet(pPacket, pCodecContext, pFrame, &decompressed, &sws_context, &bufpos);
            decompressed = start + bufpos;
            if (response < 0)
                break;
            if (bufpos >= nbytes)
                break;
        }
        av_packet_unref(pPacket);
    }
    avformat_close_input(&pFormatContext);
    av_packet_free(&pPacket);
    av_frame_free(&pFrame);
    avcodec_free_context(&pCodecContext);
    if (isBytes == 1)
    {
        if (pioContext)
        {
            av_freep(&pioContext->buffer);
            av_freep(&pioContext);
        }
    }
    return 0;
}

static int decode_video_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame, unsigned char **decompressed, struct SwsContext **sws_context, int *bufpos)
{
    int response = avcodec_send_packet(pCodecContext, pPacket);
    while (response >= 0)
    {
        response = avcodec_receive_frame(pCodecContext, pFrame);
        if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
        {
            break;
        }
        else if (response < 0)
        {
            return response;
        }

        if (response >= 0)
        {
            int height = pFrame->height;
            int width = pFrame->width;
            const int out_linesize[1] = {3 * width};
            (*sws_context) = sws_getCachedContext((*sws_context), width, height, pFrame->format, width, height, AV_PIX_FMT_RGB24, 0, 0, 0, 0);
            sws_scale((*sws_context), (const uint8_t *const *)&pFrame->data, pFrame->linesize, 0, height, (uint8_t *const *)decompressed, out_linesize);
            *bufpos += height * width * 3;
        }
        break;
    }
    return 0;
}

int readFunc(void *opaque, uint8_t *buf, int buf_size)
{
    struct buffer_data *bd = (struct buffer_data *)opaque;
    buf_size = FFMIN(buf_size, (int)bd->size);
    memmove(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;

    return buf_size;
}

static void logging(const char *fmt, ...)
{
    va_list args;
    fprintf(stderr, "LOG: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}
