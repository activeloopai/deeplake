/** \file znzlib.c
    \brief Low level i/o interface to compressed and noncompressed files.
        Written by Mark Jenkinson, FMRIB

This library provides an interface to both compressed (gzip/zlib) and
uncompressed (normal) file IO.  The functions are written to have the
same interface as the standard file IO functions.

To use this library instead of normal file IO, the following changes
are required:
 - replace all instances of FILE* with znzFile
 - change the name of all function calls, replacing the initial character
   f with the znz  (e.g. fseek becomes znzseek)
   one exception is rewind() -> znzrewind()
 - add a third parameter to all calls to znzopen (previously fopen)
   that specifies whether to use compression (1) or not (0)
 - use znz_isnull rather than any (pointer == NULL) comparisons in the code
   for znzfile types (normally done after a return from znzopen)

NB: seeks for writable files with compression are quite restricted

 */

#include "znzlib/znzlib.h"
#include "RNifti/NiftiImage_print.h"

/*
znzlib.c  (zipped or non-zipped library)

*****            This code is released to the public domain.            *****

*****  Author: Mark Jenkinson, FMRIB Centre, University of Oxford       *****
*****  Date:   September 2004                                           *****

*****  Neither the FMRIB Centre, the University of Oxford, nor any of   *****
*****  its employees imply any warranty of usefulness of this software  *****
*****  for any purpose, and do not assume any liability for damages,    *****
*****  incidental or otherwise, caused by any use of this document.     *****

*/

/* Note extra argument (use_compression) where
   use_compression==0 is no compression
   use_compression!=0 uses zlib (gzip) compression
*/

namespace {
#ifdef HAVE_ZLIB
#define CHUNK_SIZE 16384
int decompress_gzip(const unsigned char* compressed_data,
                    unsigned long compressed_size,
                    unsigned long estimated_size,
                    unsigned char** decompressed_data,
                    unsigned long* decompressed_size)
{
    int ret;
    z_stream strm;
    unsigned long total_size = 0;
    unsigned long offset = 0;
    unsigned long current_buffer_size = estimated_size ? estimated_size : 1;

    // Allocate initial memory for decompressed data
    *decompressed_data = (unsigned char*)malloc(current_buffer_size);
    if (*decompressed_data == NULL) {
        Rc_fprintf_stderr("Memory allocation failed\n");
        return -1;
    }

    // Initialize zlib stream
    memset(&strm, 0, sizeof(z_stream));
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;

    // Initialize inflate
    ret = inflateInit2(&strm, 15 + 32); // 15 + 32 for automatic detection of gzip format
    if (ret != Z_OK) {
        Rc_fprintf_stderr("Failed to initialize zlib: %s\n", zError(ret));
        free(*decompressed_data);
        *decompressed_data = NULL;
        return ret;
    }

    strm.avail_in = compressed_size;
    strm.next_in = const_cast<Bytef*>(compressed_data);

    // Decompress loop
    do {
        unsigned long avail_to_write = current_buffer_size - offset;
        strm.avail_out = avail_to_write;
        strm.next_out = *decompressed_data + offset;

        ret = inflate(&strm, Z_NO_FLUSH);
        switch (ret) {
        case Z_NEED_DICT:
        case Z_DATA_ERROR:
        case Z_MEM_ERROR:
            Rc_fprintf_stderr("Decompression failed: %s\n", zError(ret));
            inflateEnd(&strm);
            free(*decompressed_data);
            *decompressed_data = NULL;
            return ret;
        }

        // Calculate total size of decompressed data
        long output_size = avail_to_write - strm.avail_out;
        total_size += output_size;

        if (!strm.avail_out && strm.avail_in) {
            // Reallocate memory for decompressed data
            unsigned long prev_buf_size = current_buffer_size;
            current_buffer_size = total_size + strm.avail_in;
            *decompressed_data = (unsigned char*)realloc(*decompressed_data, current_buffer_size);
            if (*decompressed_data == NULL) {
                Rc_fprintf_stderr("Memory reallocation failed\n");
                inflateEnd(&strm);
                return -1;
            }
        }

        offset += output_size;
    } while (ret != Z_STREAM_END);

    // Set the final decompressed size
    *decompressed_size = total_size;

    // Clean up
    inflateEnd(&strm);

    return Z_OK;
}
#endif // HAVE_ZLIB
} // namespace

// Windows compatibility: fmemopen alternative using tmpfile
#ifdef _WIN32
static FILE* fmemopen_windows(void* buf, size_t size, const char* mode)
{
    // Create a temporary file
    FILE* tmpf = tmpfile();
    if (tmpf == NULL) {
        Rc_fprintf_stderr("** ERROR: Failed to create temporary file\n");
        return NULL;
    }

    // Write the buffer contents to the temporary file
    if (fwrite(buf, 1, size, tmpf) != size) {
        Rc_fprintf_stderr("** ERROR: Failed to write buffer to temporary file\n");
        fclose(tmpf);
        return NULL;
    }

    // Rewind to the beginning for reading
    rewind(tmpf);
    return tmpf;
}
#endif

znzFile znzopen(const char* path, const char* mode, int use_compression)
{
    znzFile file;
    file = (znzFile)calloc(1, sizeof(struct znzptr));
    if (file == NULL) {
        Rc_fprintf_stderr("** ERROR: znzopen failed to alloc znzptr\n");
        return NULL;
    }

    file->nzfptr = NULL;

#ifdef HAVE_ZLIB
    file->zfptr = NULL;
    file->zmemptr = NULL;

    if (use_compression) {
        file->withz = 1;
        if ((file->zfptr = gzopen(path, mode)) == NULL) {
            free(file);
            file = NULL;
        }
    } else {
#endif

        file->withz = 0;
        if ((file->nzfptr = fopen(path, mode)) == NULL) {
            free(file);
            file = NULL;
        }

#ifdef HAVE_ZLIB
    }
#endif

    return file;
}

znzFile znzmemopen(znzFile fp,
                   const uint8_t* data,
                   const size_t size,
                   const char* mode,
                   int use_compression,
                   const size_t estimated_output_size)
{
    znzFile file = fp;
    if (file == NULL) {
        file = (znzFile)calloc(1, sizeof(struct znzptr));
        if (file == NULL) {
            Rc_fprintf_stderr("** ERROR: znzopen failed to alloc znzptr\n");
            return NULL;
        }
    }

    file->nzfptr = NULL;

#ifdef HAVE_ZLIB
    file->zfptr = NULL;
    file->zmemptr = NULL;

    if (use_compression) {
        unsigned char* decompressed_data = NULL;
        unsigned long decompressed_size = 0;

        decompress_gzip(
            (const unsigned char*)data, size, estimated_output_size, &decompressed_data, &decompressed_size);

        if (decompressed_data == NULL) {
            Rc_fprintf_stderr("** ERROR: znzmemopen failed to decompress memory buffer\n");
        } else {
            file->withz = 1;
            file->zmemptr = new zfmem();
            file->zmemptr->pos = 0;
            file->zmemptr->buffer = std::make_shared<zfmembuf>(decompressed_data);
            file->zmemptr->size = decompressed_size;
        }
    } else {
#endif

        file->withz = 0;
#ifdef _WIN32
        // Use Windows-compatible alternative
        if ((file->nzfptr = fmemopen_windows((void*)data, size, mode)) == NULL) {
#else
    // Use standard POSIX fmemopen on Unix/Linux/Mac
    if ((file->nzfptr = fmemopen((void*)data, size, mode)) == NULL) {
#endif
            if (fp == NULL) {
                free(file);
            }
            file = NULL;
        }

#ifdef HAVE_ZLIB
    }
#endif

    return file;
}

int Xznzclose(znzFile* file)
{
    int retval = 0;
    if (*file != NULL) {
#ifdef HAVE_ZLIB
        if ((*file)->zfptr != NULL) {
            retval = gzclose((*file)->zfptr);
        }
        if ((*file)->zmemptr != nullptr) {
            (*file)->zmemptr->buffer = nullptr;
            delete (*file)->zmemptr;
            (*file)->zmemptr = NULL;
        }
#endif
        if ((*file)->nzfptr != NULL) {
            retval = fclose((*file)->nzfptr);
        }

        free(*file);
        *file = NULL;
    }
    return retval;
}

/* we already assume ints are 4 bytes */
#undef ZNZ_MAX_BLOCK_SIZE
#define ZNZ_MAX_BLOCK_SIZE (1 << 30)

size_t znzread(void* buf, size_t size, size_t nmemb, znzFile file)
{
    size_t remain = size * nmemb;
    char* cbuf = (char*)buf;
    unsigned n2read;
    int nread;

    if (file == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    if (file->zfptr != NULL) {
        /* gzread/write take unsigned int length, so maybe read in int pieces
           (noted by M Hanke, example given by M Adler)   6 July 2010 [rickr] */
        while (remain > 0) {
            n2read = (remain < ZNZ_MAX_BLOCK_SIZE) ? remain : ZNZ_MAX_BLOCK_SIZE;
            nread = gzread(file->zfptr, (void*)cbuf, n2read);
            if (nread < 0)
                return nread; /* returns -1 on error */

            remain -= nread;
            cbuf += nread;

            /* require reading n2read bytes, so we don't get stuck */
            if (nread < (int)n2read)
                break; /* return will be short */
        }

        /* warn of a short read that will seem complete */
        if (remain > 0 && remain < size)
            Rc_fprintf_stderr("** znzread: read short by %u bytes\n", (unsigned)remain);

        return nmemb - remain / size; /* return number of members processed */
    }
    if (file->zmemptr != NULL) {
        memcpy(buf, file->zmemptr->buffer->data() + file->zmemptr->pos, remain);
        file->zmemptr->pos += remain;
        return remain;
    }
#endif
    return fread(buf, size, nmemb, file->nzfptr);
}

// this method is meant to be used only for cases when `zmemptr` is not null.
// this provides an access to the underlying memory_buffer for znzFile.
// the buffer is shared and is used whenever the decoder gets the data
size_t znzassign(void** buf, size_t size, size_t nmemb, znzFile file)
{
    if (file == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    if (file->zfptr != NULL) {
        Rc_fprintf_stderr("** znzassign is not supported for file decompressor, use znzread instead\n");
        return -1;
    }
    if (file->zmemptr != NULL) {
        size_t remain = size * nmemb;
        *buf = file->zmemptr->buffer->data() + file->zmemptr->pos;
        file->zmemptr->pos += remain;
        return remain;
    }
#endif
    Rc_fprintf_stderr("** znzassign is not supported for file decompressor, use znzread instead\n");
    return -1;
}

size_t znzwrite(const void* buf, size_t size, size_t nmemb, znzFile file)
{
    size_t remain = size * nmemb;
    const char* cbuf = (const char*)buf;
    unsigned n2write;
    int nwritten;

    if (file == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    if (file->zfptr != NULL) {
        while (remain > 0) {
            n2write = (remain < ZNZ_MAX_BLOCK_SIZE) ? remain : ZNZ_MAX_BLOCK_SIZE;
            nwritten = gzwrite(file->zfptr, (const void*)cbuf, n2write);

            /* gzread returns 0 on error, but in case that ever changes... */
            if (nwritten < 0)
                return nwritten;

            remain -= nwritten;
            cbuf += nwritten;

            /* require writing n2write bytes, so we don't get stuck */
            if (nwritten < (int)n2write)
                break;
        }

        /* warn of a short write that will seem complete */
        if (remain > 0 && remain < size)
            Rc_fprintf_stderr("** znzwrite: write short by %u bytes\n", (unsigned)remain);

        return nmemb - remain / size; /* return number of members processed */
    }
#endif
    return fwrite(buf, size, nmemb, file->nzfptr);
}

long znzseek(znzFile file, long offset, int whence)
{
    if (file == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    if (file->zfptr != NULL)
        return (long)gzseek(file->zfptr, offset, whence);
    if (file->zmemptr != NULL) {
        if (whence == SEEK_SET) {
            file->zmemptr->pos = offset;
        } else if (whence == SEEK_END) {
            file->zmemptr->pos = file->zmemptr->size - 1 - offset;
        } else {
            file->zmemptr->pos += offset;
        }

        return file->zmemptr->pos;
    }
#endif
    return fseek(file->nzfptr, offset, whence);
}

int znzrewind(znzFile stream)
{
    if (stream == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    /* On some systems, gzrewind() fails for uncompressed files.
       Use gzseek(), instead.               10, May 2005 [rickr]

       if (stream->zfptr!=NULL) return gzrewind(stream->zfptr);
    */

    if (stream->zfptr != NULL)
        return (int)gzseek(stream->zfptr, 0L, SEEK_SET);
    if (stream->zmemptr != NULL) {
        stream->zmemptr->pos = 0;
        return 0;
    }
#endif
    rewind(stream->nzfptr);
    return 0;
}

long znztell(znzFile file)
{
    if (file == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    if (file->zfptr != NULL)
        return (long)gztell(file->zfptr);
    if (file->zmemptr != NULL)
        return file->zmemptr->pos;
#endif
    return ftell(file->nzfptr);
}

int znzputs(const char* str, znzFile file)
{
    if (file == NULL) {
        return 0;
    }
#ifdef HAVE_ZLIB
    if (file->zfptr != NULL)
        return gzputs(file->zfptr, str);
#endif
    return fputs(str, file->nzfptr);
}