#ifndef _ZNZLIB_H_
#define _ZNZLIB_H_

/*
znzlib.h  (zipped or non-zipped library)

*****            This code is released to the public domain.            *****

*****  Author: Mark Jenkinson, FMRIB Centre, University of Oxford       *****
*****  Date:   September 2004                                           *****

*****  Neither the FMRIB Centre, the University of Oxford, nor any of   *****
*****  its employees imply any warranty of usefulness of this software  *****
*****  for any purpose, and do not assume any liability for damages,    *****
*****  incidental or otherwise, caused by any use of this document.     *****

*/

/*

This library provides an interface to both compressed (gzip/zlib) and
uncompressed (normal) file IO.  The functions are written to have the
same interface as the standard file IO functions.

To use this library instead of normal file IO, the following changes
are required:
 - replace all instances of FILE* with znzFile
 - change the name of all function calls, replacing the initial character
   f with the znz  (e.g. fseek becomes znzseek)
 - add a third parameter to all calls to znzopen (previously fopen)
   that specifies whether to use compression (1) or not (0)
 - use znz_isnull rather than any (pointer == NULL) comparisons in the code

NB: seeks for writable files with compression are quite restricted

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

#include <memory>

/* include optional check for HAVE_FDOPEN here, from deleted config.h:

   uncomment the following line if fdopen() exists for your compiler and
   compiler options
*/
/* #define HAVE_FDOPEN */


#ifdef HAVE_ZLIB

#if defined(ITKZLIB) && !defined(ITK_USE_SYSTEM_ZLIB)
#include "itk_zlib.h"
#else
#include "zlib.h"
#endif
#endif

#ifdef HAVE_ZLIB
class zfmembuf {
public:
    zfmembuf(unsigned char* data): data_(data) {}
    ~zfmembuf() {
        free(data_);
    }
    unsigned char* data() const {
        return data_;
    }
private:
    unsigned char* data_ = NULL;
};

typedef struct zfmem_ {
  size_t pos;
  std::shared_ptr<zfmembuf> buffer;
  size_t size;
} zfmem;
#endif // HAVE_ZLIB

struct znzptr {
  int withz;
  FILE* nzfptr;
#ifdef HAVE_ZLIB
  gzFile zfptr;
  zfmem* zmemptr;
#endif
};

/* the type for all file pointers */
typedef struct znzptr * znzFile;

#define znz_isnull(f) ((f) == NULL)
#define znzclose(f)   Xznzclose(&(f))

/* Note extra argument (use_compression) where
   use_compression==0 is no compression
   use_compression!=0 uses zlib (gzip) compression
*/

znzFile znzopen(const char *path, const char *mode, int use_compression);
znzFile znzmemopen(znzFile fp, const uint8_t* data, const size_t size, const char *mode, int use_compression, const size_t estimated_output_size);

int Xznzclose(znzFile * file);

size_t znzread(void* buf, size_t size, size_t nmemb, znzFile file);
size_t znzassign(void** buf, size_t size, size_t nmemb, znzFile file);

size_t znzwrite(const void* buf, size_t size, size_t nmemb, znzFile file);

long znzseek(znzFile file, long offset, int whence);

int znzrewind(znzFile stream);

long znztell(znzFile file);

int znzputs(const char *str, znzFile file);

#endif
