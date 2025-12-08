#ifndef _RNIFTI_H_
#define _RNIFTI_H_

// RNiftyReg and divest have used HAVE_R, so accept this variant for compatibility
#if !defined(USING_R) && defined(HAVE_R)
#define USING_R
#endif

// Defined since RNifti v0.10.0, and equal to 100 * (major version) + (minor version). May not
// change if the API does not change, and in particular never changes with patch level
#define RNIFTI_VERSION 104

// Versions 1 and 2 of the NIfTI reference library are mutually incompatible, but RNifti does some
// work to get them to play nicely:
// 
// - The compile-time constant RNIFTI_NIFTILIB_VERSION indicates which version of the library has
//   precedence. nifti1_io.h sets this to 1, and nifti2.io.h to 2, so the first-included header
//   wins unless the user sets a value explicitly.
// - nifti_image is aliased to the appropriate struct type according to the library version in use.
// - Library functions with the same name but different signatures in the two versions are renamed
//   to use "nifti2" in place of "nifti" in the version 2 library. They are aliased back to their
//   original names if RNIFTI_NIFTILIB_VERSION is 2 and NO_REMAP_NIFTI2_FUNCTIONS *is not* defined.
// - Library functions that are essentially the same in the two versions are fenced out of
//   nifti1_io.c (if RNIFTI_NIFTILIB_DEDUPLICATE is defined), to avoid duplicate symbols in the
//   compiled package library.
// 
// There are therefore several possible modes of usage:
// 
// 1. Standalone programs that include RNifti.h can *first* define RNIFTI_NIFTILIB_VERSION to
//    choose the library version required (the default is 1). They should link against nifti1_io.o
//    or nifti2_io.o, accordingly. (A mismatch will result in compiler/linker errors.) See the
//    "standalone" directory for an example.
// 2. Standalone or linked R package C/C++ code can include "niftilib/nifti1_io.h" or "niftilib/
//    nifti2_io.h", use the appropriate version of the library, and not worry about the clash. This
//    will make most sense for existing code already written for one or other version of the
//    NIfTI library. Standalone code will again need to link to the appropriate object file; R will
//    handle linkage for packages, but the API header "RNiftiAPI.h" must also be included. See the
//    "clients" directory for an example of the latter.
// 3. Code that explicitly wants to handle both versions of the library should define
//    NO_REMAP_NIFTI2_FUNCTIONS to avoid name clashes, include both library headers, and use
//    nifti2_* functions explicitly when required.
#if !defined(RNIFTI_NIFTILIB_VERSION) || (RNIFTI_NIFTILIB_VERSION == 1)
#include "niftilib/nifti1_io.h"
#include "niftilib/nifti2_image.h"
#else
#include "niftilib/nifti2_io.h"
#endif

#ifdef __cplusplus
#include "RNifti/NiftiImage.h"

// Defined since RNifti v0.3.0
#define HAVE_RNIFTI_NAMESPACE

extern "C" {
#endif // __cplusplus

extern void niftilib_register_all (void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
