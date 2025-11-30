/** \file nifti2.h
    \brief Header structure for NIFTI-2 format.
 */

#ifndef __NIFTI2_HEADER
#define __NIFTI2_HEADER

/*---------------------------------------------------------------------------*/
/* Changes to the header from NIFTI-1 to NIFTI-2 are intended to allow for
   larger and more accurate fields.  The changes are as follows:

      - short dim[8]         -> int64_t dim[8]
      - float intent_p1,2,3  -> double intent_p1,2,3    (3 fields)
      - float pixdim[8]      -> double pixdim[8]
      - float vox_offset     -> int64_t vox_offset
      - float scl_slope      -> double scl_slope
      - float scl_inter      -> double scl_inter
      - float cal_max        -> double cal_max
      - float cal_min        -> double cal_min
      - float slice_duration -> double slice_duration
      - float toffset        -> double toffset
      - short slice_start    -> int64_t slice_start
      - short slice_end      -> int64_t slice_end
      - char slice_code      -> int32_t slice_code
      - char xyzt_units      -> int32_t xyzt_units
      - short intent_code    -> int32_t intent_code
      - short qform_code     -> int32_t qform_code
      - short sform_code     -> int32_t sform_code
      - float quatern_b,c,d  -> double quatern_b,c,d    (3 fields)
      - float srow_x,y,z[4]  -> double srow_x,y,z[4]    (3 fields)
      - char magic[4]        -> char magic[8]
      - char unused_str[15]  -> padding added at the end of the header

      - previously unused fields have been removed:
           data_type, db_name, extents, session_error, regular, glmax, glmin

      - the field order has been changed, notably with magic after sizeof_hdr

                                                          2 Jan, 2014 [rickr]
-----------------------------------------------------------------------------*/

#include <stdint.h>

/*! \struct nifti_2_header
    \brief Data structure defining the fields in the nifti2 header.
           This binary header should be found at the beginning of a valid
           NIFTI-2 header file.
 */

/* hopefully cross-platform solution to byte padding added by some compilers */
#pragma pack(push)
#pragma pack(1)

                           /*****************************/ /***********************/ /************/
struct nifti_2_header {    /* NIFTI-2 usage             */ /* NIFTI-1 usage       */ /*  offset  */
                           /*****************************/ /***********************/ /************/
   int32_t sizeof_hdr;     /*!< MUST be 540             */ /* MUST be 348         */ /*   0 */
   char    magic[8];       /*!< MUST be valid signature */ /* char magic[4]       */ /*   4 */
   int16_t datatype;       /*!< Defines data type!      */ /* short datatype      */ /*  12 */
   int16_t bitpix;         /*!< Number bits/voxel       */ /* short bitpix        */ /*  14 */
   int64_t dim[8];         /*!< Data array dimensions   */ /* short dim[8]        */ /*  16 */
   double  intent_p1;      /*!< 1st intent parameter    */ /* float intent_p1     */ /*  80 */
   double  intent_p2;      /*!< 2nd intent parameter    */ /* float intent_p2     */ /*  88 */
   double  intent_p3;      /*!< 3rd intent parameter    */ /* float intent_p3     */ /*  96 */
   double  pixdim[8];      /*!< Grid spacings           */ /* float pixdim[8]     */ /* 104 */
   int64_t vox_offset;     /*!< Offset into .nii file   */ /* float vox_offset    */ /* 168 */
   double  scl_slope;      /*!< Data scaling: slope     */ /* float scl_slope     */ /* 176 */
   double  scl_inter;      /*!< Data scaling: offset    */ /* float scl_inter     */ /* 184 */
   double  cal_max;        /*!< Max display intensity   */ /* float cal_max       */ /* 192 */
   double  cal_min;        /*!< Min display intensity   */ /* float cal_min       */ /* 200 */
   double  slice_duration; /*!< Time for 1 slice        */ /* float slice_duration*/ /* 208 */
   double  toffset;        /*!< Time axis shift         */ /* float toffset       */ /* 216 */
   int64_t slice_start;    /*!< First slice index       */ /* short slice_start   */ /* 224 */
   int64_t slice_end;      /*!< Last slice index        */ /* short slice_end     */ /* 232 */
   char    descrip[80];    /*!< any text you like       */ /* char descrip[80]    */ /* 240 */
   char    aux_file[24];   /*!< auxiliary filename      */ /* char aux_file[24]   */ /* 320 */
   int32_t qform_code;     /*!< NIFTI_XFORM_* code      */ /* short qform_code    */ /* 344 */
   int32_t sform_code;     /*!< NIFTI_XFORM_* code      */ /* short sform_code    */ /* 348 */
   double  quatern_b;      /*!< Quaternion b param      */ /* float quatern_b     */ /* 352 */
   double  quatern_c;      /*!< Quaternion c param      */ /* float quatern_c     */ /* 360 */
   double  quatern_d;      /*!< Quaternion d param      */ /* float quatern_d     */ /* 368 */
   double  qoffset_x;      /*!< Quaternion x shift      */ /* float qoffset_x     */ /* 376 */
   double  qoffset_y;      /*!< Quaternion y shift      */ /* float qoffset_y     */ /* 384 */
   double  qoffset_z;      /*!< Quaternion z shift      */ /* float qoffset_z     */ /* 392 */
   double  srow_x[4];      /*!< 1st row affine transform*/ /* float srow_x[4]     */ /* 400 */
   double  srow_y[4];      /*!< 2nd row affine transform*/ /* float srow_y[4]     */ /* 432 */
   double  srow_z[4];      /*!< 3rd row affine transform*/ /* float srow_z[4]     */ /* 464 */
   int32_t slice_code;     /*!< Slice timing order      */ /* char slice_code     */ /* 496 */
   int32_t xyzt_units;     /*!< Units of pixdim[1..4]   */ /* char xyzt_units     */ /* 500 */
   int32_t intent_code;    /*!< NIFTI_INTENT_* code     */ /* short intent_code   */ /* 504 */
   char    intent_name[16];/*!< name or meaning of data */ /* char intent_name[16]*/ /* 508 */
   char    dim_info;       /*!< MRI slice ordering      */ /* char dim_info       */ /* 524 */
   char    unused_str[15]; /*!< unused, filled with \0  */                           /* 525 */
};                                                                 /****** total bytes: 540 */
typedef struct nifti_2_header nifti_2_header;

/* restore packing behavior */
#pragma pack(pop)

/* base swap test on the suggested version check, rather than dim[0]
   swap4(348)==1543569408, swap4(540)==469893120 */
#define NIFTI2_NEEDS_SWAP(h) \
   ((h).sizeof_hdr == 1543569408 || (h).sizeof_hdr == 469893120)


#endif /* __NIFTI2_HEADER */
