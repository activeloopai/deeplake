/** \file nifti1_io.h
    \brief Data structures for using nifti1_io API.
           - Written by Bob Cox, SSCC NIMH
           - Revisions by Rick Reynolds, SSCC NIMH
 */
#ifndef _NIFTI_IO_HEADER_
#define _NIFTI_IO_HEADER_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#ifndef DONT_INCLUDE_ANALYZE_STRUCT
#define DONT_INCLUDE_ANALYZE_STRUCT  /*** not needed herein ***/
#endif
#include "niftilib/nifti1.h"                  /*** NIFTI-1 header specification ***/

#ifndef RNIFTI_NIFTILIB_VERSION
#define RNIFTI_NIFTILIB_VERSION 1
#endif

#include "RNifti/NiftiImage_print.h"
#include <znzlib/znzlib.h>

/*****===================================================================*****/
/*****         File nifti1_io.h == Declarations for nifti1_io.c          *****/
/*****...................................................................*****/
/*****            This code is released to the public domain.            *****/
/*****...................................................................*****/
/*****  Author: Robert W Cox, SSCC/DIRP/NIMH/NIH/DHHS/USA/EARTH          *****/
/*****  Date:   August 2003                                              *****/
/*****...................................................................*****/
/*****  Neither the National Institutes of Health (NIH), nor any of its  *****/
/*****  employees imply any warranty of usefulness of this software for  *****/
/*****  any purpose, and do not assume any liability for damages,        *****/
/*****  incidental or otherwise, caused by any use of this document.     *****/
/*****===================================================================*****/

/*
   Modified by: Mark Jenkinson (FMRIB Centre, University of Oxford, UK)
   Date: July/August 2004

      Mainly adding low-level IO and changing things to allow gzipped files
      to be read and written
      Full backwards compatability should have been maintained

   Modified by: Rick Reynolds (SSCC/DIRP/NIMH, National Institutes of Health)
   Date: December 2004

      Modified and added many routines for I/O.
*/

/********************** Some sample data structures **************************/

#if RNIFTI_NIFTILIB_VERSION == 1

typedef struct {                   /** 4x4 matrix struct **/
  float m[4][4] ;
} mat44 ;

typedef struct {                   /** 3x3 matrix struct **/
  float m[3][3] ;
} mat33 ;

/*...........................................................................*/

/*! \enum analyze_75_orient_code
 *  \brief Old-style analyze75 orientation
 *         codes.
 */
typedef enum _analyze75_orient_code {
  a75_transverse_unflipped = 0,
  a75_coronal_unflipped = 1,
  a75_sagittal_unflipped = 2,
  a75_transverse_flipped = 3,
  a75_coronal_flipped = 4,
  a75_sagittal_flipped = 5,
  a75_orient_unknown = 6
} analyze_75_orient_code;

/*! \struct nifti_image
    \brief High level data structure for open nifti datasets in the
           nifti1_io API.  Note that this structure is not part of the
           nifti1 format definition; it is used to implement one API
           for reading/writing formats in the nifti1 format.
 */
typedef struct {                /*!< Image storage struct **/

  int ndim ;                    /*!< last dimension greater than 1 (1..7) */
  int nx ;                      /*!< dimensions of grid array             */
  int ny ;                      /*!< dimensions of grid array             */
  int nz ;                      /*!< dimensions of grid array             */
  int nt ;                      /*!< dimensions of grid array             */
  int nu ;                      /*!< dimensions of grid array             */
  int nv ;                      /*!< dimensions of grid array             */
  int nw ;                      /*!< dimensions of grid array             */
  int dim[8] ;                  /*!< dim[0]=ndim, dim[1]=nx, etc.         */
  size_t nvox ;                    /*!< number of voxels = nx*ny*nz*...*nw   */
  int nbyper ;                  /*!< bytes per voxel, matches datatype    */
  int datatype ;                /*!< type of data in voxels: DT_* code    */

  float dx ;                    /*!< grid spacings      */
  float dy ;                    /*!< grid spacings      */
  float dz ;                    /*!< grid spacings      */
  float dt ;                    /*!< grid spacings      */
  float du ;                    /*!< grid spacings      */
  float dv ;                    /*!< grid spacings      */
  float dw ;                    /*!< grid spacings      */
  float pixdim[8] ;             /*!< pixdim[1]=dx, etc. */

  float scl_slope ;             /*!< scaling parameter - slope        */
  float scl_inter ;             /*!< scaling parameter - intercept    */

  float cal_min ;               /*!< calibration parameter, minimum   */
  float cal_max ;               /*!< calibration parameter, maximum   */

  int qform_code ;              /*!< codes for (x,y,z) space meaning  */
  int sform_code ;              /*!< codes for (x,y,z) space meaning  */

  int freq_dim  ;               /*!< indexes (1,2,3, or 0) for MRI    */
  int phase_dim ;               /*!< directions in dim[]/pixdim[]     */
  int slice_dim ;               /*!< directions in dim[]/pixdim[]     */

  int   slice_code  ;           /*!< code for slice timing pattern    */
  int   slice_start ;           /*!< index for start of slices        */
  int   slice_end   ;           /*!< index for end of slices          */
  float slice_duration ;        /*!< time between individual slices   */

  /*! quaternion transform parameters
    [when writing a dataset, these are used for qform, NOT qto_xyz]   */
  float quatern_b , quatern_c , quatern_d ,
        qoffset_x , qoffset_y , qoffset_z ,
        qfac      ;

  mat44 qto_xyz ;               /*!< qform: transform (i,j,k) to (x,y,z) */
  mat44 qto_ijk ;               /*!< qform: transform (x,y,z) to (i,j,k) */

  mat44 sto_xyz ;               /*!< sform: transform (i,j,k) to (x,y,z) */
  mat44 sto_ijk ;               /*!< sform: transform (x,y,z) to (i,j,k) */

  float toffset ;               /*!< time coordinate offset */

  int xyz_units  ;              /*!< dx,dy,dz units: NIFTI_UNITS_* code  */
  int time_units ;              /*!< dt       units: NIFTI_UNITS_* code  */

  int nifti_type ;              /*!< 0==ANALYZE, 1==NIFTI-1 (1 file),
                                                 2==NIFTI-1 (2 files),
                                                 3==NIFTI-ASCII (1 file) */
  int   intent_code ;           /*!< statistic type (or something)       */
  float intent_p1 ;             /*!< intent parameters                   */
  float intent_p2 ;             /*!< intent parameters                   */
  float intent_p3 ;             /*!< intent parameters                   */
  char  intent_name[16] ;       /*!< optional description of intent data */

  char descrip[80]  ;           /*!< optional text to describe dataset   */
  char aux_file[24] ;           /*!< auxiliary filename                  */

  char *fname ;                 /*!< header filename (.hdr or .nii)         */
  char *iname ;                 /*!< image filename  (.img or .nii)         */
  int   iname_offset ;          /*!< offset into iname where data starts    */
  int   swapsize ;              /*!< swap unit in image data (might be 0)   */
  int   byteorder ;             /*!< byte order on disk (MSB_ or LSB_FIRST) */
  void *data ;                  /*!< pointer to data: nbyper*nvox bytes     */

  int                num_ext ;  /*!< number of extensions in ext_list       */
  nifti1_extension * ext_list ; /*!< array of extension structs (with data) */
  analyze_75_orient_code analyze75_orient; /*!< for old analyze files, orient */

  // reserved impl bytes. Is not part of nifti format
  std::shared_ptr<zfmembuf> decompressed_memory_buffer;

} nifti1_image ;
#endif


/* struct for return from nifti_image_read_bricks() */
typedef struct {
  int       nbricks;    /* the number of allocated pointers in 'bricks' */
  size_t    bsize;      /* the length of each data block, in bytes      */
  void   ** bricks;     /* array of pointers to data blocks             */
} nifti1_brick_list;

#if RNIFTI_NIFTILIB_VERSION == 1
typedef nifti1_image        nifti_image;
typedef nifti1_brick_list   nifti_brick_list;
#endif

/*****************************************************************************/
/*------------------ NIfTI version of ANALYZE 7.5 structure -----------------*/

/* (based on fsliolib/dbh.h, but updated for version 7.5) */

#if RNIFTI_NIFTILIB_VERSION == 1
typedef struct {
       /* header info fields - describes the header    overlap with NIfTI */
       /*                                              ------------------ */
       int sizeof_hdr;                  /* 0 + 4        same              */
       char data_type[10];              /* 4 + 10       same              */
       char db_name[18];                /* 14 + 18      same              */
       int extents;                     /* 32 + 4       same              */
       short int session_error;         /* 36 + 2       same              */
       char regular;                    /* 38 + 1       same              */
       char hkey_un0;                   /* 39 + 1                40 bytes */

       /* image dimension fields - describes image sizes */
       short int dim[8];                /* 0 + 16       same              */
       short int unused8;               /* 16 + 2       intent_p1...      */
       short int unused9;               /* 18 + 2         ...             */
       short int unused10;              /* 20 + 2       intent_p2...      */
       short int unused11;              /* 22 + 2         ...             */
       short int unused12;              /* 24 + 2       intent_p3...      */
       short int unused13;              /* 26 + 2         ...             */
       short int unused14;              /* 28 + 2       intent_code       */
       short int datatype;              /* 30 + 2       same              */
       short int bitpix;                /* 32 + 2       same              */
       short int dim_un0;               /* 34 + 2       slice_start       */
       float pixdim[8];                 /* 36 + 32      same              */

       float vox_offset;                /* 68 + 4       same              */
       float funused1;                  /* 72 + 4       scl_slope         */
       float funused2;                  /* 76 + 4       scl_inter         */
       float funused3;                  /* 80 + 4       slice_end,        */
                                                     /* slice_code,       */
                                                     /* xyzt_units        */
       float cal_max;                   /* 84 + 4       same              */
       float cal_min;                   /* 88 + 4       same              */
       float compressed;                /* 92 + 4       slice_duration    */
       float verified;                  /* 96 + 4       toffset           */
       int glmax,glmin;                 /* 100 + 8              108 bytes */

       /* data history fields - optional */
       char descrip[80];                /* 0 + 80       same              */
       char aux_file[24];               /* 80 + 24      same              */
       char orient;                     /* 104 + 1      NO GOOD OVERLAP   */
       char originator[10];             /* 105 + 10     FROM HERE DOWN... */
       char generated[10];              /* 115 + 10                       */
       char scannum[10];                /* 125 + 10                       */
       char patient_id[10];             /* 135 + 10                       */
       char exp_date[10];               /* 145 + 10                       */
       char exp_time[10];               /* 155 + 10                       */
       char hist_un0[3];                /* 165 + 3                        */
       int views;                       /* 168 + 4                        */
       int vols_added;                  /* 172 + 4                        */
       int start_field;                 /* 176 + 4                        */
       int field_skip;                  /* 180 + 4                        */
       int omax, omin;                  /* 184 + 8                        */
       int smax, smin;                  /* 192 + 8              200 bytes */
} nifti_analyze75;                                   /* total:  348 bytes */
#endif

size_t nifti_image_impl_reserved_bytes_offset();

/*****************************************************************************/
/*--------------- Prototypes of functions defined in this file --------------*/

char const * nifti_datatype_string   ( int dt ) ;
char const *nifti_units_string      ( int uu ) ;
char const *nifti_intent_string     ( int ii ) ;
char const *nifti_xform_string      ( int xx ) ;
char const *nifti_slice_string      ( int ss ) ;
char const *nifti_orientation_string( int ii ) ;

int   nifti_is_inttype( int dt ) ;

mat44 nifti_mat44_inverse( mat44 R ) ;

mat33 nifti_mat33_inverse( mat33 R ) ;
mat33 nifti_mat33_polar  ( mat33 A ) ;
float nifti_mat33_rownorm( mat33 A ) ;
float nifti_mat33_colnorm( mat33 A ) ;
float nifti_mat33_determ ( mat33 R ) ;
mat33 nifti_mat33_mul    ( mat33 A , mat33 B ) ;

#if RNIFTI_NIFTILIB_VERSION == 1
void  nifti_swap_2bytes ( size_t n , void *ar ) ;
void  nifti_swap_4bytes ( size_t n , void *ar ) ;
void  nifti_swap_8bytes ( size_t n , void *ar ) ;
void  nifti_swap_16bytes( size_t n , void *ar ) ;
void  nifti_swap_Nbytes ( size_t n , int siz , void *ar ) ;
#endif

int    nifti_datatype_is_valid   (int dtype, int for_nifti);
int    nifti_datatype_from_string(const char * name);
const char * nifti_datatype_to_string  (int dtype);

int   nifti_get_filesize( const char *pathname ) ;
#if RNIFTI_NIFTILIB_VERSION == 1
void  swap_nifti_header ( struct nifti_1_header *h , int is_nifti ) ;
#endif
void  old_swap_nifti_header( struct nifti_1_header *h , int is_nifti );
#if RNIFTI_NIFTILIB_VERSION == 1
int   nifti_swap_as_analyze( nifti_analyze75 *h );
#endif


/* main read/write routines */

nifti_image *nifti_image_read_bricks(const char *hname , int nbricks,
                                     const int *blist, nifti_brick_list * NBL);
int          nifti_image_load_bricks(nifti_image *nim , int nbricks,
                                     const int *blist, nifti_brick_list * NBL);
void         nifti_free_NBL( nifti_brick_list * NBL );

nifti_image *nifti_image_read    ( const char *hname , int read_data ) ;
nifti_image *nifti_image_mem_read( const uint8_t *data , const size_t size , int gz, const size_t estimated_size ) ;

int          nifti_image_load    ( nifti_image *nim ) ;
void         nifti_image_unload  ( nifti_image *nim ) ;
void         nifti_image_free    ( nifti_image *nim ) ;
int          nifti_image_is_data_owner(const nifti_image* nim) ;

int          nifti_read_collapsed_image( nifti_image * nim, const int dims [8],
                                         void ** data );

int          nifti_read_subregion_image( nifti_image * nim,
                                         const int *start_index, const int *region_size,
                                         void ** data );

void         nifti_image_write   ( nifti_image * nim ) ;
void         nifti_image_write_bricks(nifti_image * nim,
                                      const nifti_brick_list * NBL);
void         nifti_image_infodump( const nifti_image * nim ) ;

void         nifti_disp_lib_hist( void ) ;     /* to display library history */
void         nifti_disp_lib_version( void ) ;  /* to display library version */
int          nifti_disp_matrix_orient( const char * mesg, mat44 mat );
int          nifti_disp_type_list( int which );


char *       nifti_image_to_ascii  ( const nifti_image * nim ) ;
nifti_image *nifti_image_from_ascii( const char * str, int * bytes_read ) ;

size_t       nifti_get_volsize(const nifti_image *nim) ;

/* basic file operations */
int    nifti_set_filenames(nifti_image * nim, const char * prefix, int check,
                           int set_byte_order);
char * nifti_makehdrname  (const char * prefix, int nifti_type, int check,
                           int comp);
char * nifti_makeimgname  (const char * prefix, int nifti_type, int check,
                           int comp);
int    is_nifti_file      (const char *hname);
char * nifti_find_file_extension(const char * name);
int    nifti_is_complete_filename(const char* fname);
int    nifti_validfilename(const char* fname);

int    disp_nifti_1_header(const char * info, const nifti_1_header * hp ) ;
void   nifti_set_debug_level( int level ) ;
void   nifti_set_skip_blank_ext( int skip ) ;
void   nifti_set_allow_upper_fext( int allow ) ;

int    valid_nifti_brick_list(nifti_image * nim , int nbricks,
                              const int * blist, int disp_error);

/* znzFile operations */
znzFile nifti_image_open(const char * hname, const char * opts, nifti_image ** nim);
znzFile nifti_image_write_hdr_img(nifti_image *nim, int write_data,
                                  const char* opts);
znzFile nifti_image_write_hdr_img2( nifti_image *nim , int write_opts ,
               const char* opts, znzFile imgfile, const nifti_brick_list * NBL);
size_t  nifti_read_buffer(znzFile fp, void* dataptr, size_t ntot, nifti_image *nim);
size_t  nifti_assign_buffer(znzFile fp, void** dataptr, size_t ntot, nifti_image *nim);
int     nifti_write_all_data(znzFile fp, nifti_image * nim,
                             const nifti_brick_list * NBL);
size_t  nifti_write_buffer(znzFile fp, const void * buffer, size_t numbytes);
nifti_image *nifti_read_ascii_image(znzFile fp, char *fname, int flen,
                         int read_data);
znzFile nifti_write_ascii_image(nifti_image *nim, const nifti_brick_list * NBL,
                         const char * opts, int write_data, int leave_open);


void nifti_datatype_sizes( int datatype , int *nbyper, int *swapsize ) ;

void nifti_mat44_to_quatern( mat44 R ,
                             float *qb, float *qc, float *qd,
                             float *qx, float *qy, float *qz,
                             float *dx, float *dy, float *dz, float *qfac ) ;

mat44 nifti_quatern_to_mat44( float qb, float qc, float qd,
                              float qx, float qy, float qz,
                              float dx, float dy, float dz, float qfac );

mat44 nifti_make_orthog_mat44( float r11, float r12, float r13 ,
                               float r21, float r22, float r23 ,
                               float r31, float r32, float r33  ) ;

int nifti_short_order(void) ;              /* CPU byte order */


/* Orientation codes that might be returned from nifti_mat44_to_orientation().*/

#define NIFTI_L2R  1    /* Left to Right         */
#define NIFTI_R2L  2    /* Right to Left         */
#define NIFTI_P2A  3    /* Posterior to Anterior */
#define NIFTI_A2P  4    /* Anterior to Posterior */
#define NIFTI_I2S  5    /* Inferior to Superior  */
#define NIFTI_S2I  6    /* Superior to Inferior  */

void nifti_mat44_to_orientation( mat44 R , int *icod, int *jcod, int *kcod ) ;

/*--------------------- Low level IO routines ------------------------------*/

char * nifti_findhdrname (const char* fname);
char * nifti_findimgname (const char* fname , int nifti_type);
int    nifti_is_gzfile   (const char* fname);

char * nifti_makebasename(const char* fname);


/* other routines */
struct nifti_1_header   nifti_convert_nim2nhdr(const nifti_image* nim);
nifti_1_header * nifti_make_new_header(const int arg_dims[], int arg_dtype);
nifti_1_header * nifti_read_header(const char *hname, int *swapped, int check);
nifti_image    * nifti_copy_nim_info(const nifti_image * src);
nifti_image    * nifti_make_new_nim(const int dims[], int datatype,
                                                      int data_fill);
nifti_image    * nifti_simple_init_nim(void);
nifti_image    * nifti_convert_nhdr2nim(struct nifti_1_header nhdr,
                                        const char * fname);

int    nifti_hdr_looks_good        (const nifti_1_header * hdr);
int    nifti_is_valid_datatype     (int dtype);
int    nifti_is_valid_ecode        (int ecode);
int    nifti_nim_is_valid          (nifti_image * nim, int complain);
int    nifti_nim_has_valid_dims    (nifti_image * nim, int complain);
int    is_valid_nifti_type         (int nifti_type);
int    nifti_test_datatype_sizes   (int verb);
int    nifti_type_and_names_match  (nifti_image * nim, int show_warn);
int    nifti_update_dims_from_array(nifti_image * nim);
void   nifti_set_iname_offset      (nifti_image *nim);
int    nifti_set_type_from_names   (nifti_image * nim);
int    nifti_add_extension(nifti_image * nim, const char * data, int len,
                           int ecode );
int    nifti_compiled_with_zlib    (void);
int    nifti_copy_extensions (nifti_image *nim_dest,const nifti_image *nim_src);
int    nifti_free_extensions (nifti_image *nim);
int  * nifti_get_intlist     (int nvals , const char *str);
char * nifti_strdup          (const char *str);
int    valid_nifti_extensions(const nifti_image *nim);


/*-------------------- Some C convenience macros ----------------------------*/

/* NIfTI-1.1 extension codes:
   see http://nifti.nimh.nih.gov/nifti-1/documentation/faq#Q21 */

#define NIFTI_ECODE_IGNORE           0  /* changed from UNKNOWN, 29 June 2005 */

#define NIFTI_ECODE_DICOM            2  /* intended for raw DICOM attributes  */

#define NIFTI_ECODE_AFNI             4  /* Robert W Cox: rwcox@nih.gov
                                           https://afni.nimh.nih.gov/afni     */

#define NIFTI_ECODE_COMMENT          6  /* plain ASCII text only              */

#define NIFTI_ECODE_XCEDE            8  /* David B Keator: dbkeator@uci.edu
                                           http://www.nbirn.net/Resources
                                                /Users/Applications/
                                                /xcede/index.htm              */

#define NIFTI_ECODE_JIMDIMINFO      10  /* Mark A Horsfield:
                                           mah5@leicester.ac.uk
                                           http://someplace/something         */

#define NIFTI_ECODE_WORKFLOW_FWDS   12  /* Kate Fissell: fissell@pitt.edu
                                           http://kraepelin.wpic.pitt.edu
                                            /~fissell/NIFTI_ECODE_WORKFLOW_FWDS
                                            /NIFTI_ECODE_WORKFLOW_FWDS.html   */

#define NIFTI_ECODE_FREESURFER      14  /* http://surfer.nmr.mgh.harvard.edu  */

#define NIFTI_ECODE_PYPICKLE        16  /* embedded Python objects
                                           http://niftilib.sourceforge.net
                                                 /pynifti                     */

        /* LONI MiND codes: http://www.loni.ucla.edu/twiki/bin/view/Main/MiND */
#define NIFTI_ECODE_MIND_IDENT      18  /* Vishal Patel: vishal.patel@ucla.edu*/
#define NIFTI_ECODE_B_VALUE         20
#define NIFTI_ECODE_SPHERICAL_DIRECTION 22
#define NIFTI_ECODE_DT_COMPONENT    24
#define NIFTI_ECODE_SHC_DEGREEORDER 26  /* end LONI MiND codes                */

#define NIFTI_ECODE_VOXBO           28  /* Dan Kimberg: www.voxbo.org         */

#define NIFTI_ECODE_CARET           30  /* John Harwell: john@brainvis.wustl.edu
                                           http://brainvis.wustl.edu/wiki
                                             /index.php/Caret:Documentation
                                             :CaretNiftiExtension             */

#define NIFTI_ECODE_CIFTI           32  /* CIFTI-2_Main_FINAL_1March2014.pdf */

#define NIFTI_ECODE_VARIABLE_FRAME_TIMING 34

/* 36 is currently unassigned, waiting on NIFTI_ECODE_AGILENT_PROCPAR */

#define NIFTI_ECODE_EVAL            38  /* Munster University Hospital */

/* http://www.mathworks.com/matlabcentral/fileexchange/42997-dicom-to-nifti-converter */
#define NIFTI_ECODE_MATLAB          40  /* MATLAB extension */

/* Quantiphyse extension
   https://quantiphyse.readthedocs.io/en/latest/advanced/nifti_extension.html*/
#define NIFTI_ECODE_QUANTIPHYSE     42  /* Quantiphyse extension */

/* Magnetic Resonance Spectroscopy (MRS)
   link to come... */
#define NIFTI_ECODE_MRS             44  /* MRS extension */

#define NIFTI_MAX_ECODE             44  /******* maximum extension code *******/

/* nifti_type file codes */
#if RNIFTI_NIFTILIB_VERSION == 1
#define NIFTI_FTYPE_ANALYZE   0
#define NIFTI_FTYPE_NIFTI1_1  1
#define NIFTI_FTYPE_NIFTI1_2  2
#define NIFTI_FTYPE_ASCII     3
#define NIFTI_MAX_FTYPE       3    /* this should match the maximum code */
#endif

/*------------------------------------------------------------------------*/
/*-- the rest of these apply only to nifti1_io.c, check for _NIFTI1_IO_C_ */
/*                                                    Feb 9, 2005 [rickr] */
#ifdef _NIFTI1_IO_C_

typedef struct {
    int debug;               /*!< debug level for status reports  */
    int skip_blank_ext;      /*!< skip extender if no extensions  */
    int allow_upper_fext;    /*!< allow uppercase file extensions */
} nifti_global_options;

typedef struct {
    int    type;           /* should match the NIFTI_TYPE_ #define */
    int    nbyper;         /* bytes per value, matches nifti_image */
    int    swapsize;       /* bytes per swap piece, matches nifti_image */
    char const * const name;           /* text string to match #define */
} nifti_type_ele;

#undef  LNI_FERR /* local nifti file error, to be compact and repetative */
#ifdef USING_R
#define LNI_FERR(func,msg,file)                                      \
            Rf_warning("%s: %s '%s'\n",func,msg,file)
#else
#define LNI_FERR(func,msg,file)                                      \
            Rc_fprintf_stderr("** ERROR (%s): %s '%s'\n",func,msg,file)
#endif

#undef  swap_2
#undef  swap_4
#define swap_2(s) nifti_swap_2bytes(1,&(s)) /* s: 2-byte short; swap in place */
#define swap_4(v) nifti_swap_4bytes(1,&(v)) /* v: 4-byte value; swap in place */

                        /***** isfinite() is a C99 macro, which is
                               present in many C implementations already *****/

#undef IS_GOOD_FLOAT
#undef FIXED_FLOAT

#ifdef isfinite       /* use isfinite() to check floats/doubles for goodness */
#  define IS_GOOD_FLOAT(x) isfinite(x)       /* check if x is a "good" float */
#  define FIXED_FLOAT(x)   (isfinite(x) ? (x) : 0)           /* fixed if bad */
#else
#  define IS_GOOD_FLOAT(x) 1                               /* don't check it */
#  define FIXED_FLOAT(x)   (x)                               /* don't fix it */
#endif

#undef  ASSIF                                 /* assign v to *p, if possible */
#define ASSIF(p,v) if( (p)!=NULL ) *(p) = (v)

#undef  MSB_FIRST
#undef  LSB_FIRST
#undef  REVERSE_ORDER
#define LSB_FIRST 1
#define MSB_FIRST 2
#define REVERSE_ORDER(x) (3-(x))    /* convert MSB_FIRST <--> LSB_FIRST */

#define LNI_MAX_NIA_EXT_LEN 100000  /* consider a longer extension invalid */

#endif  /* _NIFTI1_IO_C_ section */
/*------------------------------------------------------------------------*/

#endif /* _NIFTI_IO_HEADER_ */
