#ifndef _NIFTI2_IMAGE_H_
#define _NIFTI2_IMAGE_H_

#include <inttypes.h>

// This is repetitious and inelegant, but a definition for nifti2_image is needed to allow
// conversion to/from nifti1_image. This is a straight copy of the relevant parts of nifti2_io.h.
#if RNIFTI_NIFTILIB_VERSION == 1

typedef struct {                   /** 4x4 matrix struct (double) **/
  double m[4][4] ;
} nifti_dmat44 ;


typedef struct {                /*!< Image storage struct **/

  int64_t ndim ;                /*!< last dimension greater than 1 (1..7) */
  int64_t nx ;                  /*!< dimensions of grid array             */
  int64_t ny ;                  /*!< dimensions of grid array             */
  int64_t nz ;                  /*!< dimensions of grid array             */
  int64_t nt ;                  /*!< dimensions of grid array             */
  int64_t nu ;                  /*!< dimensions of grid array             */
  int64_t nv ;                  /*!< dimensions of grid array             */
  int64_t nw ;                  /*!< dimensions of grid array             */
  int64_t dim[8] ;              /*!< dim[0]=ndim, dim[1]=nx, etc.         */
  int64_t nvox ;                /*!< number of voxels = nx*ny*nz*...*nw   */
  int nbyper ;                  /*!< bytes per voxel, matches datatype    */
  int datatype ;                /*!< type of data in voxels: DT_* code    */

  double dx ;                   /*!< grid spacings      */
  double dy ;                   /*!< grid spacings      */
  double dz ;                   /*!< grid spacings      */
  double dt ;                   /*!< grid spacings      */
  double du ;                   /*!< grid spacings      */
  double dv ;                   /*!< grid spacings      */
  double dw ;                   /*!< grid spacings      */
  double pixdim[8] ;            /*!< pixdim[1]=dx, etc. */

  double scl_slope ;            /*!< scaling parameter - slope        */
  double scl_inter ;            /*!< scaling parameter - intercept    */

  double cal_min ;              /*!< calibration parameter, minimum   */
  double cal_max ;              /*!< calibration parameter, maximum   */

  int qform_code ;              /*!< codes for (x,y,z) space meaning  */
  int sform_code ;              /*!< codes for (x,y,z) space meaning  */

  int freq_dim  ;               /*!< indexes (1,2,3, or 0) for MRI    */
  int phase_dim ;               /*!< directions in dim[]/pixdim[]     */
  int slice_dim ;               /*!< directions in dim[]/pixdim[]     */

  int     slice_code  ;         /*!< code for slice timing pattern    */
  int64_t slice_start ;         /*!< index for start of slices        */
  int64_t slice_end   ;         /*!< index for end of slices          */
  double  slice_duration ;      /*!< time between individual slices   */

  /*! quaternion transform parameters
    [when writing a dataset, these are used for qform, NOT qto_xyz]   */
  double quatern_b , quatern_c , quatern_d ,
         qoffset_x , qoffset_y , qoffset_z ,
         qfac      ;

  nifti_dmat44 qto_xyz ;        /*!< qform: transform (i,j,k) to (x,y,z) */
  nifti_dmat44 qto_ijk ;        /*!< qform: transform (x,y,z) to (i,j,k) */

  nifti_dmat44 sto_xyz ;        /*!< sform: transform (i,j,k) to (x,y,z) */
  nifti_dmat44 sto_ijk ;        /*!< sform: transform (x,y,z) to (i,j,k) */

  double toffset ;              /*!< time coordinate offset */

  int xyz_units  ;              /*!< dx,dy,dz units: NIFTI_UNITS_* code  */
  int time_units ;              /*!< dt       units: NIFTI_UNITS_* code  */

  int nifti_type ;              /*!< see NIFTI_FTYPE_* codes, below:
                                        0==ANALYZE,
                                        1==NIFTI-1     (1 file),
                                        2==NIFTI-1     (2 files),
                                        3==NIFTI-ASCII (1 file)
                                        4==NIFTI-2     (1 file),
                                        5==NIFTI-2     (2 files) */

  int    intent_code ;          /*!< statistic type (or something)       */
  double intent_p1 ;            /*!< intent parameters                   */
  double intent_p2 ;            /*!< intent parameters                   */
  double intent_p3 ;            /*!< intent parameters                   */
  char   intent_name[16] ;      /*!< optional description of intent data */

  char descrip[80]  ;           /*!< optional text to describe dataset   */
  char aux_file[24] ;           /*!< auxiliary filename                  */

  char *fname ;                 /*!< header filename (.hdr or .nii)         */
  char *iname ;                 /*!< image filename  (.img or .nii)         */
  int64_t iname_offset ;        /*!< offset into iname where data starts    */
  int   swapsize ;              /*!< swap unit in image data (might be 0)   */
  int   byteorder ;             /*!< byte order on disk (MSB_ or LSB_FIRST) */
  void *data ;                  /*!< pointer to data: nbyper*nvox bytes     */

  int                num_ext ;  /*!< number of extensions in ext_list       */
  nifti1_extension * ext_list ; /*!< array of extension structs (with data) */
  analyze_75_orient_code analyze75_orient; /*!< for old analyze files, orient */

  // reserved impl bytes. Is not part of nifti format.
  std::shared_ptr<zfmembuf> decompressed_memory_buffer;

} nifti2_image ;

#endif // RNIFTI_NIFTILIB_VERSION

#endif
