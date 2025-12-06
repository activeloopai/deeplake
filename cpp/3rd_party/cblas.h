#ifndef CBLAS_H
#define CBLAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */

#define CBLAS_INDEX size_t

#ifndef BFLOAT16
#include <stdint.h>
typedef uint16_t bfloat16;
#endif


typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;
	
float  cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy);
double cblas_dsdot (const int n, const float *x, const int incx, const float *y, const int incy);
float  cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy);
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);

void  cblas_cdotu_sub(const int n, const void  *x, const int incx, const void  *y, const int incy, void  *ret);
void  cblas_cdotc_sub(const int n, const void  *x, const int incx, const void  *y, const int incy, void  *ret);
void  cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *ret);
void  cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *ret);

float  cblas_sasum (const int n, const float  *x, const int incx);
double cblas_dasum (const int n, const double *x, const int incx);
float  cblas_scasum(const int n, const void  *x, const int incx);
double cblas_dzasum(const int n, const void *x, const int incx);

float  cblas_ssum (const int n, const float  *x, const int incx);
double cblas_dsum (const int n, const double *x, const int incx);
float  cblas_scsum(const int n, const void  *x, const int incx);
double cblas_dzsum(const int n, const void *x, const int incx);

float  cblas_snrm2 (const int N, const float  *X, const int incX);
double cblas_dnrm2 (const int N, const double *X, const int incX);
float  cblas_scnrm2(const int N, const void  *X, const int incX);
double cblas_dznrm2(const int N, const void *X, const int incX);

CBLAS_INDEX cblas_isamax(const int n, const float  *x, const int incx);
CBLAS_INDEX cblas_idamax(const int n, const double *x, const int incx);
CBLAS_INDEX cblas_icamax(const int n, const void  *x, const int incx);
CBLAS_INDEX cblas_izamax(const int n, const void *x, const int incx);

CBLAS_INDEX cblas_isamin(const int n, const float  *x, const int incx);
CBLAS_INDEX cblas_idamin(const int n, const double *x, const int incx);
CBLAS_INDEX cblas_icamin(const int n, const void  *x, const int incx);
CBLAS_INDEX cblas_izamin(const int n, const void *x, const int incx);

CBLAS_INDEX cblas_ismax(const int n, const float  *x, const int incx);
CBLAS_INDEX cblas_idmax(const int n, const double *x, const int incx);
CBLAS_INDEX cblas_icmax(const int n, const void  *x, const int incx);
CBLAS_INDEX cblas_izmax(const int n, const void *x, const int incx);

CBLAS_INDEX cblas_ismin(const int n, const float  *x, const int incx);
CBLAS_INDEX cblas_idmin(const int n, const double *x, const int incx);
CBLAS_INDEX cblas_icmin(const int n, const void  *x, const int incx);
CBLAS_INDEX cblas_izmin(const int n, const void *x, const int incx);

void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy);
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
void cblas_caxpy(const int n, const void *alpha, const void *x, const int incx, void *y, const int incy);
void cblas_zaxpy(const int n, const void *alpha, const void *x, const int incx, void *y, const int incy);

void cblas_scopy(const int n, const float *x, const int incx, float *y, const int incy);
void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy);
void cblas_ccopy(const int n, const void *x, const int incx, void *y, const int incy);
void cblas_zcopy(const int n, const void *x, const int incx, void *y, const int incy);

void cblas_sswap(const int n, float *x, const int incx, float *y, const int incy);
void cblas_dswap(const int n, double *x, const int incx, double *y, const int incy);
void cblas_cswap(const int n, void *x, const int incx, void *y, const int incy);
void cblas_zswap(const int n, void *x, const int incx, void *y, const int incy);

void cblas_srot(const int N, float *X, const int incX, float *Y, const int incY, const float c, const float s);
void cblas_drot(const int N, double *X, const int incX, double *Y, const int incY, const double c, const double  s);
void cblas_csrot(const int n, const void *x, const int incx, void *y, const int incY, const float c, const float s);
void cblas_zdrot(const int n, const void *x, const int incx, void *y, const int incY, const double c, const double s);

void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_crotg(void *a, void *b, float *c, void *s);
void cblas_zrotg(void *a, void *b, double *c, void *s);


void cblas_srotm(const int N, float *X, const int incX, float *Y, const int incY, const float *P);
void cblas_drotm(const int N, double *X, const int incX, double *Y, const int incY, const double *P);

void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);

void cblas_sscal(const int N, const float alpha, float *X, const int incX);
void cblas_dscal(const int N, const double alpha, double *X, const int incX);
void cblas_cscal(const int N, const void *alpha, void *X, const int incX);
void cblas_zscal(const int N, const void *alpha, void *X, const int incX);
void cblas_csscal(const int N, const float alpha, void *X, const int incX);
void cblas_zdscal(const int N, const double alpha, void *X, const int incX);

void cblas_sgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const float alpha, const float  *a, const int lda,  const float  *x, const int incx,  const float beta,  float  *y, const int incy);
void cblas_dgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy);
void cblas_cgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const void *alpha, const void  *a, const int lda,  const void  *x, const int incx,  const void *beta,  void  *y, const int incy);
void cblas_zgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const void *alpha, const void  *a, const int lda,  const void  *x, const int incx,  const void *beta,  void  *y, const int incy);

void cblas_sger (const enum CBLAS_ORDER order, const int M, const int N, const float   alpha, const float  *X, const int incX, const float  *Y, const int incY, float  *A, const int lda);
void cblas_dger (const enum CBLAS_ORDER order, const int M, const int N, const double  alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda);
void cblas_cgeru(const enum CBLAS_ORDER order, const int M, const int N, const void  *alpha, const void  *X, const int incX, const void  *Y, const int incY, void  *A, const int lda);
void cblas_cgerc(const enum CBLAS_ORDER order, const int M, const int N, const void  *alpha, const void  *X, const int incX, const void  *Y, const int incY, void  *A, const int lda);
void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N, const void *alpha, const void *X, const int incX, const void *Y, const int incY, void *A, const int lda);
void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N, const void *alpha, const void *X, const int incX, const void *Y, const int incY, void *A, const int lda);

void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX);
void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX);
void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const void *A, const int lda, void *X, const int incX);
void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const void *A, const int lda, void *X, const int incX);

void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX);
void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX);
void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const void *A, const int lda, void *X, const int incX);
void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const void *A, const int lda, void *X, const int incX);

void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, float *A, const int lda);
void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, double *A, const int lda);
void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const void *X, const int incX, void *A, const int lda);
void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const void *X, const int incX, void *A, const int lda);

void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,const int N, const float alpha, const float *X,
                const int incX, const float *Y, const int incY, float *A, const int lda);
void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X,
                const int incX, const double *Y, const int incY, double *A, const int lda);
void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *A, const int lda);
void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const void *alpha, const void *X, const int incX,
                const void *Y, const int incY, void *A, const int lda);

void cblas_sgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
void cblas_cgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha, const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);
void cblas_zgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const void *alpha, const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);

void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K, const float alpha, const float *A,
                 const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);


void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, float *X, const int incX);
void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, double *X, const int incX);
void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, void *X, const int incX);
void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, void *X, const int incX);

void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, float *X, const int incX);
void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, double *X, const int incX);
void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, void *X, const int incX);
void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const void *A, const int lda, void *X, const int incX);

void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);
void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX);
void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX);

void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);
void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX);
void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const void *Ap, void *X, const int incX);

void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);
void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const void *alpha, const void *A,
                 const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);


void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *Ap,
                 const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *Ap,
                 const double *X, const int incX, const double beta, double *Y, const int incY);

void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, float *Ap);
void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, double *Ap);

void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const void *X, const int incX, void *A);
void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const void *X,const int incX, void *A);

void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, const float *Y, const int incY, float *A);
void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, const double *Y, const int incY, double *A);
void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const void *alpha, const void *X, const int incX, const void *Y, const int incY, void *Ap);
void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const void *alpha, const void *X, const int incX, const void *Y, const int incY, void *Ap);

void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);
void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);

void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
		 const void *alpha, const void *Ap, const void *X, const int incX, const void *beta, void *Y, const int incY);
void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
		 const void *alpha, const void *Ap, const void *X, const int incX, const void *beta, void *Y, const int incY);

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
void cblas_cgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
void cblas_zgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);


void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);

void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const float alpha, const float *A, const int lda, const float beta, float *C, const int ldc);
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc);
void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const void *alpha, const void *A, const int lda, const void *beta, void *C, const int ldc);
void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const void *alpha, const void *A, const int lda, const void *beta, void *C, const int ldc);

void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);

void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb);
void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);
void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const void *alpha, const void *A, const int lda, void *B, const int ldb);
void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const void *alpha, const void *A, const int lda, void *B, const int ldb);

void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb);
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);
void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const void *alpha, const void *A, const int lda, void *B, const int ldb);
void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const void *alpha, const void *A, const int lda, void *B, const int ldb);

void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);

void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const void *A, const int lda, const float beta, void *C, const int ldc);
void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const void *A, const int lda, const double beta, void *C, const int ldc);

void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda, const void *B, const int ldb, const float beta, void *C, const int ldc);
void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const void *alpha, const void *A, const int lda, const void *B, const int ldb, const double beta, void *C, const int ldc);

void cblas_xerbla(int p, const char *rout, const char *form, ...);

/*** BLAS extensions ***/

void cblas_saxpby(const int n, const float alpha, const float *x, const int incx,const float beta, float *y, const int incy);

void cblas_daxpby(const int n, const double alpha, const double *x, const int incx,const double beta, double *y, const int incy);

void cblas_caxpby(const int n, const void *alpha, const void *x, const int incx,const void *beta, void *y, const int incy);

void cblas_zaxpby(const int n, const void *alpha, const void *x, const int incx,const void *beta, void *y, const int incy);

void cblas_somatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float calpha, const float *a, 
		     const int clda, float *b, const int cldb); 
void cblas_domatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double calpha, const double *a,
		     const int clda, double *b, const int cldb); 
void cblas_comatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float* calpha, const float* a, 
		     const int clda, float*b, const int cldb); 
void cblas_zomatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double* calpha, const double* a, 
		     const int clda,  double *b, const int cldb); 

void cblas_simatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float calpha, float *a, 
		     const int clda, const int cldb); 
void cblas_dimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double calpha, double *a,
		     const int clda, const int cldb); 
void cblas_cimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float* calpha, float* a, 
		     const int clda, const int cldb); 
void cblas_zimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double* calpha, double* a, 
		     const int clda, const int cldb); 

void cblas_sgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const float calpha, float *a, const int clda, const float cbeta, 
		  float *c, const int cldc); 
void cblas_dgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const double calpha, double *a, const int clda, const double cbeta, 
		  double *c, const int cldc); 
void cblas_cgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const float *calpha, float *a, const int clda, const float *cbeta, 
		  float *c, const int cldc); 
void cblas_zgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const double *calpha, double *a, const int clda, const double *cbeta, 
		  double *c, const int cldc); 

/*** BFLOAT16 and INT8 extensions ***/
/* convert float array to BFLOAT16 array by rounding */
void   cblas_sbstobf16(const int n, const float  *in, const int incin, bfloat16 *out, const int incout);
/* convert double array to BFLOAT16 array by rounding */
void   cblas_sbdtobf16(const int n, const double *in, const int incin, bfloat16 *out, const int incout);
/* convert BFLOAT16 array to float array */
void   cblas_sbf16tos(const int n, const bfloat16 *in, const int incin, float  *out, const int incout);
/* convert BFLOAT16 array to double array */
void   cblas_dbf16tod(const int n, const bfloat16 *in, const int incin, double *out, const int incout);
/* dot production of BFLOAT16 input arrays, and output as float */
float  cblas_sbdot(const int n, const bfloat16 *x, const int incx, const bfloat16 *y, const int incy);
void   cblas_sbgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n, const float alpha, const bfloat16 *a, const int lda, const bfloat16 *x, const int incx, const float beta, float *y, const int incy);

void   cblas_sbgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		    const float alpha, const bfloat16 *A, const int lda, const bfloat16 *B, const int ldb, const float beta, float *C, const int ldc);
#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif
