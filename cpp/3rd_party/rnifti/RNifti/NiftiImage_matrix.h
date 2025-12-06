#ifndef _NIFTI_IMAGE_MATRIX_H_
#define _NIFTI_IMAGE_MATRIX_H_

template <>
inline SquareMatrix<mat33,float,3> SquareMatrix<mat33,float,3>::inverse () const
{
    return SquareMatrix<mat33,float,3>(nifti_mat33_inverse(*niftiPointer()));
}

template <>
inline SquareMatrix<mat33,float,3> SquareMatrix<mat33,float,3>::polar () const
{
    return SquareMatrix<mat33,float,3>(nifti_mat33_polar(*niftiPointer()));
}

template <>
inline float SquareMatrix<mat33,float,3>::colnorm () const
{
    return nifti_mat33_colnorm(*niftiPointer());
}

template <>
inline float SquareMatrix<mat33,float,3>::rownorm () const
{
    return nifti_mat33_rownorm(*niftiPointer());
}

template <>
inline float SquareMatrix<mat33,float,3>::determ () const
{
    return nifti_mat33_determ(*niftiPointer());
}

template <>
inline SquareMatrix<mat33,float,3> SquareMatrix<mat33,float,3>::multiply (const SquareMatrix<mat33,float,3> &other) const
{
    return SquareMatrix<mat33,float,3>(nifti_mat33_mul(*niftiPointer(), *other.niftiPointer()));
}

template <>
inline SquareMatrix<mat44,float,4> SquareMatrix<mat44,float,4>::inverse () const
{
    return SquareMatrix<mat44,float,4>(nifti_mat44_inverse(*niftiPointer()));
}

#if RNIFTI_NIFTILIB_VERSION == 1

// NB: niftilib v1 does not define nifti_mat44_mul
template <>
inline SquareMatrix<mat44,float,4> SquareMatrix<mat44,float,4>::multiply (const SquareMatrix<mat44,float,4> &other) const
{
    SquareMatrix<mat44,float,4> result;
    for (int i=0; i < 4; i++)
    {
        for (int j=0; j < 4; j++)
        {
            result(i,j) = 0.0;
            for (int k=0; k<4; k++)
                result(i,j) += (*this)(i,k) * other(k,j);
        }
    }
    return result;
}

#elif RNIFTI_NIFTILIB_VERSION == 2

template <>
inline SquareMatrix<nifti_dmat33,double,3> SquareMatrix<nifti_dmat33,double,3>::inverse () const
{
    return SquareMatrix<nifti_dmat33,double,3>(nifti_dmat33_inverse(*niftiPointer()));
}

template <>
inline SquareMatrix<nifti_dmat33,double,3> SquareMatrix<nifti_dmat33,double,3>::polar () const
{
    return SquareMatrix<nifti_dmat33,double,3>(nifti_dmat33_polar(*niftiPointer()));
}

template <>
inline double SquareMatrix<nifti_dmat33,double,3>::colnorm () const
{
    return nifti_dmat33_colnorm(*niftiPointer());
}

template <>
inline double SquareMatrix<nifti_dmat33,double,3>::rownorm () const
{
    return nifti_dmat33_rownorm(*niftiPointer());
}

template <>
inline double SquareMatrix<nifti_dmat33,double,3>::determ () const
{
    return nifti_dmat33_determ(*niftiPointer());
}

template <>
inline SquareMatrix<nifti_dmat33,double,3> SquareMatrix<nifti_dmat33,double,3>::multiply (const SquareMatrix<nifti_dmat33,double,3> &other) const
{
    return SquareMatrix<nifti_dmat33,double,3>(nifti_dmat33_mul(*niftiPointer(), *other.niftiPointer()));
}

template <>
inline SquareMatrix<mat44,float,4> SquareMatrix<mat44,float,4>::multiply (const SquareMatrix<mat44,float,4> &other) const
{
    return SquareMatrix<mat44,float,4>(nifti_mat44_mul(*niftiPointer(), *other.niftiPointer()));
}

template <>
inline SquareMatrix<nifti_dmat44,double,4> SquareMatrix<nifti_dmat44,double,4>::inverse () const
{
    return SquareMatrix<nifti_dmat44,double,4>(nifti_dmat44_inverse(*niftiPointer()));
}

template <>
inline SquareMatrix<nifti_dmat44,double,4> SquareMatrix<nifti_dmat44,double,4>::multiply (const SquareMatrix<nifti_dmat44,double,4> &other) const
{
    return SquareMatrix<nifti_dmat44,double,4>(nifti_dmat44_mul(*niftiPointer(), *other.niftiPointer()));
}

#endif

template <class NiftiType, typename ElementType, int Order>
inline Vector<ElementType,Order> SquareMatrix<NiftiType,ElementType,Order>::multiply (const Vector<ElementType,Order> &vec) const
{
    Vector<ElementType,Order> result;
    for (int i=0; i<Order; i++)
    {
        for (int j=0; j<Order; j++)
            result[i] += (*this)(i,j) * vec[j];
    }
    return result;
}

#endif
