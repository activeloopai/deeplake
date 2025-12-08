#include "std_span_array.hpp"

namespace nd {

inline array data_iterator::operator*() const
{
    return array(impl::std_span_array_nd(nullptr, data_, shape_, dtype_));
}

template <typename T>
inline array iterator::concrete_holder<T>::value() const
{
    return *value_;
}

inline array iterator::operator*() const
{
    switch (iterator_.index()) {
    case 0:
        return *std::get<0>(iterator_);
    case 1:
        return *std::get<1>(iterator_);
    case 2:
        return *std::get<2>(iterator_);
    case 3:
        return std::get<3>(iterator_)->value();
    default:
        ASSERT(false);
    }
    return array();
}

inline void iterator::operator++()
{
    switch (iterator_.index()) {
    case 0:
        ++std::get<0>(iterator_);
        break;
    case 1:
        ++std::get<1>(iterator_);
        break;
    case 2:
        ++std::get<2>(iterator_);
        break;
    case 3:
        ++(*std::get<3>(iterator_));
        break;
    default:
        ASSERT(false);
    }
}

inline void iterator::operator+=(int n)
{
    switch (iterator_.index()) {
    case 0:
        std::get<0>(iterator_) += n;
        break;
    case 1:
        std::get<1>(iterator_) += n;
        break;
    case 2:
        std::get<2>(iterator_) += n;
        break;
    case 3:
        (*std::get<3>(iterator_)) += n;
        break;
    default:
        ASSERT(false);
    }
}

inline void iterator::operator--()
{
    switch (iterator_.index()) {
    case 0:
        --std::get<0>(iterator_);
        break;
    case 1:
        --std::get<1>(iterator_);
        break;
    case 2:
        --std::get<2>(iterator_);
        break;
    case 3:
        --(*std::get<3>(iterator_));
        break;
    default:
        ASSERT(false);
    }
}

inline void iterator::operator-=(int n)
{
    switch (iterator_.index()) {
    case 0:
        std::get<0>(iterator_) -= n;
        break;
    case 1:
        std::get<1>(iterator_) -= n;
        break;
    case 2:
        std::get<2>(iterator_) -= n;
        break;
    case 3:
        (*std::get<3>(iterator_)) -= n;
        break;
    default:
        ASSERT(false);
    }
}

inline std::ptrdiff_t iterator::operator-(const iterator& other) const
{
    ASSERT(iterator_.index() == other.iterator_.index());
    switch (iterator_.index()) {
    case 0:
        return std::get<0>(iterator_) - std::get<0>(other.iterator_);
    case 1:
        return std::get<1>(iterator_) - std::get<1>(other.iterator_);
    case 2:
        return std::get<2>(iterator_) - std::get<2>(other.iterator_);
    case 3:
        return std::get<3>(iterator_)->operator-(*std::get<3>(other.iterator_));
    default:
        ASSERT(false);
        return 0;
    }
}

}
