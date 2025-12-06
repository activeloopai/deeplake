#pragma once

#include "array.hpp"
#include "impl/transformed_array.hpp"

namespace nd {

template <typename F>
array transform(array arr, F f)
{
    return impl::create_transformed_array(std::move(arr), std::move(f));
}

template <typename F>
array transform(array a1, array a2, F f)
{
    return impl::create_transformed_array(std::move(a1), std::move(a2), std::move(f));
}

}