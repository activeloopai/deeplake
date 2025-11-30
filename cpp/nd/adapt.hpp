#pragma once

/**
 * @file adapt.hpp
 * @brief Definitions and implementations of `adapt`, `adapt_shape`, `empty`, `dynamic_empty`, `dynamic` functions.
 */

#include "array.hpp"
#include "dict.hpp"

#include "impl/scalar_array.hpp"
#include "impl/std_array_array.hpp"
#include "impl/std_span_array.hpp"
#include "impl/vector_array.hpp"
#include "iterator.hpp"

#include <base/type_traits.hpp>

#include <icm/shape_array.hpp>
#include <icm/small_vector.hpp>
#include <icm/vector.hpp>

#include <icm/json.hpp>

#include <array>
#include <span>
#include <vector>

namespace nd {

template <typename T>
requires(base::arithmetic<T>)
inline array adapt(T value)
{
    return array(value);
}

inline array adapt(dict value)
{
    return array(impl::scalar_array(std::move(value)));
}

template <typename T, std::size_t size>
requires(base::arithmetic<T>)
inline array adapt(std::array<T, size> value)
{
    return array(impl::std_array_array<T, size>(value));
}

template <typename T>
inline array adapt(std::vector<T> value)
{
    return array(impl::vector_array<std::vector<T>>(std::move(value)));
}

template <>
array adapt(std::vector<std::string> value);

template <typename T>
requires(base::arithmetic<T>)
inline array adapt(std::vector<T> value, icm::shape&& shape)
{
    return array(impl::vector_array_with_shape<std::vector<T>>(std::move(value), std::move(shape)));
}

template <typename T>
requires(base::arithmetic<T>)
inline array adapt(icm::vector<T> value)
{
    return array(impl::vector_array<icm::vector<T>>(std::move(value)));
}

array adapt(base::memory_buffer buffer, enum dtype dtype);

array adapt(base::memory_buffer buffer, icm::shape&& shape, enum dtype dtype);

array adapt(nd::array arr, std::vector<int32_t>&& transform_info);

array adapt(icm::string_map<nd::array> dict);

template <typename T>
inline array adapt(std::initializer_list<T> value)
{
    return array(impl::vector_array<icm::vector<T>>(value));
}

template <typename T>
inline array adapt(icm::vector<T> value, icm::shape&& shape)
{
    return array(impl::vector_array_with_shape<icm::vector<T>>(std::move(value), std::move(shape)));
}

array adapt(const icm::json& js);

array adapt(std::string str);

array adapt_shape(const array& arr);
array adapt_shape(const icm::shape& shape);

array empty(dtype type, const icm::shape& shape);

array dynamic_empty(dtype type, uint32_t shape);
array dynamic_empty(dtype type, std::vector<icm::shape> shapes);
array dynamic_empty(dtype type, icm::shape_array<uint32_t> shapes);

array dynamic_repeated(array arr, int64_t count);
array dynamic(std::vector<array> arrays);
array dynamic(icm::vector<array> arrays);

} // namespace nd
