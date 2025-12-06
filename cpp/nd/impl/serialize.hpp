#pragma once

#include "../dtype.hpp"

#include <icm/small_vector.hpp>
#include <icm/vector.hpp>

#include <iostream>

namespace nd {

namespace impl {

static constexpr inline int version = 2;

template <typename T>
inline void write_pod(std::ostream& s, const T& v)
{
    s.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
inline void read_pod(std::istream& s, T& v)
{
    s.read(reinterpret_cast<char*>(&v), sizeof(T));
}

template <typename C>
inline void write_container(std::ostream& s, const C& c)
{
    using V = typename C::value_type;
    static_assert(std::is_standard_layout_v<V> && std::is_trivial_v<V>);
    write_pod(s, nd::dtype_enum_v<V>);
    write_pod(s, c.size());
    s.write(reinterpret_cast<const char*>(c.data()), c.size() * sizeof(V));
}

template <typename F>
inline auto read_vector(std::istream& s, F f)
{
    nd::dtype t;
    read_pod(s, t);
    std::size_t size = 0;
    read_pod(s, size);
    return nd::switch_dtype(t, [&s, size, f]<typename T>() mutable {
        icm::vector<T> r;
        r.resize(size);
        s.read(reinterpret_cast<char*>(r.data()), size * sizeof(T));
        return f(std::move(r));
    });
}

template <typename T>
inline auto read_vector(std::istream& s, std::vector<T>& v)
{
    nd::dtype t;
    read_pod(s, t);
    if (t != nd::dtype_enum_v<T>) {
        throw nd::invalid_operation("Dtype mismatch");
    }
    std::size_t size = 0;
    read_pod(s, size);
    v.resize(size);
    s.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
}

template <typename T>
inline auto read_small_vector(std::istream& s, icm::small_vector<T>& v)
{
    nd::dtype t;
    read_pod(s, t);
    if (t != nd::dtype_enum_v<T>) {
        throw nd::invalid_operation("Dtype mismatch");
    }
    std::size_t size = 0;
    read_pod(s, size);
    v.resize(size);
    s.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
}

inline auto read_shape(std::istream& s, icm::shape& v)
{
    icm::small_vector<icm::shape::value_type> r;
    read_small_vector<icm::shape::value_type>(s, r);
    v = icm::shape(std::move(r));
}
constexpr inline std::size_t version_size = 4;
constexpr inline std::size_t is_dynamic_flag_size = 1;

inline constexpr std::size_t header_size()
{
    return version_size + is_dynamic_flag_size;
}

inline dtype container_dtype(const uint8_t* ptr)
{
    return static_cast<dtype>(*ptr);
}

inline std::size_t container_size(const uint8_t* ptr)
{
    return *(reinterpret_cast<const std::size_t*>(ptr + sizeof(dtype)));
}

inline const uint8_t* container_data(const uint8_t* ptr)
{
    return ptr + sizeof(dtype) + sizeof(std::size_t);
}

inline const uint8_t* skip_container(const uint8_t* ptr)
{
    auto d = container_dtype(ptr);
    auto size = container_size(ptr);
    return ptr + sizeof(dtype) + sizeof(std::size_t) + size * dtype_bytes(d);
}

} // namespace impl

} // namespace nd
