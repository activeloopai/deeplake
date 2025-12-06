#pragma once

#include "buffer.hpp"
#include "read_result.hpp"

#include <base/assert.hpp>

#include <cstdint>
#include <cstring>
#include <tuple>
#include <variant>

namespace format {

namespace impl {

template <int index, typename... Args>
requires(index == sizeof...(Args))
void read_tuple(const base::memory_buffer&, const int64_t&, std::tuple<Args...>&)
{
    /// Intentionally empty.
}

template <int index, typename... Args>
requires(index < sizeof...(Args))
void read_tuple(const base::memory_buffer& bytes, int64_t& offset, std::tuple<Args...>& result)
{
    using T = std::remove_cvref_t<decltype(std::get<index>(result))>;
    auto [obj, off] = ::format::read<T>(bytes, offset);
    std::get<index>(result) = std::move(obj);
    offset = off;
    read_tuple<index + 1>(bytes, offset, result);
}

template <int index, typename... Args>
requires(index == sizeof...(Args))
int64_t tuple_output_size(int64_t size, const std::tuple<Args...>&)
{
    return size;
}

template <int index, typename... Args>
requires(index < sizeof...(Args))
int64_t tuple_output_size(int64_t size, const std::tuple<Args...>& result)
{
    using T = std::remove_cvref_t<decltype(std::get<index>(result))>;
    size += serializable<T>::output_size(std::get<index>(result));
    return tuple_output_size<index + 1>(size, result);
}

template <int index, typename... Args>
requires(index == sizeof...(Args))
inline void write_tuple(const std::tuple<Args...>&, const buffer_t&, int64_t)
{
    /// Intentionally empty.
}

template <int index, typename... Args>
requires(index < sizeof...(Args))
inline void write_tuple(const std::tuple<Args...>& o, buffer_t& bytes, int64_t offset)
{
    using T = std::remove_cvref_t<decltype(std::get<index>(o))>;
    offset = format::write(std::get<index>(o), bytes, offset);
    write_tuple<index + 1>(o, bytes, offset);
}

template <int index, typename... Args>
requires(index >= sizeof...(Args))
inline format::read_result<std::variant<Args...>> read_variant(int64_t, const base::memory_buffer&, int64_t)
{
    return format::read_result<std::variant<Args...>>();
}

template <int index, typename... Args>
requires(index < sizeof...(Args))
inline format::read_result<std::variant<Args...>> read_variant(int64_t i, const base::memory_buffer& bytes, int64_t offset)
{
    if (index != i) {
        return read_variant<index + 1, Args...>(i, bytes, offset);
    }
    using T = std::variant_alternative_t<index, std::variant<Args...>>;
    auto [t, o] = format::read<T>(bytes, offset);
    return {std::variant<Args...>(std::in_place_index_t<index>(), std::move(t)), o};
}

}

/**
 * @brief Utility to read tuple.
 *
 * @tparam Args
 * @param bytes
 * @param offset
 * @return read_result<std::tuple<Args ...>>
 */
template <typename ... Args>
inline read_result<std::tuple<Args ...>> read_tuple(const base::memory_buffer& bytes, int64_t offset)
{
    std::tuple<Args...> result;
    impl::read_tuple<0>(bytes, offset, result);
    return read_result<std::tuple<Args ...>>{std::move(result), offset};
}

/**
 * @brief Utility to get output size of tuple.
 *
 * @tparam Args
 * @param result
 * @return int64_t
 */
template <typename ... Args>
inline int64_t tuple_output_size(const std::tuple<Args ...>& result)
{
    return impl::tuple_output_size<0>(0L, result);
}

/**
 * @brief Utility to write tuple.
 *
 * @tparam Args
 * @param o
 * @param bytes
 * @param offset
 */
template <typename ... Args>
inline void write_tuple(const std::tuple<Args ...>& o, buffer_t& bytes, int64_t offset)
{
    impl::write_tuple<0>(o, bytes, offset);
}

/**
 * @brief Utility to write tuple.
 *
 * @tparam Args
 * @param o
 * @return buffer_t
 */
template <typename ... Args>
inline buffer_t write_tuple(const std::tuple<Args ...>& o)
{
    buffer_t bytes;
    impl::write_tuple<0>(o, bytes, 0L);
    return bytes;
}

/**
 * @brief Utility to read variant.
 * 
 * @tparam Args
 * @param bytes
 * @param offset
 * @return read_result<std::variant<Args...>>
 */
template <typename... Args>
inline read_result<std::variant<Args...>> read_variant(const base::memory_buffer& bytes, int64_t offset)
{
    auto [index, o] = format::read<int64_t>(bytes, offset);
    return impl::read_variant<0, Args...>(index, bytes, o);
}

/**
 * @brief Utility to read vector like container - vector, list, boost::vector, boost::small_vector.
 *
 * @tparam Container
 * @tparam Size
 * @param bytes
 * @param offset
 * @return read_result<Container>
 */
template <typename Container, typename Size = int64_t>
inline read_result<Container> read_vector_like(const base::memory_buffer& bytes, int64_t offset)
{
    auto [size, o] = read<Size>(bytes, offset);
    offset = o;
    auto r = Container();
    r.resize(size);
    if constexpr (std::is_standard_layout_v<typename Container::value_type> &&
                  std::is_trivial_v<typename Container::value_type>) {
        std::memcpy(r.data(), bytes.data() + offset, size * sizeof(typename Container::value_type));
        offset += size * sizeof(typename Container::value_type);
    } else {
        for (auto i = 0; i < size; ++i) {
            auto [t, o1] = read<typename Container::value_type>(bytes, offset);
            r[i] = std::move(t);
            offset = o1;
        }
    }
    return {std::move(r), offset};
}

/**
 * @brief Utility to read map like container - map, unordered_map, set.
 *
 * @tparam Container
 * @tparam Size
 * @param bytes
 * @param offset
 * @return read_result<Container>
 */
template <typename Container, typename Size = int64_t>
inline read_result<Container> read_map_like(const base::memory_buffer& bytes, int64_t offset)
{
    auto [size, o] = read<Size>(bytes, offset);
    offset = o;
    auto r = Container();
    for (auto i = 0; i < size; ++i) {
        auto [t, o1] = read<typename Container::value_type>(bytes, offset);
        r.emplace(std::move(t));
        offset = o1;
    }
    return {std::move(r), offset};
}

/**
 * @brief Utility to get output size of vector like container.
 *
 * @tparam Container
 * @tparam Size
 * @param o
 * @return int64_t
 */
template <typename Container, typename Size = int64_t>
inline int64_t container_output_size(const Container& o) noexcept
{
    if constexpr (std::is_standard_layout_v<typename Container::value_type> &&
                  std::is_trivial_v<typename Container::value_type>) {
        return sizeof(Size) + o.size() * sizeof(typename Container::value_type);
    } else {
        int64_t size = sizeof(Size);
        for (const auto& t : o) {
            size += serializable<typename Container::value_type>::output_size(t);
        }
        return size;
    }
}

template <typename Container, typename Size = int64_t>
inline void write_container(const Container& o, buffer_t& bytes, int64_t offset)
{
    offset = write(static_cast<Size>(o.size()), bytes, offset);
    for (const auto& t : o) {
        offset = write(t, bytes, offset);
    }
}
/**
 * @brief Utility to write vector like container, which has known output size.
 *
 * @tparam Container
 * @tparam Size
 * @param o
 * @param bytes
 * @param offset
 */
template <typename Container, typename Size = int64_t>
inline void write_vector_like(const Container& o, buffer_t& bytes, int64_t offset)
{
    if constexpr (std::is_standard_layout_v<typename Container::value_type> &&
                  std::is_trivial_v<typename Container::value_type>) {
        offset = write(static_cast<Size>(o.size()), bytes, offset);
        ASSERT(bytes.size() - offset >= o.size() * sizeof(typename Container::value_type));
        std::memcpy(bytes.data() + offset, o.data(), o.size() * sizeof(typename Container::value_type));
    } else {
        write_container(o, bytes, offset);
    }
}

/**
 * @brief Utility to write vector like container, which has no known output size.
 *
 * @tparam Container
 * @tparam Size
 * @param o
 * @return buffer_t
 */
template <typename Container, typename Size = int64_t>
inline buffer_t write_container(const Container& o)
{
    auto bytes = buffer_t();
    auto offset = 0L;
    offset = write(static_cast<Size>(o.size()), bytes, offset);
    for (const auto& t : o) {
        offset = write(t, bytes, offset);
    }
    return bytes;
}

}