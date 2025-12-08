#pragma once

#include "read_result.hpp"
#include "serializable.hpp"
#include "pod_serializable.hpp"
#include "serializer.hpp"
#include "stl_common.hpp"

#include <base/span_cast.hpp>

#include <boost/container/vector.hpp>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace format {

template <typename F, typename S>
requires(!std::is_standard_layout_v<std::pair<F, S>> || !std::is_trivial_v<std::pair<F, S>>)
struct serializable<std::pair<F, S>>
{
    using A = std::remove_cv_t<F>;
    using B = std::remove_cv_t<S>;

    inline static read_result<std::pair<F, S>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        auto r = std::pair<F, S>();
        auto read_f = format::read<A>(bytes, offset);
        auto read_s = format::read<B>(bytes, read_f.offset);
        return {std::make_pair<F, S>(std::move(read_f.object), std::move(read_s.object)), read_s.offset};
    }

    inline static buffer_t write(const std::pair<A, B>& o)
    requires(!impl::has_output_size_member_function_v<A> ||
             !impl::has_output_size_member_function_v<B>)
    {
        buffer_t buf;
        int64_t offset = 0L;
        offset = serializable<A>::write(o.first, buf, offset);
        offset = serializable<B>::write(o.second, buf, offset);
        return buf;
    }

    inline static int64_t output_size(const std::pair<F, S>& o) noexcept
    requires(impl::has_output_size_member_function_v<serializable<A>> &&
             impl::has_output_size_member_function_v<serializable<B>>)
    {
        return serializable<A>::output_size(o.first) + serializable<B>::output_size(o.second);
    }

    inline static void write(const std::pair<F, S>& o, buffer_t& bytes, int64_t offset)
    requires(impl::has_output_size_member_function_v<serializable<A>> &&
             impl::has_output_size_member_function_v<serializable<B>>)
    {
        offset = format::write<A>(o.first, bytes, offset);
        format::write<B>(o.second, bytes, offset);
    }
};

template <typename... Args>
requires(!impl::has_output_size_member_function_v<serializable<Args>> || ...)
struct serializable<std::tuple<Args...>>
{
    inline static read_result<std::tuple<Args...>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_tuple<Args...>(bytes, offset);
    }

    inline static buffer_t write(const std::tuple<Args...>& o)
    {
        return write_tuple(o);
    }
};

template <typename... Args>
requires(impl::has_output_size_member_function_v<serializable<Args>> && ...)
struct serializable<std::tuple<Args...>>
{
    inline static read_result<std::tuple<Args...>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_tuple<Args...>(bytes, offset);
    }

    inline static int64_t output_size(const std::tuple<Args...>& o) noexcept
    {
        return tuple_output_size(o);
    }

    inline static void write(const std::tuple<Args ...>& o, buffer_t& bytes, int64_t offset)
    {
        write_tuple(o, bytes, offset);
    }
};


template <typename... Args>
requires(!impl::has_output_size_member_function_v<serializable<Args>> || ...)
struct serializable<std::variant<Args...>>
{
    inline static read_result<std::variant<Args...>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_variant<Args...>(bytes, offset);
    }

    inline static buffer_t write(const std::variant<Args...>& o)
    {
        buffer_t bytes;
        auto offset = format::write(static_cast<int64_t>(o.index()), bytes, 0L);
        std::visit(
            [&bytes, offset](const auto& a) {
                format::write(a, bytes, offset);
            },
            o);
        return bytes;
    }
};

template <typename... Args>
requires(impl::has_output_size_member_function_v<serializable<Args>> && ...)
struct serializable<std::variant<Args...>>
{
    inline static read_result<std::variant<Args...>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_variant<Args...>(bytes, offset);
    }

    inline static int64_t output_size(const std::variant<Args...>& o) noexcept
    {
        return std::visit(
                   []<typename T>(const T& a) {
                       return serializable<T>::output_size(a);
                   },
                   o) +
               sizeof(int64_t);
    }

    inline static void write(const std::variant<Args ...>& o, buffer_t& bytes, int64_t offset)
    {
        auto off = format::write(static_cast<int64_t>(o.index()), bytes, offset);
        std::visit(
            [&bytes, off]<typename T>(const T& a) {
                format::write<T>(a, bytes, off);
            },
            o);
    }
};

template <typename T>
requires (!impl::has_output_size_member_function_v<serializable<T>>)
struct serializable<std::optional<T>>
{
    inline static read_result<std::optional<T>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        auto [b, off] = format::read<bool>(bytes, offset);
        if (b) {
            auto [v, o] = format::read<T>(bytes, off);
            return {std::optional<T>(std::move(v)), o};
        }
        return {std::optional<T>(), off};
    }

    inline static buffer_t write(const std::optional<T>& o)
    {
        buffer_t bytes;
        auto offset = format::write(o.has_value(), bytes, 0L);
        if (o.has_value()) {
            format::write<T>(o.value(), bytes, offset);
        }
        return bytes;
    }
};

template <typename T>
requires (impl::has_output_size_member_function_v<serializable<T>>)
struct serializable<std::optional<T>>
{
    inline static read_result<std::optional<T>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        auto [b, off] = format::read<bool>(bytes, offset);
        if (b) {
            auto [v, o] = format::read<T>(bytes, off);
            return {std::optional<T>(std::move(v)), o};
        }
        return {std::optional<T>(), off};
    }

    inline static int64_t output_size(const std::optional<T>& o) noexcept
    {
        auto b = o.has_value();
        if (b) {
            return serializable<bool>::output_size(b) + serializable<T>::output_size(o.value());
        }
        return serializable<bool>::output_size(b);
    }

    inline static void write(const std::optional<T> o, buffer_t& bytes, int64_t offset)
    {
        offset = format::write(o.has_value(), bytes, offset);
        if (o.has_value()) {
            format::write<T>(o.value(), bytes, offset);
        }
    }
};

template <typename T, std::size_t N>
requires(!std::is_standard_layout_v<T> || !std::is_trivial_v<T>)
struct serializable<std::array<T, N>>
{
    inline static read_result<std::array<T, N>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        auto r = std::array<T, N>();
        for (auto i = 0; i < N; ++i) {
            auto [t, o] = format::read<T>(bytes, offset);
            r[i] = std::move(t);
            offset = o;
        }
        return {std::move(r), offset};
    }

    inline static buffer_t write(const std::array<T, N>& o)
    requires(!impl::has_output_size_member_function_v<serializable<T>>)
    {
        buffer_t buf;
        int64_t offset = 0L;
        for (const auto& t: o) {
            offset = format::write(t, buf, offset);
        }
        return buf;
    }

    inline static int64_t output_size(const std::array<T, N>& o) noexcept
    requires(impl::has_output_size_member_function_v<serializable<T>>)
    {
        int64_t size = 0L;
        for (const auto& t: o) {
            size += serializable<T>::output_size(t);
        }
        return size;
    }

    inline static void write(const std::array<T, N>& o, buffer_t& bytes, int64_t offset)
    requires(impl::has_output_size_member_function_v<serializable<T>>)
    {
        for (const auto& t : o) {
            offset = format::write(t, bytes, offset);
        }
    }
};

template <typename T>
requires (!impl::has_output_size_member_function_v<serializable<T>>)
struct serializable<std::vector<T>>
{
    inline static read_result<std::vector<T>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_vector_like<std::vector<T>>(bytes, offset);
    }

    inline static buffer_t write(const std::vector<T>& o)
    {
        return write_container(o);
    }
};

template <typename T>
requires (impl::has_output_size_member_function_v<serializable<T>>)
struct serializable<std::vector<T>>
{
    inline static read_result<std::vector<T>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_vector_like<std::vector<T>>(bytes, offset);
    }

    inline static int64_t output_size(const std::vector<T>& o) noexcept
    {
        return container_output_size(o);
    }

    inline static void write(const std::vector<T>& o, buffer_t& bytes, int64_t offset)
    {
        write_vector_like(o, bytes, offset);
    }
};

template <>
struct serializable<std::string>
{
    inline static read_result<std::string> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_vector_like<std::string>(bytes, offset);
    }

    inline static int64_t output_size(const std::string& o) noexcept
    {
        return container_output_size(o);
    }

    inline static void write(const std::string& o, buffer_t& bytes, int64_t offset)
    {
        write_container(o, bytes, offset);
    }
};

template <typename T, typename O, typename A>
struct serializable<std::set<T, O, A>>
{
    inline static read_result<std::set<T, O, A>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_map_like<std::set<T, O, A>>(bytes, offset);
    }

    inline static buffer_t write(const std::set<T, O, A>& o)
    requires(!impl::has_output_size_member_function_v<serializable<T>>)
    {
        return write_container(o);
    }

    inline static int64_t output_size(const std::set<T, O, A>& o) noexcept
    requires(impl::has_output_size_member_function_v<serializable<T>>)
    {
        return container_output_size(o);
    }

    inline static void write(const std::set<T, O, A>& o, buffer_t& bytes, int64_t offset)
    requires(impl::has_output_size_member_function_v<serializable<T>>)
    {
        write_container(o, bytes, offset);
    }
};

template <typename T, typename H, typename O, typename A>
struct serializable<std::unordered_set<T, H, O, A>>
{
    inline static read_result<std::unordered_set<T, H, O, A>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_map_like<std::unordered_set<T, H, O, A>>(bytes, offset);
    }

    inline static buffer_t write(const std::unordered_set<T, H, O, A>& o)
    requires(!impl::has_output_size_member_function_v<serializable<T>>)
    {
        return write_container(o);
    }

    inline static int64_t output_size(const std::unordered_set<T, H, O, A>& o) noexcept
    requires(impl::has_output_size_member_function_v<serializable<T>>)
    {
        return container_output_size(o);
    }

    inline static void write(const std::unordered_set<T, H, O, A>& o, buffer_t& bytes, int64_t offset)
    requires(impl::has_output_size_member_function_v<serializable<T>>)
    {
        write_container(o, bytes, offset);
    }
};

template <typename K, typename V, typename O, typename A>
struct serializable<std::map<K, V, O, A>>
{
    inline static read_result<std::map<K, V, O, A>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_map_like<std::map<K, V, O, A>>(bytes, offset);
    }

    inline static buffer_t write(const std::map<K, V, O, A>& o)
    requires(!impl::has_output_size_member_function_v<serializable<K>> ||
             !impl::has_output_size_member_function_v<serializable<V>>)
    {
        return write_container(o);
    }

    inline static int64_t output_size(const std::map<K, V, O, A>& o) noexcept
    requires(impl::has_output_size_member_function_v<serializable<K>> &&
             impl::has_output_size_member_function_v<serializable<V>>)
    {
        return container_output_size(o);
    }

    inline static void write(const std::map<K, V, O, A>& o, buffer_t& bytes, int64_t offset)
    requires(impl::has_output_size_member_function_v<serializable<K>> &&
             impl::has_output_size_member_function_v<serializable<V>>)
    {
        write_container(o, bytes, offset);
    }
};

template <typename K, typename V, typename H, typename O, typename A>
struct serializable<std::unordered_map<K, V, H, O, A>>
{
    inline static read_result<std::unordered_map<K, V, H, O, A>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_map_like<std::unordered_map<K, V, H, O, A>>(bytes, offset);
    }

    inline static buffer_t write(const std::unordered_map<K, V, H, O, A>& o)
    requires(!impl::has_output_size_member_function_v<serializable<K>> ||
             !impl::has_output_size_member_function_v<serializable<V>>)
    {
        return write_container(o);
    }

    inline static int64_t output_size(const std::unordered_map<K, V, H, O, A>& o) noexcept
    requires(impl::has_output_size_member_function_v<serializable<K>> &&
             impl::has_output_size_member_function_v<serializable<V>>)
    {
        return container_output_size(o);
    }

    inline static void write(const std::unordered_map<K, V, H, O, A>& o, buffer_t& bytes, int64_t offset)
    requires(impl::has_output_size_member_function_v<serializable<K>> &&
             impl::has_output_size_member_function_v<serializable<V>>)
    {
        write_container(o, bytes, offset);
    }
};

template <typename T>
requires (!impl::has_output_size_member_function_v<serializable<T>>)
struct serializable<boost::container::vector<T>>
{
    inline static read_result<boost::container::vector<T>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_vector_like<boost::container::vector<T>>(bytes, offset);
    }

    inline static buffer_t write(const boost::container::vector<T>& o)
    {
        return write_container(o);
    }
};

template <typename T>
requires (impl::has_output_size_member_function_v<serializable<T>>)
struct serializable<boost::container::vector<T>>
{
    inline static read_result<boost::container::vector<T>> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return read_vector_like<boost::container::vector<T>>(bytes, offset);
    }

    inline static int64_t output_size(const boost::container::vector<T>& o) noexcept
    {
        return container_output_size(o);
    }

    inline static void write(const boost::container::vector<T>& o, buffer_t& bytes, int64_t offset)
    {
        write_vector_like(o, bytes, offset);
    }
};
}
