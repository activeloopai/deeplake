#pragma once

#include "json_wrapper.hpp"

#include <base/assert.hpp>
#include <base/type_traits.hpp>

#include <boost/json.hpp>
#include <boost/json/system_error.hpp>
#include <boost/system/system_error.hpp>

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace icm {

class json
{
public:
    using array = boost::json::array;
    using number_integer_t = int32_t;
    using number_unsigned_t = uint32_t;
    using iterator = mutable_json_wrapper::iterator;
    using const_iterator = mutable_json_wrapper::const_iterator;

    using exception = boost::system::system_error;
    using parse_error = boost::system::system_error;

private:
    mutable_json_wrapper wrapper() noexcept
    {
        return mutable_json_wrapper(val_);
    }

    const_json_wrapper wrapper() const noexcept
    {
        return const_json_wrapper(val_);
    }

    /// @name Constructors
    /// @{
public:
    json()
    {
        val_.emplace_object();
    }
    json(const json&) noexcept = default;
    json(json&&) noexcept = default;
    json& operator=(const json&) noexcept = default;
    json& operator=(json&&) noexcept = default;

    json(array&& v)
        : val_(std::move(v))
    {
    }

    explicit json(const char* v)
        : val_(v)
    {
    }

    explicit json(boost::json::value&& val)
        : val_(std::move(val))
    {
    }

    explicit json(const boost::json::value& val)
        : val_(val)
    {
    }

    template <typename T>
    requires std::is_same_v<T, std::string_view>
    explicit json(T v)
        : val_(v)
    {
    }

    template <typename T>
    requires std::is_same_v<T, std::string>
    explicit json(const T& v)
        : val_(v)
    {
    }

    template <typename T>
    explicit json(T v)
    requires base::arithmetic<T>
        : val_(std::move(v))
    {
    }

    template <typename T>
    json(std::initializer_list<T> v)
    requires base::arithmetic<T>
        : json(std::span<const T>(v))
    {
    }

    template <bool b>
    json(const impl::key_value_pair<b>& kv)
    {
        if (kv.is_object()) {
            val_.emplace_object()[kv.key()] = kv.value().value();
        }
        val_ = kv.value().value();
    }

    template <typename T>
    requires std::is_same_v<T, std::string>
    json(std::string_view k, T v)
    {
        val_.emplace_object().emplace(k, std::move(v));
    }

    template <typename T>
    json(std::string_view k, T v)
    requires std::is_enum_v<T>
    {
        val_.emplace_object().emplace(k, static_cast<int64_t>(v));
    }

    template <typename T>
    requires std::is_same_v<T, std::string_view>
    json(std::string_view k, T v)
    {
        val_.emplace_object().emplace(k, v);
    }

    json(std::string_view k, boost::json::value v)
    {
        val_.emplace_object().emplace(k, std::move(v));
    }

    json(std::string_view k, impl::is_map auto& m)
    {
        boost::json::object obj;
        for (const auto& [key, value] : m) {
            obj[key] = value;
        }
        val_.emplace_object().emplace(k, std::move(obj));
    }

    template <impl::has_to_json T>
    json(std::string_view k, const T& v)
    {
        val_.emplace_object().emplace(k, std::move(v.to_json().value()));
    }

    template <typename T>
    json(std::string_view k, const std::vector<T>& v)
    {
        array arr;
        for (const auto& i : v) {
            arr.push_back(boost::json::value(i));
        }
        val_.emplace_object().emplace(k, std::move(arr));
    }

    template <typename T, std::size_t N>
    json(std::string_view k, const std::array<T, N>& v)
    {
        array arr;
        for (const auto& i : v) {
            arr.push_back(boost::json::value(i));
        }
        val_.emplace_object().emplace(k, arr);
    }

    template <impl::is_map T>
    json(const T& m)
    {
        boost::json::object obj;
        for (const auto& [key, value] : m) {
            obj[key] = value;
        }
        val_ = std::move(obj);
    }

    template <impl::has_to_json T>
    json(const T& v)
        : val_(v.to_json().value())
    {
    }

    json(std::initializer_list<json> l)
    {
        for (const auto& i : l) {
            if (i.is_null()) {
                continue;
            }
            if (i.is_array()) {
                val_.emplace_array() = i.val_.as_array();
            } else if (i.is_object()) {
                if (!val_.is_object()) {
                    val_.emplace_object();
                }
                for (const auto& j : i.items()) {
                    val_.as_object().emplace(j.key(), j.value().value());
                }
            }
        }
    }

    json(const const_json_wrapper& v)
        : val_(v.value())
    {
    }

    json(const mutable_json_wrapper& v)
        : val_(v.value())
    {
    }

    template <typename T>
    json(const std::vector<T>& v)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : v) {
            arr.push_back(i);
        }
    }

    template <typename T>
    json(const std::span<T>& v)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : v) {
            arr.push_back(i);
        }
    }

    template <typename T, std::size_t N>
    json(const std::array<T, N>& v)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : v) {
            arr.push_back(i);
        }
    }

    /// @}

    template <typename T>
    mutable_json_wrapper operator=(const T& t)
    {
        if constexpr (std::is_same_v<json, T>) {
            return wrapper().template operator= <T>(t.wrapper());
        }
        return wrapper().template operator= <T>(t);
    }

    template <typename T>
    static json parse(T&& i)
    {
        return json(boost::json::parse(std::forward<T>(i)));
    }

    static json object()
    {
        return json{};
    }

    auto begin()
    {
        return wrapper().begin();
    }

    auto begin() const
    {
        return wrapper().begin();
    }

    auto end()
    {
        return wrapper().end();
    }

    auto end() const
    {
        return wrapper().end();
    }

    auto cbegin()
    {
        return wrapper().cbegin();
    }

    auto cbegin() const
    {
        return wrapper().cbegin();
    }

    auto cend()
    {
        return wrapper().cend();
    }

    auto cend() const
    {
        return wrapper().cend();
    }

    bool contains(std::string_view key) const
    {
        return wrapper().contains(key);
    }

    bool contains(const_json_wrapper key) const
    {
        return wrapper().contains(key);
    }

    bool is_null() const noexcept
    {
        return wrapper().is_null();
    }

    bool empty() const noexcept
    {
        return wrapper().empty();
    }

    bool is_array() const noexcept
    {
        return wrapper().is_array();
    }

    bool is_string() const noexcept
    {
        return wrapper().is_string();
    }

    bool is_boolean() const noexcept
    {
        return wrapper().is_boolean();
    }

    bool is_number_integer() const noexcept
    {
        return wrapper().is_number_integer();
    }

    bool is_number_unsigned() const noexcept
    {
        return wrapper().is_number_unsigned();
    }

    bool is_number_float() const noexcept
    {
        return wrapper().is_number_float();
    }

    bool is_number() const noexcept
    {
        return wrapper().is_number();
    }

    bool is_structured() const noexcept
    {
        return wrapper().is_structured();
    }

    bool is_object() const noexcept
    {
        return wrapper().is_object();
    }

    auto size() const noexcept
    {
        return wrapper().size();
    }

    auto& object_val()
    {
        return wrapper().object_val();
    }

    const auto& object_val() const
    {
        return wrapper().object_val();
    }

    boost::json::value& value() noexcept
    {
        return val_;
    }

    const boost::json::value& value() const noexcept
    {
        return val_;
    }

    template <typename T>
    T get() const
    {
        return wrapper().template get<T>();
    }

    template <typename T>
    void get_to(T& t) const
    {
        wrapper().template get_to<T>(t);
    }

    template <typename T>
    explicit operator T() const
    {
        return wrapper().operator T();
    }

    operator boost::json::value() const
    {
        return val_;
    }

    /**
     * @name operator[] lvalue reference overloads
     */
    /// @{
    auto operator[](std::string_view sv) &
    {
        return wrapper().operator[](sv);
    }

    auto operator[](std::string_view sv) const&
    {
        return wrapper().operator[](sv);
    }

    auto operator[](const_json_wrapper w) &
    {
        return wrapper().operator[](w);
    }

    auto operator[](const_json_wrapper w) const&
    {
        return wrapper().operator[](w);
    }

    auto operator[](std::size_t i) &
    {
        return wrapper().operator[](i);
    }

    auto operator[](std::size_t i) const&
    {
        return wrapper().operator[](i);
    }
    /// @}

    /**
     * @name operator[] rvalue reference overloads
     * @brief
     */
    /// @{

    json operator[](std::string_view sv) &&
    {
        return json(operator[](sv));
    }

    json operator[](std::string_view sv) const&&
    {
        return json(operator[](sv));
    }

    json operator[](const_json_wrapper w) &&
    {
        return json(operator[](w));
    }

    json operator[](const_json_wrapper w) const&&
    {
        return json(operator[](w));
    }

    json operator[](std::size_t i) &&
    {
        return json(operator[](i));
    }

    json operator[](std::size_t i) const&&
    {
        return json(operator[](i));
    }

    /// @}

    auto at(std::string_view sv)
    {
        return wrapper().at(sv);
    }

    auto at(std::string_view sv) const
    {
        return wrapper().at(sv);
    }

    auto at(std::size_t i)
    {
        return wrapper().at(i);
    }

    auto at(std::size_t i) const
    {
        return wrapper().at(i);
    }

    auto dump(int level = 0) const
    {
        return wrapper().dump(level);
    }

    template <typename T>
    T value(std::string_view key, T default_value) const
    {
        return wrapper().value(key, default_value);
    }

    const_json_wrapper items() const
    {
        return wrapper().items();
    }

    auto find(std::string_view k)
    {
        return wrapper().find(k);
    }

    auto find(std::string_view k) const
    {
        return wrapper().find(k);
    }

    iterator erase(const_iterator pos) noexcept
    {
        return wrapper().erase(pos);
    }

    std::size_t erase(std::string_view key) noexcept
    {
        return wrapper().erase(key);
    }

    void clear() noexcept
    {
        wrapper().clear();
    }

    template <typename T>
    void emplace(std::string_view k, T&& v)
    {
        wrapper().emplace(k, std::forward<T>(v));
    }

    void push_back(json&& j)
    {
        if (!val_.is_array()) {
            val_.emplace_array();
        }
        val_.as_array().push_back(std::move(j.value()));
    }

    mutable_json_wrapper back()
    {
        ASSERT(is_array());
        return mutable_json_wrapper(val_.as_array().back());
    }

    const_json_wrapper back() const
    {
        ASSERT(is_array());
        return const_json_wrapper(val_.as_array().back());
    }

    mutable_json_wrapper front()
    {
        ASSERT(is_array());
        return mutable_json_wrapper(val_.as_array().front());
    }

    const_json_wrapper front() const
    {
        ASSERT(is_array());
        return const_json_wrapper(val_.as_array().front());
    }

    bool operator==(const json& rhs) const noexcept = default;

    friend std::istream& operator>>(std::istream& is, json& j)
    {
        j = json(boost::json::parse(is));
        return is;
    }

    friend std::ostream& operator<<(std::ostream& os, const json& j)
    {
        os << j.dump();
        return os;
    }

private:
    boost::json::value val_;
};

template <bool is_const>
template <impl::has_to_json T>
inline json_wrapper<is_const>& json_wrapper<is_const>::operator=(const T& t)
requires(!is_const)
{
    val_ = t.to_json().value();
    return *this;
}

template <bool is_const>
inline json_wrapper<is_const>& json_wrapper<is_const>::operator=(const json& t)
requires(!is_const)
{
    val_ = t.value();
    return *this;
}

template <bool is_const>
inline json_wrapper<is_const>& json_wrapper<is_const>::operator=(std::initializer_list<json> l)
requires(!is_const)
{
    val_ = json(l).value();
    return *this;
}

inline void tag_invoke(boost::json::value_from_tag, boost::json::value& v, const icm::json& j)
{
    v = j.value();
}

inline icm::json tag_invoke(boost::json::value_to_tag<icm::json>, const boost::json::value& v)
{
    icm::json j;
    j.value() = v;
    return j;
}

inline namespace literals {

/// _json literal is reserved by nlohmann::json
inline auto operator""_js(const char* s, std::size_t n)
{
    return icm::json::parse(s);
}

} // namespace literals

} // namespace icm

using icm::literals::operator""_js;
