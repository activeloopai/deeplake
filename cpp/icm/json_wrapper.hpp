#pragma once

#include <base/assert.hpp>
#include <base/type_traits.hpp>

#include <boost/json.hpp>

#include <array>
#include <concepts>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace icm {

class json;

namespace impl {

template<typename T>
concept is_container = requires(T a) {
    typename T::value_type;
    { std::begin(a) } -> std::same_as<typename T::iterator>;
    { std::end(a) } -> std::same_as<typename T::iterator>;
    { std::cbegin(a) } -> std::same_as<typename T::const_iterator>;
    { std::cend(a) } -> std::same_as<typename T::const_iterator>;
} && !std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view>;

template<typename T>
concept is_map = requires(T t) {
    typename T::key_type;
    typename T::mapped_type;
    typename T::value_type;
    requires std::is_same_v<typename T::key_type, std::string>;
    requires std::is_same_v<typename T::mapped_type, std::string>;
    { std::begin(t) } -> std::same_as<typename T::iterator>;
    { std::end(t) } -> std::same_as<typename T::iterator>;
};

template <typename T>
concept has_to_json = requires(T t) {
    { t.to_json() } -> std::same_as<json>;
};

template <bool is_const>
class key_value_pair;

} /// impl namespace

template <bool is_const>
class json_wrapper
{
private:
    using value_ref = std::conditional_t<is_const, const boost::json::value&, boost::json::value&>;
    using other_value_ref = std::conditional_t<is_const, boost::json::value&, const boost::json::value&>;
    using wrapper = json_wrapper<is_const>;
    using other_wrapper = json_wrapper<!is_const>;

public:
    explicit json_wrapper(value_ref val)
        : val_(val)
    {
    }

    explicit json_wrapper(boost::json::value& v) requires (is_const)
        : val_(v)
    {
    }

    template <bool b = is_const>
    requires(b && std::is_convertible_v<other_value_ref, value_ref>)
    json_wrapper(const other_wrapper& v)
        : val_(v.value())
    {
    }

    json_wrapper(const json_wrapper&) noexcept = default;
    json_wrapper(json_wrapper&&) noexcept = default;

    /// @brief Assignment operators
    /// @{
    json_wrapper& operator=(const json_wrapper& o) noexcept requires (!is_const)
    {
        val_ = o.val_;
        return *this;
    }

    json_wrapper& operator=(json_wrapper&& o) noexcept requires (!is_const)
    {
        val_ = std::move(o.val_);
        return *this;
    }

    template<typename T>
    json_wrapper& operator=(const T& t) requires (!is_const && !impl::is_container<T>)
    {
        if constexpr (std::is_same_v<wrapper, T>) {
            val_ = t.val_;
        } else if constexpr (std::is_same_v<other_wrapper, T>) {
            val_ = t.value();
        } else if constexpr (std::is_same_v<boost::json::value, T>) {
            val_ = boost::json::value(t);
        } else {
            val_ = t;
        }
        return *this;
    }

    template<impl::has_to_json T>
    json_wrapper& operator=(const T& t) requires (!is_const);

    json_wrapper& operator=(const json&) requires (!is_const);

    json_wrapper& operator=(std::initializer_list<json> l) requires (!is_const);

    template<typename T>
    json_wrapper& operator=(const std::span<T>& t) requires (!is_const)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : t) {
            arr.emplace_back(boost::json::value(i));
        }
        return *this;
    }

    template<typename T>
    json_wrapper& operator=(const std::vector<T>& t) requires (!is_const)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : t) {
            arr.emplace_back(boost::json::value(i));
        }
        return *this;
    }

    template<typename T, std::size_t N>
    json_wrapper& operator=(const std::array<T, N>& t) requires (!is_const)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : t) {
            arr.emplace_back(boost::json::value(i));
        }
        return *this;
    }

    template<impl::is_container Cont>
    json_wrapper& operator=(const Cont& t) requires (!is_const)
    {
        auto& arr = val_.emplace_array();
        for (const auto& i : t) {
            arr.emplace_back(boost::json::value(i));
        }
        return *this;
    }

    /// @}

    template<bool is_const_iter>
    struct generic_iterator
    {
        using object_iter = std::conditional_t<is_const_iter, boost::json::object::const_iterator, boost::json::object::iterator>;
        using array_iter = std::conditional_t<is_const_iter, boost::json::array::const_iterator, boost::json::array::iterator>;
        using iterator_category = std::random_access_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type = object_iter;
        using reference = object_iter&;
        using pointer = std::remove_reference_t<object_iter>*;

        generic_iterator() noexcept = default;
        generic_iterator(const generic_iterator&) noexcept = default;
        generic_iterator(generic_iterator&&) noexcept = default;
        generic_iterator& operator=(const generic_iterator&) noexcept = default;
        generic_iterator& operator=(generic_iterator&&) noexcept = default;
        ~generic_iterator() noexcept = default;

        explicit generic_iterator(object_iter iter)
            : iter_(iter)
        {
        }

        explicit generic_iterator(array_iter iter)
            : iter_(iter)
        {
        }

        template<bool other_const>
        requires(std::is_convertible_v<typename json_wrapper::template generic_iterator<other_const>::object_iter, object_iter> ||
                 std::is_convertible_v<typename json_wrapper::template generic_iterator<other_const>::array_iter, array_iter>)
        generic_iterator(const generic_iterator<other_const>& other)
        {
            if (other.is_object()) {
                iter_ = std::get<0>(other.iter_);
            } else if (other.is_array()) {
                iter_ = std::get<1>(other.iter_);
            }
        }

        inline bool is_object() const noexcept
        {
            return (iter_.index() == 0);
        }

        inline bool is_array() const noexcept
        {
            return (iter_.index() == 1);
        }

        inline auto* operator->() noexcept
        {
            if (is_array()) {
                return &*std::get<1>(iter_);
            }
            return &std::get<0>(iter_)->value();
        }

        inline auto* operator->() const noexcept
        {
            if (is_array()) {
                return &*std::get<1>(iter_);
            }
            return &std::get<0>(iter_)->value();
        }

        std::string_view key() const noexcept
        {
            if (is_object()) {
                return std::get<0>(iter_)->key();
            }
            return std::string_view{};
        }

        json_wrapper<is_const_iter> value() const noexcept
        {
            if (is_object()) {
                return json_wrapper<is_const_iter>(std::get<0>(iter_)->value());
            }
            return json_wrapper<is_const_iter>(*std::get<1>(iter_));
        }

        generic_iterator& operator++()
        {
            if (is_object()) {
                ++std::get<0>(iter_);
            } else if (is_array()) {
                ++std::get<1>(iter_);
            }
            return *this;
        }

        generic_iterator operator++(int)
        {
            generic_iterator temp = *this;
            ++(*this);
            return temp;
        }

        generic_iterator& operator+=(difference_type offset)
        {
            if (is_object()) {
                std::get<0>(iter_) += offset;
            } else if (is_array()) {
                std::get<1>(iter_) += offset;
            }
            return *this;
        }

        generic_iterator& operator--()
        {
            if (is_object()) {
                --std::get<0>(iter_);
            } else if (is_array()) {
                --std::get<1>(iter_);
            }
            return *this;
        }

        generic_iterator operator--(int)
        {
            generic_iterator temp = *this;
            --(*this);
            return temp;
        }

        friend difference_type operator-(const generic_iterator& lhs, const generic_iterator& rhs)
        {
            if (lhs.is_object()) {
                return std::get<0>(lhs.iter_) - std::get<0>(rhs.iter_);
            }
            return std::get<1>(lhs.iter_) - std::get<1>(rhs.iter_);
        }

        friend generic_iterator operator-(const generic_iterator& it, difference_type step)
        {
            auto tmp = it;
            tmp -= step;
            return tmp;
        }

        generic_iterator operator-=(difference_type step) const
        {
            auto tmp = *this;
            tmp -= step;
            return tmp;
        }

        generic_iterator& operator-=(difference_type step)
        {
            if (is_object()) {
                std::get<0>(iter_) -= step;
            } else if (is_array()) {
                std::get<1>(iter_) -= step;
            }
            return *this;
        }

        impl::key_value_pair<is_const_iter> operator*() const;

        template<bool other_iter_const>
        bool operator==(const generic_iterator<other_iter_const>& other) const noexcept
        {
            if (is_object()) {
                return std::get<0>(iter_) == std::get<0>(other.iter_);
            }
            return std::get<1>(iter_) == std::get<1>(other.iter_);
        }

        template<bool other_iter_const>
        bool operator==(const typename other_wrapper::template generic_iterator<other_iter_const>& other) const noexcept
        {
            if (is_object()) {
                return std::get<0>(iter_) == std::get<0>(other.iter_);
            }
            return std::get<1>(iter_) == std::get<1>(other.iter_);
        }

        std::variant<object_iter, array_iter> iter_;
    };

    using const_iterator = generic_iterator<true>;
    using iterator = std::conditional_t<is_const, const_iterator, generic_iterator<false>>;

    iterator begin()
    {
        if (is_object()) {
            return iterator(object_val().begin());
        } else if (is_array()) {
            return iterator(val_.as_array().begin());
        }
        return iterator{};
    }

    const_iterator begin() const
    {
        if (is_object()) {
            return const_iterator(object_val().begin());
        } else if (is_array()) {
            return const_iterator(val_.as_array().begin());
        }
        return const_iterator{};
    }

    iterator end()
    {
        if (is_object()) {
            return iterator(object_val().end());
        } else if (is_array()) {
            return iterator(val_.as_array().end());
        }
        return iterator{};
    }

    const_iterator end() const
    {
        if (is_object()) {
            return const_iterator(object_val().end());
        } else if (is_array()) {
            return const_iterator(val_.as_array().end());
        }
        return const_iterator{};
    }

    const_iterator cbegin()
    {
        if (is_object()) {
            return const_iterator(object_val().begin());
        } else if (is_array()) {
            return const_iterator(val_.as_array().begin());
        }
        return const_iterator{};
    }

    const_iterator cbegin() const
    {
        return begin();
    }

    const_iterator cend()
    {
        if (is_object()) {
            return const_iterator(object_val().end());
        } else if (is_array()) {
            return const_iterator(val_.as_array().end());
        }
        return const_iterator{};
    }

    const_iterator cend() const
    {
        return end();
    }

    bool contains(std::string_view key) const noexcept
    {
        return is_object() && object_val().find(key) != object_val().end();
    }

    bool contains(json_wrapper key) const noexcept
    {
        return (key.is_string() ? contains(key.value().as_string()) : false);
    }

    bool is_null() const noexcept
    {
        return value().is_null() || (is_object() && object_val().empty());
    }

    bool empty() const noexcept
    {
        return is_null() || (is_array() && val_.as_array().empty()) || (is_object() && object_val().empty());
    }

    bool is_object() const noexcept
    {
        return val_.is_object();
    }

    bool is_array() const noexcept
    {
        return val_.is_array();
    }

    bool is_string() const noexcept
    {
        return val_.is_string();
    }

    bool is_boolean() const noexcept
    {
        return val_.is_bool();
    }

    bool is_number_integer() const noexcept
    {
        return val_.is_int64();
    }

    bool is_number_unsigned() const noexcept
    {
        return val_.is_uint64();
    }

    bool is_number_float() const noexcept
    {
        return val_.is_double();
    }

    bool is_number() const noexcept
    {
        return val_.is_number();
    }

    bool is_structured() const noexcept
    {
        return val_.is_structured();
    }

    std::size_t size() const noexcept
    {
        if (is_array()) {
            return val_.as_array().size();
        } else if (is_object()) {
            return object_val().size();
        } else if (is_null()) {
            return 0;
        }
        return 1;
    }

    auto& object_val()
    {
        ASSERT(is_object());
        return val_.as_object();
    }

    const boost::json::object& object_val() const
    {
        ASSERT(is_object());
        return val_.as_object();
    }

    value_ref& value() noexcept
    {
        return val_;
    }

    const boost::json::value& value() const noexcept
    {
        return val_;
    }

    template<typename T>
    T get() const
    {
        if constexpr (base::arithmetic<T>) {
            if (is_number_integer()) {
                return static_cast<T>(val_.as_int64());
            } else if (is_number_unsigned()) {
                return static_cast<T>(val_.as_uint64());
            } else if (is_number_float()) {
                return static_cast<T>(val_.as_double());
            } else if (is_boolean()) {
                return static_cast<T>(val_.as_bool());
            }
        }
        return boost::json::value_to<T>(val_);
    }

    template<typename T>
    void get_to(T& t) const
    {
        t = get<T>();
    }

    template<typename T>
    explicit operator T() const requires (!std::is_same_v<T, json_wrapper<true>> && !std::is_same_v<T, json_wrapper<false>>)
    {
        return get<T>();
    }

    template<typename T>
    explicit operator std::vector<T>() const
    {
        ASSERT(!is_object());
        ASSERT(is_array());
        std::vector<T> res;
        const auto& arr = value().as_array();
        res.reserve(arr.size());
        for (const auto& i : arr) {
            res.push_back(json_wrapper<true>(i).get<T>());
        }
        return res;
    }

    template<typename T, std::size_t N>
    explicit operator std::array<T, N>() const
    {
        ASSERT(!is_object());
        ASSERT(is_array());
        std::array<T, N> res;
        const auto& arr = value().as_array();
        for (std::size_t i = 0u; i < arr.size(); ++i) {
            res[i] = json_wrapper(arr[i]).get<T>();
        }
        return res;
    }

    bool operator==(const json_wrapper& other) const noexcept
    {
        return val_ == other.val_;
    }
 
    bool operator==(std::string_view sv) const noexcept
    {
        return is_string() ? (val_.as_string() == sv) : false;
    }

    json_wrapper operator[](std::string_view sv)
    {
        if constexpr (is_const) {
            if (val_.is_object()) {
                auto it = val_.as_object().find(sv);
                if (it != val_.as_object().end()) {
                    return json_wrapper(it->value());
                }
            }
            static const boost::json::value s_null;
            return json_wrapper<true>(s_null);
        } else {
            if (!is_object()) {
                val_.emplace_object();
            }
            if (auto it = object_val().find(sv); it != object_val().end()) {
                return json_wrapper(it->value());
            }
            return json_wrapper(object_val()[sv]);
        }
    }

    json_wrapper<true> operator[](std::string_view sv) const
    {
        ASSERT(is_object());
        return json_wrapper<true>(object_val().at(sv));
    }

    json_wrapper operator[](json_wrapper<true> w)
    {
        ASSERT(w.is_string());
        return operator [](w.value().as_string());
    }

    json_wrapper<true> operator[](json_wrapper<true> w) const
    {
        ASSERT(w.is_string());
        return operator [](w.value().as_string());
    }

    json_wrapper operator[](const impl::key_value_pair<true>& kv);

    json_wrapper operator[](const impl::key_value_pair<true>& kv) const;
    
    json_wrapper operator[](std::size_t i)
    {
        ASSERT(is_array());
        return json_wrapper(val_.as_array()[i]);
    }

    json_wrapper operator[](std::size_t i) const
    {
        ASSERT(is_array());
        return json_wrapper(val_.as_array()[i]);
    }

    json_wrapper at(std::string_view sv)
    {
        ASSERT(is_object());
        return json_wrapper(object_val().at(sv));
    }

    json_wrapper at(std::string_view sv) const
    {
        return operator[](sv);
    }

    json_wrapper at(std::size_t i)
    {
        return operator[](i);
    }

    json_wrapper at(std::size_t i) const
    {
        return operator[](i);
    }

    std::string dump([[maybe_unused]] int level = 0) const
    {
        if (is_object()) {
            return boost::json::serialize(object_val());
        } else if (is_array()) {
            return boost::json::serialize(value().as_array());
        } else if (is_string()) {
            return value().as_string().c_str();
        } else if (is_boolean()) {
            return value().as_bool() ? "true" : "false";
        } else if (is_number_integer()) {
            return std::to_string(value().as_int64());
        } else if (is_number_unsigned()) {
            return std::to_string(value().as_uint64());
        } else if (is_number_float()) {
            return std::to_string(value().as_double());
        }
        return std::string{};
    }

    template <typename T>
    T value(std::string_view key, T default_value) const
    {
        ASSERT(is_object());
        if (auto it = object_val().find(key); it != object_val().end()) {
            return json_wrapper<true>(it->value()).get<T>();
        }
        return default_value;
    }

    auto items() const
    {
        return *this;
    }

    iterator find(std::string_view k)
    {
        if (!is_object()) {
            return end();
        }
        return iterator(object_val().find(k));
    }

    const_iterator find(std::string_view k) const
    {
        if (!is_object()) {
            return end();
        }
        return const_iterator(object_val().find(k));
    }

    iterator erase(const_iterator pos) noexcept requires (!is_const)
    {
        if (is_object()) {
            return iterator(object_val().erase(std::get<0>(pos.iter_)));
        } else if (is_array()) {
            return iterator(val_.as_array().erase(std::get<1>(pos.iter_)));
        }
        return end();
    }

    std::size_t erase(std::string_view key) noexcept requires (!is_const)
    {
        ASSERT(is_object());
        return object_val().erase(key);
    }

    void clear() noexcept requires (!is_const)
    {
        if constexpr (!is_const) {
            val_ = boost::json::value();
            val_.emplace_object();
        } else {
            ASSERT(false);
        }
    }

    template<typename T>
    void emplace(std::string_view k, T&& v) requires (!is_const)
    {
        if constexpr (!is_const) {
            val_.emplace_object().emplace(k, std::forward<T>(v));
        } else {
            ASSERT(false);
        }
    }

    template<typename T>
    void push_back(const T& t) requires (!is_const)
    {
        if constexpr (!is_const) {
            if (!val_.is_array()) {
                val_.emplace_array();
            }
            val_.as_array().push_back(boost::json::value(t));
        } else {
            ASSERT(false);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const json_wrapper& j)
    {
        os << j.dump();
        return os;
    }

private:
    value_ref val_;
};

using const_json_wrapper = json_wrapper<true>;
using mutable_json_wrapper = json_wrapper<false>;
using const_wrapper_iterator = const_json_wrapper::generic_iterator<false>;
using const_wrapper_const_iterator = const_json_wrapper::generic_iterator<true>;
using mutable_wrapper_iterator = mutable_json_wrapper::generic_iterator<false>;
using mutable_wrapper_const_iterator = mutable_json_wrapper::generic_iterator<true>;

namespace impl {

template <bool is_const>
class key_value_pair
{
private:
    using value_ref = std::conditional_t<is_const, const boost::json::value&, boost::json::value&>;
    using other_value_ref = std::conditional_t<is_const, boost::json::value&, const boost::json::value&>;
    using key_value_ref = std::conditional_t<is_const, const boost::json::key_value_pair&, boost::json::key_value_pair&>;
    using other_key_value_ref = std::conditional_t<is_const, boost::json::key_value_pair&, const boost::json::key_value_pair&>;
    using value_wrapper = std::conditional_t<is_const, const const_json_wrapper, mutable_json_wrapper>;

public:
    std::string key_;
    value_wrapper val_;

    template<typename T>
    requires (std::is_same_v<T, key_value_pair<true>> || std::is_same_v<T, key_value_pair<false>>)
    key_value_pair(const T& other)
        : key_(other.key_)
        , val_(other.val_)
    {
    }

    explicit key_value_pair(const boost::json::key_value_pair& kv) requires (is_const)
        : key_(kv.key())
        , val_(kv.value())
    {
    }

    explicit key_value_pair(boost::json::key_value_pair& kv)
        : key_(kv.key())
        , val_(kv.value())
    {
    }

    explicit key_value_pair(const boost::json::value& v) requires (is_const)
        : val_(v)
    {
    }

    explicit key_value_pair(boost::json::value& v)
        : val_(v)
    {
    }

    bool is_object() const noexcept
    {
        return key_.empty();
    }

    const std::string& key() const noexcept
    {
        return key_;
    }

    auto value() const noexcept
    {
        return val_;
    }

    json_wrapper<is_const> operator[](std::string_view sv);
    json_wrapper<is_const> operator[](std::string_view sv) const;
    json_wrapper<is_const> operator[](const_json_wrapper w);
    json_wrapper<is_const> operator[](const_json_wrapper w) const;
    json_wrapper<is_const> operator[](const key_value_pair<true>& kv);
    json_wrapper<is_const> operator[](const key_value_pair<true>& kv) const;
    json_wrapper<is_const> operator[](std::size_t i);
    json_wrapper<is_const> operator[](std::size_t i) const;

    bool contains(std::string_view) const;
    bool contains(const_json_wrapper) const;
};

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](std::string_view sv)
{
    return json_wrapper<is_const>(value()).operator[](sv);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](std::string_view sv) const
{
    return json_wrapper<is_const>(value()).operator[](sv);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](const_json_wrapper w)
{
    return json_wrapper<is_const>(value()).operator[](w);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](const_json_wrapper w) const
{
    return json_wrapper<is_const>(value()).operator[](w);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](const key_value_pair<true>& kv)
{
    return json_wrapper<is_const>(value()).operator[](kv);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](const key_value_pair<true>& kv) const
{
    return json_wrapper<is_const>(value()).operator[](kv);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](std::size_t i)
{
    return json_wrapper<is_const>(value()).operator[](i);
}

template<bool is_const>
inline json_wrapper<is_const> key_value_pair<is_const>::operator[](std::size_t i) const
{
    return json_wrapper<is_const>(value()).operator[](i);
}

template<bool is_const>
inline bool key_value_pair<is_const>::contains(std::string_view key) const
{
    return json_wrapper<is_const>(value()).contains(key);
}

template<bool is_const>
inline bool key_value_pair<is_const>::contains(const_json_wrapper key) const
{
    return json_wrapper<is_const>(value()).contains(key);
}

} /// impl namespace

template<bool is_const>
inline json_wrapper<is_const> json_wrapper<is_const>::operator[](const impl::key_value_pair<true>& kv)
{
    return operator [](json_wrapper<is_const>(kv.value()));
}

template<bool is_const>
inline json_wrapper<is_const> json_wrapper<is_const>::operator[](const impl::key_value_pair<true>& kv) const
{
    return operator [](json_wrapper<is_const>(kv.value()));
}

template<bool is_const>
template<bool is_const_iter>
impl::key_value_pair<is_const_iter> json_wrapper<is_const>::generic_iterator<is_const_iter>::operator*() const
{
    if (is_object()) {
        return impl::key_value_pair<is_const_iter>(*std::get<0>(iter_));
    }
    return impl::key_value_pair<is_const_iter>(*std::get<1>(iter_));
}

} /// icm namespace
