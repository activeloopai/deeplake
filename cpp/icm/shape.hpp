#pragma once

#include "exceptions.hpp"
#include "impl/mpl.hpp"
#include "index_based_iterator.hpp"
#include "small_vector.hpp"

#include <base/overloads.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace icm {

class shape
{
public:
    using value_type = int64_t;

private:
    using vector_ptr = std::shared_ptr<std::vector<value_type>>;

    struct unknown_shape_tag
    {
    };

    using data_type = std::variant<std::monostate,
                                   value_type,
                                   std::array<value_type, 2>,
                                   std::array<value_type, 3>,
                                   vector_ptr,
                                   unknown_shape_tag>;

    template <typename... Ts>
    decltype(auto) visit_this(Ts&&... args) const
    {
        return std::visit(base::overloads{std::forward<Ts>(args)...}, shape_data_);
    }

    template <typename... Ts>
    decltype(auto) visit_this(Ts&&... args)
    {
        return std::visit(base::overloads{std::forward<Ts>(args)...}, shape_data_);
    }

    shape(data_type data)
        : shape_data_(std::move(data))
    {
    }

public:
    shape() = default;
    shape(const shape&) = default;
    shape(shape&&) = default;
    shape& operator=(const shape&) = default;
    shape& operator=(shape&&) = default;

    explicit shape(const icm::const_json& j)
        : shape(j.template get<std::vector<value_type>>())
    {
    }

    template <typename T>
    requires mpl::is_numeric_integral_v<T>
    explicit shape(T value)
        : shape_data_(static_cast<value_type>(value))
    {
    }

    template <typename T>
    requires std::is_convertible_v<T, std::span<const typename T::value_type>> &&
             mpl::is_numeric_integral_v<typename T::value_type>
    explicit shape(const T& list)
        : shape(std::span<const typename T::value_type>(list))
    {
    }

    template <typename T>
    requires mpl::is_numeric_integral_v<T>
    explicit shape(std::initializer_list<T>&& init_list)
        : shape(std::span<const T>(init_list))
    {
    }

    template <typename T>
    requires mpl::is_numeric_integral_v<T>
    explicit shape(std::span<const T> list)
        : shape(list.begin(), list.end())
    {
    }

    template <typename T>
    requires(std::is_pointer_v<T> && mpl::is_numeric_integral_v<typename std::pointer_traits<T>::element_type>) ||
            (mpl::is_one_of_v<typename T::iterator_category,
                              std::contiguous_iterator_tag,
                              std::random_access_iterator_tag> &&
             mpl::is_numeric_integral_v<typename T::value_type>)
    shape(T begin, T end)
    {
        auto size = std::distance(begin, end);
        switch (size) {
        case 0:
            shape_data_ = std::monostate{};
            break;
        case 1:
            if (*begin == unknown_shape_value) {
                shape_data_ = unknown_shape_tag{};
                break;
            }
            shape_data_ = static_cast<value_type>(*begin);
            break;
        case 2:
            shape_data_ =
                std::array<value_type, 2>({static_cast<value_type>(*begin), static_cast<value_type>(*(begin + 1))});
            break;
        case 3:
            shape_data_ = std::array<value_type, 3>{static_cast<value_type>(*begin),
                                                    static_cast<value_type>(*(begin + 1)),
                                                    static_cast<value_type>(*(begin + 2))};
            break;
        default:
            shape_data_ = std::make_shared<std::vector<value_type>>(size);
            std::transform(begin, end, std::begin(*std::get<vector_ptr>(shape_data_)), [](auto value) {
                return static_cast<value_type>(value);
            });
            break;
        }
    }

    static shape unknown()
    {
        return shape(unknown_shape_tag{});
    }

    icm::json to_json() const
    {
        return visit_this(
            [](std::monostate) -> icm::json {
                return icm::json::array();
            },
            [](value_type v) -> icm::json {
                return std::array{v};
            },
            [](const vector_ptr& value) -> icm::json {
                return icm::json(*value);
            },
            [](const unknown_shape_tag&) -> icm::json {
                return std::array{unknown_shape_value};
            },
            [](const auto& value) -> icm::json {
                return icm::json(value);
            });
    }

    std::string to_string() const;

    value_type at(int64_t index) const
    {
        if (index < 0 || index >= size()) {
            throw out_of_range("index out of range");
        }
        return this->operator[](index);
    }

    value_type operator[](int64_t index) const noexcept
    {
        return visit_this(
            [](std::monostate) {
                ASSERT(false);
                return value_type{};
            },
            [](value_type value) {
                return value;
            },
            [index](const vector_ptr& value) {
                return value->operator[](index);
            },
            [](const unknown_shape_tag&) {
                ASSERT(false);
                return value_type{};
            },
            [index](const auto& value) -> value_type {
                return value[index];
            });
    }

    value_type front() const
    {
        return this->operator[](0);
    }

    value_type back() const
    {
        return this->operator[](size() - 1);
    }

    int64_t size() const noexcept
    {
        return visit_this(
            [](std::monostate) -> value_type {
                return 0;
            },
            [](value_type) -> value_type {
                return 1;
            },
            [](const vector_ptr& value) -> value_type {
                return value->size();
            },
            [](const unknown_shape_tag&) -> value_type {
                ASSERT(false);
                return 0;
            },
            [](const auto& value) -> value_type {
                return std::size(value);
            });
    }

    int64_t size_bytes() const
    {
        return size() * sizeof(value_type);
    }

    const value_type* data() const noexcept
    {
        return const_cast<const value_type*>(const_cast<shape*>(this)->data());
    }

    value_type* data() noexcept
    {
        return visit_this(
            [](std::monostate) -> value_type* {
                return nullptr;
            },
            [](value_type& value) {
                return &value;
            },
            [](vector_ptr& value) {
                return value->data();
            },
            [](unknown_shape_tag&) -> value_type* {
                ASSERT(false);
                return nullptr;
            },
            [](auto& value) -> value_type* {
                return value.data();
            });
    }

    bool empty() const noexcept
    {
        return std::holds_alternative<std::monostate>(shape_data_);
    }

    bool is_known() const noexcept
    {
        return std::ranges::all_of(*this, [](auto x) {
            return x != dynamic;
        });
    }

    bool is_unknown() const noexcept
    {
        return std::holds_alternative<unknown_shape_tag>(shape_data_);
    }

    bool operator==(const shape& other) const noexcept
    {
        return shape_data_.index() == other.shape_data_.index() &&
               visit_this(
                   [&other](const vector_ptr& value) {
                       return *value == *std::get<vector_ptr>(other.shape_data_);
                   },
                   [&other](const unknown_shape_tag&) {
                       return std::holds_alternative<unknown_shape_tag>(other.shape_data_);
                   },
                   [&other]<typename T>(const T& value) {
                       return std::get<T>(other.shape_data_) == value;
                   });
    }

    using const_iterator = index_based_iterator<shape, value_type, use_container_index_tag, value_type>;

    const_iterator begin() const noexcept
    {
        return shape::const_iterator(*this, 0);
    }

    const_iterator begin() noexcept
    {
        return shape::const_iterator(*this, 0);
    }

    const_iterator end() const noexcept
    {
        return shape::const_iterator(*this, size());
    }

    const_iterator end() noexcept
    {
        return shape::const_iterator(*this, size());
    }

public:
    static constexpr value_type unknown_shape_value = -2;
    static constexpr value_type dynamic = -1;

private:
    data_type shape_data_;
};

inline shape::const_iterator begin(const shape& s) noexcept
{
    return s.begin();
}

inline shape::const_iterator end(const shape& s) noexcept
{
    return s.end();
}

inline shape operator+(shape::value_type v, const shape& s)
{
    small_vector<shape::value_type> shape_values;
    shape_values.push_back(v);
    shape_values.insert(shape_values.end(), s.begin(), s.end());
    return shape(shape_values.begin(), shape_values.end());
}

inline std::ostream& operator<<(std::ostream& stream, const shape& s)
{
    if (s.empty()) {
        stream << "()";
        return stream;
    }
    if (s.is_unknown()) {
        stream << "(unknown)";
        return stream;
    }
    auto x = s[0] == shape::dynamic ? std::string("None") : std::to_string(s[0]);
    stream << '(' << x;
    std::accumulate(s.begin() + 1, s.end(), &stream, [](auto* stream, auto x) -> std::ostream* {
        auto s = x == shape::dynamic ? std::string("None") : std::to_string(x);
        (*stream) << ", " << s;
        return stream;
    });
    stream << ')';
    return stream;
}

inline std::string shape::to_string() const
{
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

} // namespace icm
