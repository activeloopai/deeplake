#pragma once

#include "../adapt.hpp"
#include "../cast.hpp"
#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "../shape_utils.hpp"
#include "chained_iterator.hpp"

#include <icm/shape.hpp>

namespace nd::impl {

template <typename T, bool compare, typename O>
array create_binary_kernel(array a, array b, O op);

template <typename T, typename O, bool compare>
class full_dynamic_binary_kernel_expression
{
public:
    using value_type = std::invoke_result_t<O, T, T>;

public:
    full_dynamic_binary_kernel_expression(array a, array b, O op)
        : a_(std::move(a))
        , b_(std::move(b))
        , op_(op)
    {
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    icm::shape shape() const
    {
        return a_.shape();
    }

    value_type value(int64_t index) const
    {
        return op_(a_.value<T>(index), b_.value<T>(index));
    }

    static auto converter(O op)
    {
        return [op = std::move(op)](array a, array b) {
            return create_binary_kernel<T, compare>(std::move(a), std::move(b), op);
        };
    }

    using iterator = chained_iterator<decltype(converter(std::declval<O>())), nd::iterator, nd::iterator>;

    iterator begin() const
    {
        return iterator(converter(op_), a_.begin(), b_.begin());
    }

    iterator end() const
    {
        return iterator(converter(op_), a_.end(), b_.end());
    }

    array get(int64_t index) const
    {
        return create_binary_kernel<T, compare>(a_[index], b_[index], op_);
    }

    uint8_t dimensions() const
    {
        return std::max(a_.dimensions(), b_.dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    array a_;
    array b_;
    O op_;
};

template <typename T, bool is_dynamic_, typename O, bool compare>
class binary_kernel_expression
{
public:
    using value_type = std::invoke_result_t<O, T, T>;

public:
    binary_kernel_expression(array a, array b, O op)
        : a_(std::move(a))
        , b_(std::move(b))
        , op_(op)
    {
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    icm::shape shape() const
    {
        return a_.shape();
    }

    value_type value(int64_t index) const
    {
        return op_(a_.value<T>(index), b_.value<T>(index));
    }

    static auto converter(O op)
    {
        return [op = std::move(op)](nd::array a, nd::array b) {
            return create_binary_kernel<T, compare>(std::move(a), std::move(b), op);
        };
    }

    using iterator = chained_iterator<decltype(converter(std::declval<O>())), nd::iterator, nd::iterator>;

    iterator begin() const
    {
        return iterator(converter(op_), a_.begin(), b_.begin());
    }

    iterator end() const
    {
        return iterator(converter(op_), a_.end(), b_.end());
    }

    array get(int64_t index) const
    {
        return create_binary_kernel<T, compare>(a_[index], b_[index], op_);
    }

    array eval() const;

    uint8_t dimensions() const
    {
        return std::max(a_.dimensions(), b_.dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return is_dynamic_;
    }

private:
    array a_;
    array b_;
    O op_;
};

template <typename T, bool is_dynamic_, typename O, bool first, bool compare>
class binary_kernel_expression_scalar
{
public:
    using value_type = std::invoke_result_t<O, T, T>;

public:
    binary_kernel_expression_scalar(array a, array b, O op)
        : a_(std::move(a))
        , b_(std::move(b))
        , op_(op)
    {
    }

    enum dtype dtype() const
    {
        return dtype_enum_v<value_type>;
    }

    icm::shape shape() const
    {
        if constexpr (first) {
            return b_.shape();
        } else {
            return a_.shape();
        }
    }

    value_type value(int64_t index) const
    {
        if constexpr (first) {
            return op_(a_.value<T>(0), b_.value<T>(index));
        } else {
            return op_(a_.value<T>(index), b_.value<T>(0));
        }
    }

    static auto converter(O op, nd::array scalar)
    {
        if constexpr (first) {
            return [op = std::move(op), a = std::move(scalar)](nd::array b) {
                return create_binary_kernel<T, compare>(a, std::move(b), op);
            };
        } else {
            return [op = std::move(op), b = std::move(scalar)](nd::array a) {
                return create_binary_kernel<T, compare>(std::move(a), b, op);
            };
        }
    }

    using iterator = chained_iterator<decltype(converter(*static_cast<O*>(nullptr), nd::array())), nd::iterator>;

    iterator begin() const
    {
        if constexpr (first) {
            return iterator(converter(op_, a_), b_.begin());
        } else {
            return iterator(converter(op_, b_), a_.begin());
        }
    }

    iterator end() const
    {
        if constexpr (first) {
            return iterator(converter(op_, a_), b_.end());
        } else {
            return iterator(converter(op_, b_), a_.end());
        }
    }

    array get(int64_t index) const
    {
        if constexpr (first) {
            return create_binary_kernel<T, compare>(nd::array(a_), b_[index], op_);
        } else {
            return create_binary_kernel<T, compare>(a_[index], b_, op_);
        }
    }

    array eval() const;

    uint8_t dimensions() const
    {
        return first ? b_.dimensions() : a_.dimensions();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return is_dynamic_;
    }

private:
    array a_;
    array b_;
    O op_;
};

template <typename T, bool compare, typename O>
inline array create_numeric_binary_kernel(array a, array b, O op)
{
    if (a.is_none() || b.is_none()) {
        return a;
    }
    if (a.dimensions() == 0 && b.dimensions() == 0) {
        return adapt(op(a.value<T>(0), b.value<T>(0)));
    } else if (b.dimensions() == 0) {
        if (a.is_dynamic()) {
            return array(binary_kernel_expression_scalar<T, true, O, false, compare>(std::move(a), std::move(b), op));
        } else if (a.volume() == 1) {
            return adapt(op(a.value<T>(0), b.value<T>(0)));
        } else {
            return array(binary_kernel_expression_scalar<T, false, O, false, compare>(std::move(a), std::move(b), op));
        }
    } else if (a.dimensions() == 0) {
        if (b.is_dynamic()) {
            return array(binary_kernel_expression_scalar<T, true, O, true, compare>(std::move(a), std::move(b), op));
        } else if (b.volume() == 1) {
            return adapt(op(a.value<T>(0), b.value<T>(0)));
        } else {
            return array(binary_kernel_expression_scalar<T, false, O, true, compare>(std::move(a), std::move(b), op));
        }
    } else if (a.shape() == b.shape()) {
        if (a.is_dynamic() && b.is_dynamic()) {
            return array(full_dynamic_binary_kernel_expression<T, O, compare>(std::move(a), std::move(b), op));
        } else if (a.is_dynamic() || b.is_dynamic()) {
            return array(binary_kernel_expression<T, true, O, compare>(std::move(a), std::move(b), op));
        } else {
            return array(binary_kernel_expression<T, false, O, compare>(std::move(a), std::move(b), op));
        }
    }
    throw invalid_operation("Can't apply operation on arrays with different shapes.");
}

template <typename T, bool compare, typename O>
array create_binary_kernel(array a, array b, O op)
{
    if constexpr (std::is_same_v<std::string_view, T>) {
        a = nd::eval(a);
        b = nd::eval(b);
        if constexpr (compare) {
            return nd::adapt(op(base::string_view_cast(a.template data<const char>()),
                                base::string_view_cast(b.template data<const char>())));
        } else {
            throw unsupported_operator();
        }
    } else {
        return create_numeric_binary_kernel<T, compare>(std::move(a), std::move(b), op);
    }
}

template <typename T, bool is_dynamic_, typename O, bool compare>
inline array binary_kernel_expression<T, is_dynamic_, O, compare>::eval() const
{
    nd::array a = nd::eval(nd::cast<dtype_enum_v<T>>(a_));
    nd::array b = nd::eval(nd::cast<dtype_enum_v<T>>(b_));
    auto sp1 = a.data<T>();
    auto sp2 = b.data<T>();
    ASSERT(sp1.size() == sp2.size());
    icm::vector<value_type> res;
    res.reserve(sp1.size());
    for (auto i = 0; i < sp1.size(); ++i) {
        res.push_back(op_(sp1[i], sp2[i]));
    }
    return nd::adapt(std::move(res), a.shape());
}

template <typename T, bool is_dynamic_, typename O, bool first, bool compare>
inline array binary_kernel_expression_scalar<T, is_dynamic_, O, first, compare>::eval() const
{
    if constexpr (first) {
        auto a = a_.value<T>(0);
        nd::array b = nd::eval(nd::cast<dtype_enum_v<T>>(b_));
        auto sp = b.template data<T>();
        icm::vector<value_type> res;
        res.reserve(sp.size());
        for (auto i = 0; i < sp.size(); ++i) {
            res.push_back(op_(a, sp[i]));
        }
        return nd::adapt(std::move(res), b.shape());
    } else {
        nd::array a = nd::eval(nd::cast<dtype_enum_v<T>>(a_));
        auto b = b_.value<T>(0);
        auto sp = a.template data<T>();
        icm::vector<value_type> res;
        res.reserve(sp.size());
        for (auto i = 0; i < sp.size(); ++i) {
            res.push_back(op_(sp[i], b));
        }
        return nd::adapt(std::move(res), a.shape());
    }
}

} // namespace nd::impl
