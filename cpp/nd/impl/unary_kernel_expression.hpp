#pragma once

#include "../adapt.hpp"
#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "../shape_utils.hpp"
#include "../operator_type.hpp"
#include "chained_iterator.hpp"

#include <icm/shape.hpp>
#include <icm/vector.hpp>

namespace nd::impl {

template <typename U, typename O>
array create_unary_kernel(array&& a, O&& op);

template <typename U, typename O>
class unary_kernel_expression
{
    static_assert(!std::is_reference_v<O>);
    static_assert(!std::is_const_v<O>);

public:
    using value_type = std::invoke_result_t<O, U>;

public:
    unary_kernel_expression(array&& a, O op)
        : a_(std::move(a))
        , op_(std::move(op))
    {
    }

    unary_kernel_expression(const unary_kernel_expression&) = default;
    unary_kernel_expression& operator=(const unary_kernel_expression&) = delete;
    unary_kernel_expression(unary_kernel_expression&&) noexcept = default;
    unary_kernel_expression& operator=(unary_kernel_expression&&) noexcept = default;

    ~unary_kernel_expression() = default;

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
        return op_(a_.value<U>(index));
    }

    static auto converter(O op)
    {
        return [op = std::move(op)](nd::array v) {
            return create_unary_kernel<U>(std::move(v), op);
        };
    }

    using iterator = chained_iterator<decltype(converter(*static_cast<O*>(nullptr))), nd::iterator>;

    iterator begin() const
    {
        return iterator(converter(op_), a_.begin());
    }

    iterator end() const
    {
        return iterator(converter(op_), a_.end());
    }

    array get(int64_t index) const
    {
        return converter(op_)(a_[index]);
    }

    array eval() const
    {
        auto a = ::nd::eval(a_);
        auto sp = a.template data<U>();
        icm::vector<value_type> res;
        res.reserve(sp.size());
        for (auto v : sp) {
            res.push_back(op_(v));
        }
        return ::nd::adapt(std::move(res), a.shape());
    }

    void copy_data(std::span<uint8_t> buffer) const
    {
        auto a = ::nd::eval(a_);
        auto sp = a.template data<U>();
        auto res = base::span_cast<value_type>(buffer);
        ASSERT(res.size() == sp.size());
        for (auto i = 0; i < sp.size(); ++i) {
            res[i] = op_(sp[i]);
        }
    }

    uint8_t dimensions() const
    {
        return a_.dimensions();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return a_.is_dynamic();
    }

private:
    array a_;
    O op_;
};

template <typename U, typename O>
inline array create_unary_kernel(array&& a, O&& op)
{
    return array(unary_kernel_expression<U, std::decay_t<O>>(std::move(a), std::forward<O>(op)));
}

}
