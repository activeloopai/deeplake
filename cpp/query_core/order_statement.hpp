#pragma once

#include "exceptions.hpp"
#include "meta_functor.hpp"
#include "order_type.hpp"
#include "variants.hpp"

#include <base/assert.hpp>
#include <query_core/functor.hpp>

#include <cstdint>
#include <memory>
#include <variant>

namespace query_core {

class order_statement
{
public:
    inline order_statement() : type_(query_core::order_type::ascending)
    {
    }

    inline auto type() const
    {
        return type_;
    }

    inline void set_type(query_core::order_type t)
    {
        type_ = t;
    }

    inline void set_func(order_functor f)
    {
        func_ = std::move(f);
    }

    inline bool has_func() const
    {
        return func_.has_func();
    }

    template <typename F>
    inline auto switch_func(F f) const
    {
        return func_.switch_func(std::move(f));
    }

    const expr& get_expr() const
    {
        return func_.get_expr();
    }

private:
    order_functor func_;
    query_core::order_type type_;
};

} // namespace query_core
