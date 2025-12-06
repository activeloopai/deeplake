#pragma once

#include "exceptions.hpp"
#include "variants.hpp"

#include <query_core/functor.hpp>
#include <base/assert.hpp>

#include <cstdint>
#include <memory>
#include <variant>

namespace query_core {

struct sample_limit_statement
{
    sample_limit_statement()
        : value(100.0)
    {
    }

    sample_limit_statement(double v)
        : value(v)
    {
    }

    sample_limit_statement(int64_t v)
        : value(v)
    {
    }

    inline int64_t absolute_count(int64_t total_size) const
    {
        if (std::holds_alternative<double>(value)) {
            return std::get<double>(value) * static_cast<double>(total_size) / 100.0;
        }
        return std::get<int64_t>(value);
    }

    std::variant<int64_t, double> value;
};

class sample_statement
{
public:
    inline sample_statement() : limit_()
    {
    }

    inline sample_statement(sample_limit_statement l, bool r)
        : limit_(l)
        , can_repeat_(r)
    {
    }

    inline const auto& limit() const
    {
        return limit_;
    }

    inline auto can_repeat() const
    {
        return can_repeat_;
    }

    inline int64_t absolute_count(int64_t total_size) const
    {
        auto c = limit_.absolute_count(total_size);
        if (can_repeat()) {
            return c;
        }
        return std::min(total_size, limit_.absolute_count(total_size));
    }

    inline void set_func(query_core::generic_functor<double> f)
    {
        func_ = std::move(f);
    }

    inline const auto& get_func() const
    {
        return func_;
    }

    inline bool has_func() const
    {
        return static_cast<bool>(func_);
    }

    inline bool is_trivial() const
    {
        return !has_func();
    }

    const expr& get_expr() const
    {
        return func_.get_expr();
    }

private:
    query_core::generic_functor<double> func_;
    sample_limit_statement limit_;
    bool can_repeat_ = false;
};

} // namespace query_core
