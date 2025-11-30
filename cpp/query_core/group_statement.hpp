#pragma once

#include "exceptions.hpp"
#include "meta_functor.hpp"
#include "variants.hpp"

#include <base/assert.hpp>
#include <query_core/functor.hpp>

#include <cstdint>
#include <memory>

namespace query_core {

enum class across_type_t : uint8_t
{
    time,
    space
};

class group_entry
{
public:
    group_entry() = default;

    inline explicit group_entry(order_functor f)
        : func_(std::move(f))
    {
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

    inline const expr& get_expr() const
    {
        return func_.get_expr();
    }

private:
    order_functor func_;
};

class group_statement
{
public:
    group_statement() = default;

    group_statement(across_type_t type, std::vector<group_entry>&& e) : across_type_(type), entries_(std::move(e))
    {
    }

    inline const auto& entries() const
    {
        return entries_;
    }

    inline auto& entries()
    {
        return entries_;
    }

    inline auto across_type() const
    {
        return across_type_;
    }

    bool empty() const
    {
        return entries_.empty();
    }

    inline auto begin()
    {
        return entries_.begin();
    }

    inline auto end()
    {
        return entries_.end();
    }

    inline const auto begin() const
    {
        return entries_.begin();
    }

    inline const auto end() const
    {
        return entries_.end();
    }

    std::string to_string() const
    {
        std::string result;
        for (const auto& e : entries_) {
            result += fmt::format("{} ", e.get_expr().to_string());
        }
        return result;
    }

private:
    across_type_t across_type_ = across_type_t::time;
    std::vector<group_entry> entries_;
};

} // namespace query_core
