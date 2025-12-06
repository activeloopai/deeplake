#pragma once

#include "group_statement.hpp"
#include "order_statement.hpp"
#include "sample_statement.hpp"
#include "ungroup_statement.hpp"
#include "select_expr.hpp"

#include <cstdint>
#include <memory>
#include <tuple>

namespace query_core {

enum class set_operation_type : uint8_t
{
    union_,
    intersect_,
    difference_
};

struct limit_statement
{
    limit_statement()
        : offset(0)
        , limit(std::numeric_limits<int64_t>::max())
    {}

    limit_statement(int64_t o, int64_t l)
        : offset(o)
        , limit(l)
    {}

    explicit operator bool() const
    {
        return !(offset == 0 && limit == std::numeric_limits<int64_t>::max());
    }

    int64_t offset;
    int64_t limit;
};

struct where_statement
{
    where_statement() = default;

    explicit where_statement(query_core::generic_functor<nd::array> p)
        : predicate(std::move(p))
    {}

    explicit operator bool() const
    {
        return static_cast<bool>(predicate.get_expr());
    }

    auto to_string() const
    {
        return predicate.get_expr().to_string();
    }

    query_core::generic_functor<nd::array> predicate;
};

struct set_operation
{
    set_operation()
        : type(set_operation_type::union_)
    {}

    set_operation(set_operation_type t, limit_statement l)
        : type(t)
        , limit(std::move(l))
    {}

    set_operation(set_operation_type t, order_statement o, limit_statement l)
        : type(t)
        , order(std::move(o))
        , limit(std::move(l))
    {}

    set_operation_type type;
    order_statement order;
    limit_statement limit;
};

struct statement
{
    statement(where_statement w, order_statement o, limit_statement l,
              sample_statement ss, group_statement g, ungroup_statement u, std::vector<select_expr> sl)
        : select_list(std::move(sl))
        , where(std::move(w))
        , order(std::move(o))
        , limit(std::move(l))
        , sampler(std::move(ss))
        , group(std::move(g))
        , ungroup(std::move(u))
    {}

    std::vector<select_expr> select_list;
    where_statement where;
    order_statement order;
    limit_statement limit;
    sample_statement sampler;
    group_statement group;
    ungroup_statement ungroup;
};

}
