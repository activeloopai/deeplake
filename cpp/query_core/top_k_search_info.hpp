#pragma once

#include "expr.hpp"
#include "order_type.hpp"
#include "search_config.hpp"
#include "static_data_t.hpp"

// Query-core wrapper types for Top-K search info. Uses query_core expr/order types.
#include <nd/array.hpp>

#include <functional>
#include <string>
#include <vector>

namespace query_core {

struct top_k_search_info
{
    top_k_search_info() = default;

    top_k_search_info(expr filter_expr, expr order_expr, int k, order_type t)
        : batch_params()
        , filter_expr(std::move(filter_expr))
        , order_expr(std::move(order_expr))
        , k(k)
        , t(t)
    {}

    bool has_value() const { return k > 0; }

    nd::array batch_params;
    expr filter_expr;
    expr order_expr;
    int k = 0;
    order_type t = order_type::ascending;

    // Search configuration for this query
    search_config config = search_config::default_config();
};

struct top_k_binary_function_search_info
{
    top_k_binary_function_search_info() = default;

    top_k_binary_function_search_info(
        std::string t, std::string f, std::function<nd::array(const static_data_t&, int64_t)> s, int k_, order_type o)
        : sample(std::move(s))
        , column_name(std::move(t))
        , function_name(std::move(f))
        , k(k_)
        , order_type_(o)
    {}

    static top_k_binary_function_search_info from_top_k_search_info(const top_k_search_info& i);

    inline bool has_value() const { return !column_name.empty(); }

    std::function<nd::array(const static_data_t&, int64_t)> sample;
    std::string column_name;
    std::string function_name;
    int k = 0;
    order_type order_type_ = order_type::ascending;

    // Search configuration for this query
    search_config config = search_config::default_config();
};

} // namespace query_core




