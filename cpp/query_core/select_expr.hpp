#pragma once

#include "expr.hpp"

#include <string>

namespace query_core {

class select_expr
{
public:
    select_expr(expr e, std::string alias = {}) : e(std::move(e)), alias(std::move(alias))
    {
    }

    const expr& get_expr() const
    {
        return e;
    }

    const std::string& get_alias() const
    {
        return alias;
    }

    std::string to_string() const
    {
        auto res = e.to_string();
        if (!alias.empty()) {
            res += " AS " + alias;
        }
        return res;
    }

private:
    expr e;
    std::string alias;
};

} // namespace query_core
