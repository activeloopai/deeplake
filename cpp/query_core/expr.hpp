#pragma once

#include <nd/array.hpp>
#include <nd/io.hpp>
#include <nd/operator_type.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace query_core {

enum class expr_type : uint8_t
{
    literal_float,
    literal_string,
    literal_int,
    literal_bool,
    literal_array,
    star,
    param,
    column_ref,
    function_ref,
    unary_operator,
    binary_operator,
    ternary_operator,
    array,
    array_index,
    array_string_index,
    array_dynamic_index,
    slice,
};

class expr;

template <typename T>
class literal_expr
{
public:
    literal_expr(T value)
        : value(std::move(value))
    {
    }

    T get_value() const
    {
        return value;
    }

    std::string to_string() const
    {
        if constexpr (std::is_same_v<T, std::string>) {
            return "'" + value + "'";
        } else {
            return std::to_string(value);
        }
    }

private:
    T value;
};

template <>
class literal_expr<nd::array>
{
public:
    explicit literal_expr(nd::array value)
        : value(std::move(value))
    {
    }

    const nd::array& get_value() const
    {
        return value;
    }

    std::string to_string() const
    {
        return nd::to_sql_string(value);
    }

private:
    nd::array value;
};

using literal_float_expr = literal_expr<double>;
using literal_string_expr = literal_expr<std::string>;
using literal_int_expr = literal_expr<int64_t>;
using literal_bool_expr = literal_expr<bool>;
using literal_array_expr = literal_expr<nd::array>;

class star_expr
{
public:
    std::string to_string() const
    {
        return "*";
    }
};

class param_expr
{
public:
    explicit param_expr(int32_t index)
        : index_(index)
    {
    }

    inline int32_t get_index() const noexcept
    {
        return index_;
    }

    inline std::string to_string() const noexcept
    {
        return "?";
    }

private:
    int32_t index_ = 0;
};

class column_ref_expr
{
public:
    column_ref_expr(std::string name, std::string table)
        : name(std::move(name))
        , table(std::move(table))
    {
    }

    const std::string& get_name() const
    {
        return name;
    }

    const std::string& get_table() const
    {
        return table;
    }

    std::string to_string() const;

private:
    std::string name;
    std::string table;
};

class function_ref_expr
{
public:
    function_ref_expr(std::string name, std::vector<expr> args);
    function_ref_expr(std::string name, std::vector<std::pair<expr, expr>> arg_pairs);

    const std::string& get_name() const
    {
        return name;
    }

    bool has_arg_pairs() const;

    const std::vector<expr>& get_args() const;

    const std::vector<std::pair<expr, expr>>& get_arg_pairs() const;

    std::string to_string() const;

private:
    using data_t = std::variant<std::vector<expr>, std::vector<std::pair<expr, expr>>>;
    std::string name;
    std::shared_ptr<data_t> args;
};

class unary_operator_expr
{
public:
    unary_operator_expr(expr&& operand, nd::unary_operator_type type);

    const expr& get_operand() const;

    nd::unary_operator_type get_type() const
    {
        return type;
    }

    std::string to_string() const;

private:
    std::shared_ptr<expr> operand;
    nd::unary_operator_type type;
};

class binary_operator_expr
{
public:
    binary_operator_expr(expr&& left, expr&& right, nd::binary_operator_type type);

    const expr& get_left() const;

    const expr& get_right() const;

    nd::binary_operator_type get_type() const
    {
        return type;
    }

    std::string to_string() const;

private:
    std::shared_ptr<std::pair<expr, expr>> exprs;
    nd::binary_operator_type type;
};

class ternary_operator_expr
{
public:
    ternary_operator_expr(expr&& first, expr&& second, expr&& third, nd::ternary_operator_type type);

    const expr& get_first() const;

    const expr& get_second() const;

    const expr& get_third() const;

    nd::ternary_operator_type get_type() const
    {
        return type;
    }

    std::string to_string() const;

private:
    std::shared_ptr<std::array<expr, 3>> exprs;
    nd::ternary_operator_type type;
};

class array_expr
{
public:
    array_expr(std::vector<expr> elements);

    const std::vector<expr>& get_elements() const;

    std::string to_string() const;

private:
    std::shared_ptr<std::vector<expr>> elements;
};

class array_index_expr
{
public:
    array_index_expr(expr&& array, std::vector<expr> index);

    const expr& get_array() const;

    const std::vector<expr>& get_index() const;

    std::string to_string() const;

private:
    std::shared_ptr<expr> array;
    std::shared_ptr<std::vector<expr>> index;
};

class array_dynamic_index_expr
{
public:
    array_dynamic_index_expr(expr&& array, expr&& index);

    const expr& get_array() const;

    const expr& get_index() const;

    std::string to_string() const;

private:
    std::shared_ptr<expr> array;
    std::shared_ptr<expr> index;
};

class array_string_index_expr
{
public:
    array_string_index_expr(expr&& array, std::string index);

    const expr& get_array() const;

    const std::string& get_index() const;

    std::string to_string() const;

private:
    std::shared_ptr<expr> array;
    std::shared_ptr<std::string> index;
};

class slice_expr
{
public:
    slice_expr(expr&& start, expr&& stop, expr&& step);

    const expr& get_start() const;

    const expr& get_stop() const;

    const expr& get_step() const;

    std::string to_string() const;

private:
    std::shared_ptr<std::array<expr, 3>> exprs;
};

class expr
{
public:
    expr()
        : data_(std::monostate())
    {
    }

    explicit operator bool() const
    {
        return !std::holds_alternative<std::monostate>(data_);
    }

    static expr make_literal_float(double value)
    {
        return expr(literal_float_expr(value));
    }

    static expr make_literal_string(std::string value)
    {
        return expr(literal_string_expr(std::move(value)));
    }

    static expr make_literal_int(int64_t value)
    {
        return expr(literal_int_expr(value));
    }

    static expr make_literal_bool(bool value)
    {
        return expr(literal_bool_expr(value));
    }

    static expr make_literal_array(nd::array value)
    {
        return expr(literal_array_expr(std::move(value)));
    }

    static expr make_star()
    {
        return expr(star_expr());
    }

    static expr make_parameter(int32_t index)
    {
        return expr(param_expr(index));
    }

    static expr make_column_ref(std::string name, std::string table)
    {
        return expr(column_ref_expr(std::move(name), std::move(table)));
    }

    static expr make_function_ref(std::string name, std::vector<expr> args)
    {
        return expr(function_ref_expr(std::move(name), std::move(args)));
    }

    static expr make_function_ref(std::string name, std::vector<std::pair<expr, expr>> arg_pairs)
    {
        return expr(function_ref_expr(std::move(name), std::move(arg_pairs)));
    }

    static expr make_unary_operator(expr operand, nd::unary_operator_type type)
    {
        return expr(unary_operator_expr(std::move(operand), type));
    }

    static expr make_binary_operator(expr left, expr right, nd::binary_operator_type type)
    {
        return expr(binary_operator_expr(std::move(left), std::move(right), type));
    }

    static expr make_ternary_operator(expr first, expr second, expr third, nd::ternary_operator_type type)
    {
        return expr(ternary_operator_expr(std::move(first), std::move(second), std::move(third), type));
    }

    static expr make_array(std::vector<expr> elements)
    {
        return expr(array_expr(std::move(elements)));
    }

    static expr make_array_index(expr array, std::vector<expr> index)
    {
        return expr(array_index_expr(std::move(array), std::move(index)));
    }

    static expr make_array_dynamic_index(expr array, expr index)
    {
        return expr(array_dynamic_index_expr(std::move(array), std::move(index)));
    }

    static expr make_array_string_index(expr array, std::string index)
    {
        return expr(array_string_index_expr(std::move(array), std::move(index)));
    }

    static expr make_slice(expr start, expr stop, expr step)
    {
        return expr(slice_expr(std::move(start), std::move(stop), std::move(step)));
    }

    expr_type get_type() const
    {
        return static_cast<expr_type>(data_.index());
    }

    std::string to_string() const;

    const literal_float_expr& as_literal_float() const
    {
        return std::get<literal_float_expr>(data_);
    }

    const literal_string_expr& as_literal_string() const
    {
        return std::get<literal_string_expr>(data_);
    }

    const literal_int_expr& as_literal_int() const
    {
        return std::get<literal_int_expr>(data_);
    }

    const literal_bool_expr& as_literal_bool() const
    {
        return std::get<literal_bool_expr>(data_);
    }

    const literal_array_expr& as_literal_array() const
    {
        return std::get<literal_array_expr>(data_);
    }

    const star_expr& as_star() const
    {
        return std::get<star_expr>(data_);
    }

    const param_expr& as_parameter() const
    {
        return std::get<param_expr>(data_);
    }

    const column_ref_expr& as_column_ref() const
    {
        return std::get<column_ref_expr>(data_);
    }

    const function_ref_expr& as_function_ref() const
    {
        return std::get<function_ref_expr>(data_);
    }

    const unary_operator_expr& as_unary_operator() const
    {
        return std::get<unary_operator_expr>(data_);
    }

    const binary_operator_expr& as_binary_operator() const
    {
        return std::get<binary_operator_expr>(data_);
    }

    const ternary_operator_expr& as_ternary_operator() const
    {
        return std::get<ternary_operator_expr>(data_);
    }

    const array_expr& as_array() const
    {
        return std::get<array_expr>(data_);
    }

    const array_index_expr& as_array_index() const
    {
        return std::get<array_index_expr>(data_);
    }

    const array_dynamic_index_expr& as_array_dynamic_index() const
    {
        return std::get<array_dynamic_index_expr>(data_);
    }

    const array_string_index_expr& as_array_string_index() const
    {
        return std::get<array_string_index_expr>(data_);
    }

    const slice_expr& as_slice() const
    {
        return std::get<slice_expr>(data_);
    }

    template <typename Visitor>
    decltype(auto) visit(Visitor&& visitor) const
    {
        return std::visit(std::forward<Visitor>(visitor), data_);
    }

private:
    using data_t = std::variant<literal_float_expr,
                                literal_string_expr,
                                literal_int_expr,
                                literal_bool_expr,
                                literal_array_expr,
                                star_expr,
                                param_expr,
                                column_ref_expr,
                                function_ref_expr,
                                unary_operator_expr,
                                binary_operator_expr,
                                ternary_operator_expr,
                                array_expr,
                                array_index_expr,
                                array_string_index_expr,
                                array_dynamic_index_expr,
                                slice_expr,
                                std::monostate>;

    expr(data_t data)
        : data_(std::move(data))
    {
    }

    data_t data_;
};

inline function_ref_expr::function_ref_expr(std::string name, std::vector<expr> args)
    : name(std::move(name))
    , args(std::make_shared<data_t>(std::move(args)))
{
    std::transform(this->name.begin(), this->name.end(), this->name.begin(), ::toupper);
}

inline function_ref_expr::function_ref_expr(std::string name, std::vector<std::pair<expr, expr>> arg_pairs)
    : name(std::move(name))
    , args(std::make_shared<data_t>(std::move(arg_pairs)))
{
    std::transform(this->name.begin(), this->name.end(), this->name.begin(), ::toupper);
}

inline bool function_ref_expr::has_arg_pairs() const
{
    return std::holds_alternative<std::vector<std::pair<expr, expr>>>(*args);
}

inline unary_operator_expr::unary_operator_expr(expr&& operand, nd::unary_operator_type type)
    : operand(std::make_shared<expr>(std::move(operand)))
    , type(type)
{
}

inline const expr& unary_operator_expr::get_operand() const
{
    return *operand;
}

inline binary_operator_expr::binary_operator_expr(expr&& left, expr&& right, nd::binary_operator_type type)
    : exprs(std::make_shared<std::pair<expr, expr>>(std::move(left), std::move(right)))
    , type(type)
{
}

inline const expr& binary_operator_expr::get_left() const
{
    return exprs->first;
}

inline const expr& binary_operator_expr::get_right() const
{
    return exprs->second;
}

inline ternary_operator_expr::ternary_operator_expr(expr&& first,
                                                    expr&& second,
                                                    expr&& third,
                                                    nd::ternary_operator_type type)
    : exprs(std::make_shared<std::array<expr, 3>>(
          std::array<expr, 3>{std::move(first), std::move(second), std::move(third)}))
    , type(type)
{
}

inline const expr& ternary_operator_expr::get_first() const
{
    return exprs->at(0);
}

inline const expr& ternary_operator_expr::get_second() const
{
    return exprs->at(1);
}

inline const expr& ternary_operator_expr::get_third() const
{
    return exprs->at(2);
}

inline array_expr::array_expr(std::vector<expr> elements)
    : elements(std::make_shared<std::vector<expr>>(std::move(elements)))
{
}

inline const std::vector<expr>& array_expr::get_elements() const
{
    return *elements;
}

inline array_index_expr::array_index_expr(expr&& array, std::vector<expr> index)
    : array(std::make_shared<expr>(std::move(array)))
    , index(std::make_shared<std::vector<expr>>(std::move(index)))
{
}

inline const expr& array_index_expr::get_array() const
{
    return *array;
}

inline const std::vector<expr>& array_index_expr::get_index() const
{
    return *index;
}

inline array_dynamic_index_expr::array_dynamic_index_expr(expr&& array, expr&& index)
    : array(std::make_shared<expr>(std::move(array)))
    , index(std::make_shared<expr>(std::move(index)))
{
}

inline const expr& array_dynamic_index_expr::get_array() const
{
    return *array;
}

inline const expr& array_dynamic_index_expr::get_index() const
{
    return *index;
}

inline array_string_index_expr::array_string_index_expr(expr&& array, std::string index)
    : array(std::make_shared<expr>(std::move(array)))
    , index(std::make_shared<std::string>(std::move(index)))
{
}

inline const expr& array_string_index_expr::get_array() const
{
    return *array;
}

inline const std::string& array_string_index_expr::get_index() const
{
    return *index;
}

inline slice_expr::slice_expr(expr&& start, expr&& stop, expr&& step)
    : exprs(std::make_shared<std::array<expr, 3>>(
          std::array<expr, 3>{std::move(start), std::move(stop), std::move(step)}))
{
}

inline const expr& slice_expr::get_start() const
{
    return exprs->at(0);
}

inline const expr& slice_expr::get_stop() const
{
    return exprs->at(1);
}

inline const expr& slice_expr::get_step() const
{
    return exprs->at(2);
}

} // namespace query_core
