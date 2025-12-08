#pragma once

/**
 * @file exceptions.hpp
 * @brief Definitions of the exceptions for `nd` module.
 */

#include <base/exception.hpp>
#include <base/format.hpp>

#include <string>

namespace nd {

class exception : public base::exception
{
public:
    exception(std::string&& what, params_t&& params)
        : base::exception(std::move(what), std::move(params))
    {
    }

    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class invalid_type : public exception
{
public:
    explicit invalid_type(std::string&& message)
        : exception(std::move(message))
    {
    }
};

class unknown_dtype : public invalid_type
{
public:
    unknown_dtype()
        : invalid_type(std::string("Dtype is unknown."))
    {
    }

    unknown_dtype(const std::string& dtype)
        : invalid_type(std::string("Unknown dtype: " + dtype))
    {
    }
};

class non_numeric_dtype : public exception
{
public:
    explicit non_numeric_dtype(std::string_view t)
        : exception(fmt::format("Dtype {} is not numeric.", t))
    {
    }
};

class non_comparable_dtype : public exception
{
public:
    explicit non_comparable_dtype(std::string_view t)
        : exception(fmt::format("Dtype {} is not comparable.", t))
    {
    }
};

class unsupported_function : public exception
{
public:
    explicit unsupported_function(const std::string& function_name)
        : exception(std::string("The function '") + function_name + "' is not supported for the given arrays.")
    {
    }
};

class unsupported_operator : public exception
{
public:
    unsupported_operator()
        : exception(std::string("The operator is not supported for the given arrays."))
    {
    }
};

class invalid_operation : public exception
{
public:
    explicit invalid_operation(const std::string& what)
        : exception(std::string("Invalid Operation: ") + what)
    {
    }
};

class invalid_dynamic_eval : public exception
{
public:
    invalid_dynamic_eval()
        : exception("Can't eval dynamic shape array.")
    {
    }
};

class invalid_dynamic_eval_shapes_missmatch : public invalid_dynamic_eval
{
public:
    invalid_dynamic_eval_shapes_missmatch()
        : invalid_dynamic_eval()
    {
    }
};

class index_out_of_bounds : public exception
{
public:
    index_out_of_bounds(int index, int size)
        : exception(size == 0 ? (fmt::format("Cannot subscript an empty array. Tried to subscript with index: {}", index))
                              : (fmt::format("Index {} is out of bounds [0-{})", index, size)))
    {
    }
};

class cannot_cast_dtype : public exception
{
public:
    cannot_cast_dtype(std::string_view from, std::string_view to)
        : exception(fmt::format("Cannot cast from type {} to type {}", from, to))
    {
    }
};

class shape_dimensions_do_not_match : public exception
{
public:
    shape_dimensions_do_not_match(uint64_t s1, uint64_t s2)
        : exception(fmt::format("Shape dimensions do not match {} != {} ", s1, s2))
    {
    }
};

class non_array_json : public exception
{
public:
    explicit non_array_json(const std::string& json)
        : exception(fmt::format("The given json '{}' is not an array.", json))
    {
    }
};

class invalid_type_dimensions : public exception
{
public:
    explicit invalid_type_dimensions(uint32_t exp, uint32_t got)
        : exception(fmt::format("Data must have {} dimensions provided {}", exp, got))
    {
    }

    explicit invalid_type_dimensions(const std::string& type, uint32_t exp, uint32_t got)
        : exception(fmt::format("Data of type {} must have '{}' dimensions provided '{}'", type, exp, got))
    {
    }
};


} // namespace nd
