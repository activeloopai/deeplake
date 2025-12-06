#pragma once

#include <base/concat.hpp>
#include <base/exception.hpp>
#include <base/format.hpp>

namespace tql {

class exception : public base::exception
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class source_not_found : public exception
{
public:
    explicit source_not_found(const std::string& url, const std::string& error)
        : exception(fmt::format(
              "The query source - '{}', is not found or not supported. Errors from all sources:\n{}", url, error))
    {
    }

    source_not_found()
        : exception("The query source is not found or not supported.")
    {
    }
};

class function_not_found : public exception
{
public:
    explicit function_not_found(const std::string& name)
        : exception(std::string("Function '") + name + "' is not found.")
    {
    }
};

class external_function_not_found : public exception
{
public:
    explicit external_function_not_found(const std::string& name)
        : exception(std::string("Function '") + name + "' is not found in external function sources.")
    {
    }
};

class functor_type_mismatch : public exception
{
public:
    explicit functor_type_mismatch()
        : exception("Functor type is different from what was requested.")
    {
    }
};

class syntax_error : public exception
{
public:
    explicit syntax_error(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class parser_error : public exception
{
public:
    explicit parser_error(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class shape_mismatch : public exception
{
public:
    explicit shape_mismatch(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class invalid_group_function : public exception
{
public:
    invalid_group_function()
        : exception(std::string("invalid_group_function"))
    {
    }
};

class tensor_does_not_exist : public exception
{
public:
    explicit tensor_does_not_exist(const std::string& name)
        : exception(std::string("Tensor \"") + name + "\" does not exist in the dataset. If \"" + name +
                    "\" is referring to a string value, not a tensor, please wrap it in single quotes.")
    {
    }
};

class ambiguous_tensor_name : public exception
{
public:
    ambiguous_tensor_name()
        : exception(std::string("Can't automatically detect output tensor name. Please use AS 'name' at the end of "
                                "expression, to name the output tensor."))
    {
    }
};

class array_indexing_dimension_error : public exception
{
public:
    array_indexing_dimension_error(size_t actual, size_t requested)
        : exception(fmt::format("Can't slice array more than actual dimensions. {} < {}", actual, requested))
    {
    }
};

class json_key_is_not_found : public exception
{
public:
    explicit json_key_is_not_found(const std::string& key, const std::string& object)
        : exception(fmt::format("Key {} not found in object {}", key, object))
    {
    }
};

class json_index_is_not_found : public exception
{
public:
    explicit json_index_is_not_found(int64_t idx, const std::string& object)
        : exception(fmt::format("Index {} not found in object {}", idx, object))
    {
    }
};

} // namespace tql
