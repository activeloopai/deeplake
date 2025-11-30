#pragma once

#include <base/exception.hpp>
#include <base/format.hpp>

namespace deeplake_api {

class exception : public base::exception
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class unknown_string_type : public exception
{
public:
    explicit unknown_string_type(std::string_view type)
        : exception(fmt::format("The string '{}' is not convertible to data type.", type))
    {
    }
};

class unknown_text_indexing_type : public exception
{
public:
    explicit unknown_text_indexing_type(std::string_view type)
        : exception(fmt::format("The text indexing type '{}' is not valid.", type))
    {
    }
};

class invalid_type_and_format_pair : public exception
{
public:
    explicit invalid_type_and_format_pair()
        : exception("format must be empty when htype is specified.")
    {
    }
};

class read_parquet_error : public exception
{
public:
    explicit read_parquet_error(std::string&& what)
        : exception(std::move(what))
    {
    }
};

class invalid_link_type : public exception
{
public:
    explicit invalid_link_type(const std::string& kind)
        : exception(fmt::format("Link to type of '{}' is not supported yet.", kind))
    {
    }
};

class cannot_tag_view : public exception
{
public:
    explicit cannot_tag_view(const std::string& error_message)
        : exception(fmt::format("Cannot tag the dataset view. {}", error_message))
    {
    }
};

class cannot_open_uncommitted_version : public exception
{
public:
    cannot_open_uncommitted_version()
        : exception("Cannot open an uncommitted version. Please use the original dataset instead.")
    {
    }
};

class empty_column_name_exception : public exception
{
public:
    empty_column_name_exception()
        : exception("Column name cannot be empty.")
    {
    }
};

} // namespace deeplake_api
