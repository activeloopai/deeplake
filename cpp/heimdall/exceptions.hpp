#pragma once

#include "column_id_t.hpp"

#include <base/exception.hpp>
#include <base/format.hpp>
#include <base/htype.hpp>
#include <nd/dtype.hpp>

namespace heimdall {

class exception : public base::exception
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {}
};

class invalid_operation : public exception
{
public:
    explicit invalid_operation(std::string&& what)
        : exception(std::string("Invalid operation - ") + std::move(what))
    {}
};

class column_does_not_exist : public exception
{
public:
    explicit column_does_not_exist(std::string_view name)
        : exception(fmt::format("Column '{}' does not exist", name))
    {
    }

    explicit column_does_not_exist(const column_id_t field_id)
        : exception(fmt::format("Column '{}' does not exist", field_id))
    {
    }
};

class row_out_of_range : public exception
{
public:
    explicit row_out_of_range(int64_t row, int64_t max_row)
        : exception(fmt::format("Row {} is out of range [0, {})", row, max_row))
    {}
};

class column_out_of_range : public exception
{
public:
    explicit column_out_of_range(int64_t column, int64_t max_row)
        : exception(fmt::format("Column {} is out of range [0, {})", column, max_row))
    {}
};

} // namespace heimdall
