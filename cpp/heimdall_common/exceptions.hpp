#pragma once

#include <base/exception.hpp>
#include <base/format.hpp>
#include <base/htype.hpp>
#include <nd/dtype.hpp>

namespace heimdall_common {

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

class datasets_mismatch_uneven_columns : public exception
{
public:
    explicit datasets_mismatch_uneven_columns(int t1, int t2)
        : exception(std::string("Datasets can't be merged, because they have different number of columns - ") +
                    std::to_string(t1) + " compared to " + std::to_string(t2) + ".")
    {}
};

class datasets_mismatch_missing_column : public exception
{
public:
    explicit datasets_mismatch_missing_column(std::string column)
        : exception(std::string("Datasets can't be merged, because column '") + std::move(column) +
                    "' exists in one of them and missing on another.")
    {}
};

class column_mismatch : public exception
{
public:
    explicit column_mismatch(std::string column, nd::dtype f, nd::dtype s)
        : exception(fmt::format(
              "Columns with name '{}' can't be merged, because the dtypes are incompatible - {} {}.", column,
              nd::dtype_to_str(f), nd::dtype_to_str(s)))
    {}

    explicit column_mismatch(std::string column, base::htype f, base::htype s)
        : exception(fmt::format(
              "Columns with name '{}' can't be merged, because they have different htypes - {} compared to {}.", column,
              base::htype_to_str(f), base::htype_to_str(s)))
    {}
};

class column_mismatch_sequence_mismatch : public exception
{
public:
    explicit column_mismatch_sequence_mismatch(std::string column)
        : exception(std::string("Column with name '") + std::move(column) +
                    "' can't be merged, because one is sequence and another is not.")
    {}
};

} // namespace heimdall_common
