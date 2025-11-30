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

class datasets_mismatch_uneven_tensors : public exception
{
public:
    explicit datasets_mismatch_uneven_tensors(int t1, int t2)
        : exception(std::string("Datasets can't be merged, because they have different number of tensors - ") +
                    std::to_string(t1) + " compared to " + std::to_string(t2) + ".")
    {}
};

class datasets_mismatch_missing_tensor : public exception
{
public:
    explicit datasets_mismatch_missing_tensor(std::string tensor)
        : exception(std::string("Datasets can't be merged, because tensor '") + std::move(tensor) +
                    "' exists in one of them and missing on another.")
    {}
};

class tensor_mismatch : public exception
{
public:
    explicit tensor_mismatch(std::string tensor, nd::dtype f, nd::dtype s)
        : exception(fmt::format(
              "Tensors with name '{}' can't be merged, because the dtypes are incompatible - {} {}.", tensor,
              nd::dtype_to_str(f), nd::dtype_to_str(s)))
    {}

    explicit tensor_mismatch(std::string tensor, base::htype f, base::htype s)
        : exception(fmt::format(
              "Tensors with name '{}' can't be merged, because they have different htypes - {} compared to {}.", tensor,
              base::htype_to_str(f), base::htype_to_str(s)))
    {}
};

class tensor_mismatch_sequence_mismatch : public exception
{
public:
    explicit tensor_mismatch_sequence_mismatch(std::string tensor)
        : exception(std::string("Tensors with name '") + std::move(tensor) +
                    "' can't be merged, because one is sequence and another is not.")
    {}
};

} // namespace heimdall_common
