#pragma once

#include "exceptions.hpp"
#include "variants.hpp"

#include <base/assert.hpp>
#include <query_core/functor.hpp>

#include <cstdint>
#include <string>
#include <variant>

namespace query_core {

enum class ungroup_type {
    none,
    split_by_axis,
    split_by_number,
    split_by_lengths
};

inline std::string to_string(ungroup_type type)
{
    switch (type) {
    case ungroup_type::none:
        return "none";
    case ungroup_type::split_by_axis:
        return "split_by_axis";
    case ungroup_type::split_by_number:
        return "split_by_number";
    case ungroup_type::split_by_lengths:
        return "split_by_lengths";
    default:
        break;
    }
    ASSERT(false);
    return "unknown";
}

class ungroup_statement
{
public:
    inline ungroup_statement()
        : data_(std::in_place_index<0>)
    {}

    inline explicit ungroup_statement(bool)
        : data_(std::in_place_index<1>)
    {}

    inline explicit ungroup_statement(int64_t v)
        : data_(std::in_place_index<2>, v)
    {}

    inline explicit ungroup_statement(std::vector<int64_t> v)
        : data_(std::in_place_index<3>, v)
    {}

public:
    inline explicit operator bool() const
    {
        return data_.index() != 0;
    }

    inline ungroup_type type() const
    {
        switch (data_.index()) {
        case 0:
            return ungroup_type::none;
        case 1:
            return ungroup_type::split_by_axis;
        case 2:
            return ungroup_type::split_by_number;
        case 3:
            return ungroup_type::split_by_lengths;
        default:
            break;
        }
        ASSERT(false);
        return ungroup_type::none;
    }

    inline int64_t number() const
    {
        return std::get<2>(data_);
    }

    inline const std::vector<int64_t>& lengths() const
    {
        return std::get<3>(data_);
    }

    inline std::string to_string() const
    {
        return query_core::to_string(type());
    }

private:
    std::variant<std::monostate, std::monostate, int64_t, std::vector<int64_t>> data_;
};

}
