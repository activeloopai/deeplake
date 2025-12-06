#pragma once

/**
 * @file text_search_info.hpp 
 * @brief Definition of `text_search_info` struct.
 */

#include <nd/array.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace nd {
enum class binary_operator_type : uint8_t;
}

namespace query_core {

class expr;

/**
 * @struct text_search_info
 * @brief Contains information which tql needs to run text search query. The info can be obtained from the parser.
 */
struct text_search_info
{
    enum class search_type
    {
        contains,
        equals,
    };

    static search_type optype_to_search_type(nd::binary_operator_type btype);

    text_search_info();

    text_search_info(std::string t);

    static text_search_info generate(const expr& e, nd::array batch_params);

    inline bool has_value() const
    {
        return !column_name.empty() && !search_values.empty() && !search_values[0].empty();
    }

    std::string column_name;
    search_type type;
    std::vector<std::vector<std::string>> search_values;
};

} // namespace deeplake_core
