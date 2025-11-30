#pragma once

#include <nd/array.hpp>

#include <string>
#include <vector>


namespace query_core {

class expr;
class text_search_info;

enum class relational_operator {
    equals,
    between,
    less,
    less_eq,
    greater,
    greater_eq,
    invalid
};

struct inverted_index_search_info {
    inverted_index_search_info() = default;

    inline bool has_value() const
    {
        return !column_name.empty();
    }

    static inverted_index_search_info generate(const expr& e, nd::array batch_params);

    text_search_info to_text_search_info() const;

    std::string column_name;
    std::string dict_path; // For JSON path support
    relational_operator op = relational_operator::invalid;
    std::vector<nd::array> search_values;
};

} // namespace query_core
