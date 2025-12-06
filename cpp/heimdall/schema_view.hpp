#pragma once

#include "column_definition_view.hpp"
#include "column_id_t.hpp"

#include <nd/schema.hpp>

#include <sstream>

namespace heimdall {

class schema_view
{
public:
    explicit schema_view() = default;

    explicit schema_view(std::vector<column_definition_view> columns)
        : columns_(std::move(columns))
    {
    }

    [[nodiscard]] const std::vector<column_definition_view>& columns_view() const
    {
        return columns_;
    }

    [[nodiscard]] int size() const
    {
        return static_cast<int>(columns_.size());
    }

    column_definition_view get_column_view(const std::string& name) const;

    column_definition_view get_column_view(column_id_t column_id) const;

    [[nodiscard]] std::string to_string() const
    {
        if (size() == 0) {
            return "Empty schema";
        }
        std::ostringstream return_string;
        for (const auto& column : columns_view()) {
            return_string << column.to_string() << std::endl;
        }

        return std::move(return_string).str();
    }

    [[nodiscard]] nd::schema core_schema() const
    {
        icm::fifo_map<std::string, nd::type> schema_map;

        for (const auto& column : columns_) {
            schema_map.emplace(column.name(), column.core_type().data_type());
        }

        return {schema_map};
    }

protected:
    std::vector<column_definition_view> columns_;
};

} // namespace heimdall
