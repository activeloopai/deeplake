#pragma once

/**
 * @file schema.hpp
 * @brief Declaration of the schema interface.
 */

#include "schema_field.hpp"

#include "fifo_map.hpp"
#include "const_json.hpp"
#include "json.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace icm {

template<typename type_t>
class schema
{
public:
    schema() = default;

    schema(icm::fifo_map<std::string, type_t> schema)
        : schema_(std::move(schema))
    {
    }

    schema(std::map<std::string, type_t> schema)
        : schema_(schema.begin(), schema.end())
    {
    }

    schema(std::unordered_map<std::string, type_t> schema)
        : schema_(schema.begin(), schema.end())
    {
    }

    static schema from_json(const icm::const_json& j);

    [[nodiscard]] icm::json to_json() const;

    schema_field<type_t> get_field(const std::string& key) const
    {
        return schema_field(key, schema_.at(key));
    }

    const type_t& operator[](const std::string& key) const
    {
        return schema_.at(key);
    }

    const type_t& at(const std::string& key) const
    {
        return schema_.at(key);
    }

    std::vector<schema_field<type_t>> fields() const
    {
        std::vector<schema_field<type_t>> return_fields;
        for (const auto& [k, v] : schema_) {
            return_fields.emplace_back(k, v);
        }
        return return_fields;
    }

    auto begin() const
    {
        return schema_.begin();
    }

    auto end() const
    {
        return schema_.end();
    }

    auto rbegin() const
    {
        return schema_.rbegin();
    }

    auto rend() const
    {
        return schema_.rend();
    }

    auto size() const
    {
        return schema_.size();
    }

    bool empty() const
    {
        return schema_.empty();
    }

    void clear()
    {
        schema_.clear();
    }

    void insert(std::pair<std::string, type_t> pair)
    {
        schema_.insert(std::move(pair));
    }

    template <typename ... Args>
    void emplace(Args&& ... args)
    {
        schema_.emplace(std::forward<Args>(args)...);
    }

    std::string to_string() const;

    bool operator==(const schema& other) const;

private:
    icm::fifo_map<std::string, type_t> schema_;
};

} // namespace icm