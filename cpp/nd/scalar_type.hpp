#pragma once

#include "dtype.hpp"
#include "schema.hpp"

#include <memory>
#include <optional>
#include <variant>

namespace icm {
class const_json;
class json;
} // namespace icm

namespace nd {

class array;

class scalar_type
{
public:
    scalar_type()
        : scalar_type(dtype::unknown)
    {
    }

    scalar_type(dtype dtype)
        : type_(dtype)
    {
    }

    scalar_type(const schema& schema);

    /**
     * @brief Construct a type from JSON.
     * @param j The JSON object.
     */
    static scalar_type from_json(const icm::const_json& j);

    /**
     * @brief Serialize the type to JSON.
     */
    [[nodiscard]] icm::json to_json() const;

    [[nodiscard]] std::string to_string() const;

    dtype get_dtype() const noexcept
    {
        if (has_schema()) {
            return dtype::object;
        }
        return std::get<dtype>(type_);
    }

    const schema& get_schema() const
    {
        ASSERT(has_schema());
        return *std::get<std::shared_ptr<schema>>(type_);
    }

    bool has_schema() const noexcept
    {
        return std::holds_alternative<std::shared_ptr<schema>>(type_);
    }

    class array default_value() const;

    bool operator==(const scalar_type& other) const;

    bool operator!=(const scalar_type& other) const
    {
        return !(*this == other);
    }

private:
    std::variant<dtype, std::shared_ptr<schema>> type_;
};

} // namespace nd
