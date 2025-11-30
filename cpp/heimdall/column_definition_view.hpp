#pragma once

#include <deeplake_core/type.hpp>

#include <optional>
#include <string>
#include <utility>

namespace heimdall {

using column_id_t = uint32_t;

class column_definition_view
{

public:
    explicit column_definition_view(column_id_t id,
                                    std::string name,
                                    deeplake_core::type type,
                                    const std::optional<nd::array>& default_value = std::nullopt)
        : id_(id)
        , name_(std::move(name))
        , type_(std::move(type))
        , default_value_(default_value)
    {
    }

    [[nodiscard]] column_id_t id() const
    {
        return id_;
    }

    [[nodiscard]] const std::string& name() const
    {
        return name_;
    }

    [[nodiscard]] const deeplake_core::type& core_type() const
    {
        return type_;
    }

    [[nodiscard]] deeplake_core::type& type()
    {
        return type_;
    }

    [[nodiscard]] const std::optional<nd::array>& default_value() const
    {
        return default_value_;
    }

    [[nodiscard]] std::string to_string() const;

private:
    column_id_t id_;
    std::string name_;
    deeplake_core::type type_;
    std::optional<nd::array> default_value_;
};

} // namespace heimdall
