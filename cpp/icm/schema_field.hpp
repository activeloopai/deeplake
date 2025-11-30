#pragma once

#include <memory>
#include <string>

namespace icm {

template<typename type_t>
class schema_field
{
public:
    schema_field(const std::string& name, const type_t& t)
        : name_(name)
        , type_(t)
    {
    }

    const std::string& name() const
    {
        return name_;
    }

    const type_t& type() const
    {
        return type_;
    }

    [[nodiscard]] std::string to_string() const;

private:
    std::string name_;
    type_t type_;
};

} // namespace icm
