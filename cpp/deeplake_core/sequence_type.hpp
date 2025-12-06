#pragma once

#include "format_definition.hpp"

#include <async/promise.hpp>
#include <base/htype.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <memory>

namespace deeplake_core {

class type;
class convert_context;

class sequence_type
{
public:
    sequence_type(type&& type);
    sequence_type(const type& type);

    static sequence_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    base::htype htype() const noexcept;

    bool is_link() const noexcept;

    nd::type data_type() const;

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context* ctx) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context* ctx) const;

    std::string to_string() const;

    bool operator==(const sequence_type& other) const;

private:
    std::shared_ptr<type> type_;
};

} // namespace deeplake_core
