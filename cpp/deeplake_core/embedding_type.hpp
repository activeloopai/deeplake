#pragma once

#include "embedding_index_type.hpp"
#include "format_definition.hpp"

#include <base/htype.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

namespace async {
template <typename T>
class promise;
} // namespace async

namespace deeplake_core {

class convert_context;

class embedding_type
{
public:
    embedding_type(int32_t size,
                   nd::dtype dtype,
                   embedding_index_type index_type = embedding_index_type::type::clustered)
        : type_(nd::type::array(nd::scalar_type(dtype), icm::shape{size}))
        , size_(size)
        , index_type_(index_type)
    {
    }

    static embedding_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    base::htype htype() const noexcept
    {
        return base::htype::embedding;
    }

    inline bool is_link() const noexcept
    {
        return false;
    }

    nd::type data_type() const;

    format_definition default_format() const;

    async::promise<nd::array> convert_array_to_write(nd::array array) const;

    async::promise<nd::array> convert_batch_to_write(nd::array array) const;

    async::promise<nd::array> convert_array_to_read(nd::array array, const convert_context*) const;

    async::promise<nd::array> convert_batch_to_read(nd::array array, const convert_context*) const;

    std::string to_string() const;

    bool operator==(const embedding_type&) const = default;

    void resize(int32_t size)
    {
        size_ = size;
        type_ = nd::type::array(type_.get_dtype(), icm::shape{size});
    }

public:
    int32_t size() const
    {
        return size_;
    }

    const nd::type& type() const
    {
        return type_;
    }

    embedding_index_type index_type() const
    {
        return index_type_;
    }

private:
    nd::type type_;
    int32_t size_;
    embedding_index_type index_type_;
};

} // namespace deeplake_core
