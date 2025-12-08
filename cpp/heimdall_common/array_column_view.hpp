#pragma once

#include "exceptions.hpp"

#include <heimdall/column_view.hpp>
#include <icm/json.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>
#include <nd/dtype.hpp>

#include <memory>

namespace heimdall_common {

class array_column : public heimdall::column_view
{
    std::string name_;
    deeplake_core::type type_;
    nd::array data_;

public:
    array_column(std::string name, deeplake_core::type type, nd::array&& data)
        : name_(std::move(name))
        , type_(std::move(type))
        , data_(std::move(data))
    {
    }

public:
    const std::string& name() const noexcept override
    {
        return name_;
    }

    const deeplake_core::type type() const noexcept override
    {
        return type_;
    }

    codecs::compression compression() const noexcept override
    {
        return codecs::compression::null;
    }

    icm::shape min_shape() const noexcept override
    {
        return {};
    }

    icm::shape max_shape() const noexcept override
    {
        return {};
    }

    int64_t samples_count() const noexcept override
    {
        return (data_.dimensions() == 0 ? 1 : data_.size());
    }

    const icm::json& metadata() const noexcept override
    {
        static const icm::json empty = icm::json();
        return empty;
    }

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        throw heimdall_common::exception("Links info is not available for array column.");
    }

    bool is_sequence() const noexcept override
    {
        return false;
    }

    int64_t sequence_length(int64_t index) const override
    {
        throw heimdall_common::exception("sequence_length is not applicable to non-sequence column.");
        return 0;
    }

    async::promise<nd::array> request_full(storage::fetch_options options) override
    {
        return async::fulfilled(data_);
    }

    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override
    {
        return request_full(options);
    }

    bool can_fetch_bytes() const noexcept override
    {
        return false;
    }

    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override
    {
        return request_range_bytes(0, samples_count(), options);
    }

public:
    bool is_chunked() const noexcept override
    {
        return true;
    }

    int64_t chunk_size_hint() const override
    {
        return static_cast<int64_t>(data_.size());
    }

    std::vector<int64_t> chunk_ranges() const override
    {
        return std::vector<int64_t>{static_cast<int64_t>(data_.size())};
    }

protected:
    async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override
    {
        ASSERT(index >= 0);
        ASSERT(index < samples_count());
        return async::fulfilled(samples_count() == 1 ? data_ : data_[index]);
    }

    async::promise<nd::array>
    request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    async::promise<nd::array>
    request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    async::promise<nd::array> request_sample_(int64_t index,
                                              const icm::indexable_vector& source_shape,
                                              std::span<int64_t> result_shape,
                                              storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    async::promise<nd::array>
    request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        ASSERT(start_index >= 0);
        ASSERT(end_index >= 0);
        ASSERT(start_index <= samples_count());
        ASSERT(end_index <= samples_count());
        return async::fulfilled(nd::stride(data_, icm::slice_t<int64_t>::range(start_index, end_index)));
    }

    inline async::promise<nd::array>
    request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        return request_range(start_index, end_index, options);
    }

    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override
    {
        throw heimdall_common::exception("request_bytes_ is not supported.");
    }

    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        throw heimdall_common::exception("request_range_bytes_ is not supported.");
    }
};

} // namespace heimdall_common
