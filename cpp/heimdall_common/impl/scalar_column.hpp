#pragma once

#include "../exceptions.hpp"

#include <heimdall/column_view.hpp>
#include <icm/json.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>
#include <nd/dtype.hpp>

#include <vector>

namespace heimdall_common {

namespace impl {

const icm::json& empty_info();

template <typename T>
class scalar_column : public heimdall::column_view
{
public:
    scalar_column(std::string&& name, std::vector<T>&& data)
        : name_(std::move(name))
        , data_(std::move(data))
    {
    }

public:
    inline const std::string& name() const noexcept override
    {
        return name_;
    }

    void set_name(const std::string& name) noexcept
    {
        name_ = name;
    }

    inline const deeplake_core::type type() const noexcept override
    {
        return deeplake_core::type::generic(nd::type::scalar(nd::dtype_enum_v<T>));
    }

    inline codecs::compression compression() const noexcept override
    {
        return codecs::compression::null;
    }

    inline icm::shape min_shape() const noexcept override
    {
        return icm::shape(1);
    }

    inline icm::shape max_shape() const noexcept override
    {
        return icm::shape(1);
    }

    inline int64_t samples_count() const noexcept override
    {
        return data_.span<const T>().size();
    }

    inline const icm::json& metadata() const noexcept override
    {
        static const icm::json empty;
        return empty;
    }

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        throw heimdall_common::exception("Links info is not available for scalar column.");
    }

public:
    inline bool is_sequence() const noexcept override
    {
        return false;
    }

    inline int64_t sequence_length(int64_t index) const override
    {
        throw heimdall_common::exception("sequence_length is not applicable to non-sequence column.");
        return 0;
    }

public:
    inline async::promise<nd::array> request_full(storage::fetch_options options) override
    {
        return request_range(0, samples_count(), options);
    }

    inline async::promise<nd::array> request_shapes_full(storage::fetch_options options) override
    {
        return request_full(options);
    }

public:
    bool can_fetch_bytes() const noexcept override
    {
        return true;
    }

    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override
    {
        return request_range_bytes(0, samples_count(), options);
    }

public:
    inline bool is_chunked() const noexcept override
    {
        return true;
    }

    int64_t chunk_size_hint() const override
    {
        return static_cast<int64_t>(data_.size());
    }

    inline std::vector<int64_t> chunk_ranges() const override
    {
        return std::vector<int64_t>{static_cast<int64_t>(data_.size())};
    }

protected:
    inline async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override
    {
        ASSERT(index >= 0);
        ASSERT(index < samples_count());
        return async::fulfilled(nd::adapt(data_.span<const T>()[index]));
    }

    inline async::promise<nd::array>
    request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    inline async::promise<nd::array>
    request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    inline async::promise<nd::array> request_sample_(int64_t index,
                                                     const icm::indexable_vector& source_shape,
                                                     std::span<int64_t> result_shape,
                                                     storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    inline async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override
    {
        return request_sample_(index, options);
    }

    inline async::promise<nd::array>
    request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        ASSERT(start_index >= 0);
        ASSERT(end_index >= 0);
        ASSERT(start_index <= samples_count());
        ASSERT(end_index <= samples_count());
        return async::fulfilled(nd::adapt(data_.chunk<T>(start_index, end_index), nd::dtype_enum_v<T>));
    }

    inline async::promise<nd::array>
    request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        return request_range(start_index, end_index, options);
    }

    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override
    {
        return async::fulfilled(nd::adapt(data_.chunk<const T>(index, index + 1), nd::dtype::byte));
    }

    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        return async::fulfilled(nd::adapt(data_.chunk<const T>(start_index, end_index), nd::dtype::byte));
    }

private:
    std::string name_;
    base::memory_buffer data_;
};

} // namespace impl

} // namespace heimdall_common
