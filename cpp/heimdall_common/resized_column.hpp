#pragma once

#include "chained_column_view.hpp"

namespace nd {

class array;

}

namespace heimdall_common {

class resized_column: public chained_column_view
{
public:
    resized_column(heimdall::column_view_ptr s, int64_t size)
        : chained_column_view()
        , source_(s)
        , size_(size)
    {
        ASSERT(s != nullptr);
    }

public:
    inline icm::shape max_shape() const noexcept override
    {
        return source_->max_shape();
    }

    inline icm::shape min_shape() const noexcept override
    {
        return source_->min_shape();
    }

    inline int64_t samples_count() const noexcept override
    {
        return size_;
    }

public:
    inline bool is_sequence() const noexcept override
    {
        return source_->is_sequence();
    }

    inline int64_t sequence_length(int64_t index) const override
    {
        if (index < source_->samples_count()) {
            return source_->sequence_length(index);
        }
        return 0;
    }

public:
    inline heimdall::column_view_ptr source() override
    {
        return source_;
    }

    inline heimdall::const_column_view_ptr source() const override
    {
        return source_;
    }

public:
    async::promise<nd::array> request_full(storage::fetch_options options) override;
    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override;

public:
    bool can_fetch_bytes() const noexcept override;
    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override;

protected:
    async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, const icm::indexable_vector& source_shape,
                                              std::span<int64_t> result_shape, storage::fetch_options options) override;

    async::promise<nd::array> request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_range_bytes_(int64_t start_index, int64_t end_index,
                                                   storage::fetch_options options) override;
private:
    heimdall::column_view_ptr source_;
    int64_t size_;
};

heimdall::column_view_ptr create_resized_column(heimdall::column_view& s, int64_t size);

}
