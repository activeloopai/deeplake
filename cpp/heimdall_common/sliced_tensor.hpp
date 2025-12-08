#pragma once

#include "chained_column_view.hpp"
#include "exceptions.hpp"

#include <async/promise.hpp>
#include <heimdall/column_view.hpp>
#include <icm/indexable.hpp>

#include <memory>

namespace heimdall_common {

class sliced_tensor: public chained_column_view
{
public:
    sliced_tensor(heimdall::column_view_ptr s, const icm::indexable_vector& slice)
        : chained_column_view()
        , source_(s)
        , slice_(slice)
    {
        calculate_shapes();
    }

    sliced_tensor(const sliced_tensor&) = default;
    sliced_tensor& operator=(const sliced_tensor&) = default;
    sliced_tensor(sliced_tensor&&) = default;
    sliced_tensor& operator=(sliced_tensor&&) = default;
    ~sliced_tensor() override = default;

public:
    inline icm::shape max_shape() const noexcept override
    {
        return max_shape_;
    }

    inline icm::shape min_shape() const noexcept override
    {
        return min_shape_;
    }

    inline int64_t samples_count() const noexcept override
    {
        return source_->samples_count();
    }

public:
    inline bool is_sequence() const noexcept override
    {
        return source_->is_sequence();
    }

    inline int64_t sequence_length(int64_t index) const override
    {
        return source_->sequence_length(index);
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
    bool can_fetch_bytes() const noexcept override
    {
        return false;
    }

    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override
    {
        throw invalid_operation("Can't fetch bytes of the sliced tensor.");
        return async::promise<nd::array>();
    }

protected:
    async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, const icm::indexable_vector& source_shape,
                                              std::span<int64_t> result_shape, storage::fetch_options options) override;

    async::promise<nd::array> request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;
    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override
    {
        throw invalid_operation("Can't fetch bytes of the sliced tensor.");
        return async::promise<nd::array>();
    }

    async::promise<nd::array> request_range_bytes_(int64_t start_index, int64_t end_index,
                                                  storage::fetch_options options) override
    {
        throw invalid_operation("Can't fetch bytes of the sliced tensor.");
        return async::promise<nd::array>();
    }

private:
    void calculate_shapes();

private:
    heimdall::column_view_ptr source_;
    icm::indexable_vector slice_;
    icm::shape max_shape_;
    icm::shape min_shape_;
};

/**
 * @brief Creates a sliced tensor from the source tensor.
 * @param source The source tensor.
 * @param slice The slice of the samples.
 * @return column_view_ptr The sliced tensor.
 */
heimdall::column_view_ptr create_sliced_tensor(heimdall::column_view& source, const icm::indexable_vector& slice);

}
