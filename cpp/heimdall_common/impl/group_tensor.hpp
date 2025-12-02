#pragma once
#include "../chained_column_view.hpp"

namespace heimdall_common::impl {

// abstract class
// child classes must override is_sequence() method.
class group_tensor : public chained_column_view
{
public:
    group_tensor(heimdall::column_view_ptr source, const std::vector<int64_t>& sequence_lengths)
        : chained_column_view(), sequence_lengths_(sequence_lengths), source_(std::move(source))
    {
        calculate_offsets_and_shapes(*source_);
    }

public:
    const auto& offsets() const
    {
        return sequence_offsets_;
    }

    const auto& lengths() const
    {
        return sequence_lengths_;
    }

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
        return static_cast<int64_t>(sequence_lengths_.size());
    }

public:
    inline bool is_sequence() const noexcept override = 0;

    inline int64_t sequence_length(int64_t index) const override
    {
        return sequence_lengths_[index];
    }

public:
    async::promise<nd::array> request_full(storage::fetch_options options) override;
    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override;

public:
    bool can_fetch_bytes() const noexcept override;
    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override;

public:
    inline heimdall::column_view_ptr source() override
    {
        return source_;
    }

    inline heimdall::const_column_view_ptr source() const override
    {
        return source_;
    }

protected:
    async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, const icm::indexable_vector& source_shape,
                                              storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, std::span<int64_t> result_shape,
                                              storage::fetch_options options) override;
    async::promise<nd::array> request_sample_(int64_t index, const icm::indexable_vector& source_shape,
                                              std::span<int64_t> result_shape, storage::fetch_options options) override;

    async::promise<nd::array> request_range_(int64_t start_index, int64_t end_index,
                                             storage::fetch_options options) override;

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_range_shape_(int64_t start_index, int64_t end_index,
                                                   storage::fetch_options options) override;
    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array> request_range_bytes_(int64_t start_index, int64_t end_index,
                                                   storage::fetch_options options) override;

private:
    void calculate_offsets_and_shapes(const heimdall::column_view& s);

private:
    std::vector<int64_t> sequence_lengths_;
    std::vector<int64_t> sequence_offsets_;
    icm::shape max_shape_;
    icm::shape min_shape_;
    heimdall::column_view_ptr source_;
};

} // namespace heimdall_common::impl
