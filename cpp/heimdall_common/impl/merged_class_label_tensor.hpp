#pragma once

#include "../exceptions.hpp"

#include <async/promise.hpp>
#include <heimdall/column_view.hpp>
#include <heimdall/links_info_holder.hpp>
#include <icm/json.hpp>
#include <icm/small_vector.hpp>
#include <icm/vector.hpp>

#include <nd/adapt.hpp>

#include <unordered_map>
#include <unordered_set>

namespace heimdall_common::impl {

struct merged_class_info
{
    icm::json unified_class_names_;
    std::unordered_map<int32_t, int32_t> tensor2_mapping_;
};

class merged_class_label_tensor : public heimdall::column_view
{
public:
    merged_class_label_tensor(heimdall::column_view_ptr t1, heimdall::column_view_ptr t2);

public:
    const std::string& name() const noexcept override
    {
        return tensor1_->name();
    }

    const deeplake_core::type type() const noexcept override
    {
        return tensor1_->type();
    }

    base::htype htype() const noexcept override
    {
        return tensor1_->htype();
    }

    codecs::compression compression() const noexcept override
    {
        return codecs::compression::null;
    }

    icm::shape min_shape() const noexcept override
    {
        return min_shape_;
    }

    icm::shape max_shape() const noexcept override
    {
        return max_shape_;
    }

    int64_t samples_count() const noexcept override
    {
        return tensor1_->samples_count() + tensor2_->samples_count();
    }

    const icm::json& metadata() const noexcept override
    {
        return info_;
    }

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        throw ::heimdall_common::exception("Links info is not available for merged class labels tensors.");
    }

    bool is_sequence() const noexcept override
    {
        return tensor1_->is_sequence();
    }

    int64_t sequence_length(int64_t index) const override
    {
        return index < tensor1_->samples_count() ? tensor1_->sequence_length(index)
                                                 : tensor2_->sequence_length(index - tensor1_->samples_count());
    }

    async::promise<nd::array> request_full(storage::fetch_options options) override
    {
        return request_range(0, samples_count(), options);
    }

    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override
    {
        return request_range_shape(0, samples_count(), options);
    }

    bool can_fetch_bytes() const noexcept override
    {
        return tensor1_->can_fetch_bytes() && tensor2_->can_fetch_bytes();
    }

    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override
    {
        return request_range_bytes(0, samples_count(), options);
    }

    bool is_chunked() const noexcept override
    {
        return tensor1_->is_chunked() && tensor2_->is_chunked();
    }

    int64_t chunk_size_hint() const override
    {
        return std::min(tensor1_->chunk_size_hint(), tensor2_->chunk_size_hint());
    }

    std::vector<int64_t> chunk_ranges() const override
    {
        auto c1 = tensor1_->chunk_ranges();
        auto c2 = tensor2_->chunk_ranges();
        auto offset = tensor1_->samples_count();
        c1.reserve(c1.size() + c2.size());
        for (auto c : c2) {
            c1.push_back(c + offset);
        }
        return c1;
    }

protected:
    async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override;

    async::promise<nd::array>
    request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) override;

    async::promise<nd::array>
    request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) override;

    async::promise<nd::array> request_sample_(int64_t index,
                                              const icm::indexable_vector& source_shape,
                                              std::span<int64_t> result_shape,
                                              storage::fetch_options options) override;

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override
    {
        return index < tensor1_->samples_count()
                   ? tensor1_->request_sample_shape(index, options)
                   : tensor2_->request_sample_shape(index - tensor1_->samples_count(), options);
    }

    async::promise<nd::array>
    request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array>
    request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override
    {
        return index < tensor1_->samples_count() ? tensor1_->request_bytes(index, options)
                                                 : tensor2_->request_bytes(index - tensor1_->samples_count(), options);
    }

    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

private:
    nd::array remap_class_labels(nd::array array);

private:
    heimdall::column_view_ptr tensor1_;
    heimdall::column_view_ptr tensor2_;
    icm::shape min_shape_;
    icm::shape max_shape_;
    icm::json info_;
    merged_class_info maping_info_;
};

} // namespace heimdall_common::impl
