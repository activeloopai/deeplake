#pragma once

#include "../exceptions.hpp"

#include <async/promise.hpp>
#include <heimdall/column_view.hpp>
#include <heimdall/links_info_holder.hpp>
#include <icm/json.hpp>
#include <icm/small_vector.hpp>

namespace heimdall_common::impl {

inline icm::json merge_json(const icm::json& j1, const icm::json& j2)
{
    if (j1.is_null()) {
        return j2;
    }
    if (j2.is_null()) {
        return j1;
    }
    if (!j1.is_object() || !j2.is_object()) {
        return j1;
    }
    auto result = j1;
    for (const auto& [key, value] : j2) {
        if (result.contains(key) && result[key].is_object() && value.is_object()) {
            result[key] = merge_json(result[key], value);
        } else if (!result.contains(key)) {
            result[key] = value;
        }
    }
    return result;
}

class merged_column : public heimdall::column_view
{
public:
    merged_column(heimdall::column_view_ptr t1, heimdall::column_view_ptr t2)
        : column1_(std::move(t1))
        , column2_(std::move(t2))
        , info_(merge_json(column1_->metadata(), column2_->metadata()))
    {
        ASSERT(column1_->name() == column2_->name());
        ASSERT(column1_->htype() == column2_->htype());
        ASSERT(column1_->dtype() == column2_->dtype());
        ASSERT(column1_->is_sequence() == column2_->is_sequence());
        icm::small_vector<icm::shape::value_type> shape;
        const auto& min1 = column1_->min_shape();
        const auto& min2 = column2_->min_shape();
        shape.resize(std::max(min1.size(), min2.size()), 1u);
        for (auto i = 0; i < std::min(min1.size(), min2.size()); ++i) {
            shape[i] = std::min(min1[i], min2[i]);
        }
        min_shape_ = icm::shape(shape.begin(), shape.end());
        const auto& max1 = column1_->max_shape();
        const auto& max2 = column2_->max_shape();
        shape.clear();
        shape.resize(std::max(max1.size(), max2.size()), 1u);
        for (auto i = 0; i < shape.size(); ++i) {
            shape[i] = std::max(i < max1.size() ? max1[i] : 1u, i < max2.size() ? max2[i] : 1u);
        }
        max_shape_ = icm::shape(shape.begin(), shape.end());
    }

public:
    const std::string& name() const noexcept override
    {
        return column1_->name();
    }

    const deeplake_core::type type() const noexcept override
    {
        return column1_->type();
    }

    base::htype htype() const noexcept override
    {
        return column1_->htype();
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
        return column1_->samples_count() + column2_->samples_count();
    }

    const icm::json& metadata() const noexcept override
    {
        return info_;
    }

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        throw heimdall_common::exception("Links info is not available for merged columns.");
    }

public:
    bool is_sequence() const noexcept override
    {
        return column1_->is_sequence();
    }

    int64_t sequence_length(int64_t index) const override
    {
        return index < column1_->samples_count() ? column1_->sequence_length(index)
                                                 : column2_->sequence_length(index - column1_->samples_count());
    }

public:
    async::promise<nd::array> request_full(storage::fetch_options options) override
    {
        return request_range(0, samples_count(), options);
    }

    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override
    {
        return request_range_shape(0, samples_count(), options);
    }

public:
    bool can_fetch_bytes() const noexcept override
    {
        return column1_->can_fetch_bytes() && column2_->can_fetch_bytes();
    }

    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override
    {
        return request_range_bytes(0, samples_count(), options);
    }

public:
    bool is_chunked() const noexcept override
    {
        return column1_->is_chunked() && column2_->is_chunked();
    }

    int64_t chunk_size_hint() const override
    {
        return std::min(column1_->chunk_size_hint(), column2_->chunk_size_hint());
    }

    std::vector<int64_t> chunk_ranges() const override
    {
        auto c1 = column1_->chunk_ranges();
        auto c2 = column2_->chunk_ranges();
        auto offset = column1_->samples_count();
        c1.reserve(c1.size() + c2.size());
        for (auto c : c2) {
            c1.push_back(c + offset);
        }
        return c1;
    }

protected:
    async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) override
    {
        return index < column1_->samples_count() ? column1_->request_sample(index, options)
                                                 : column2_->request_sample(index - column1_->samples_count(), options);
    }

    async::promise<nd::array>
    request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) override
    {
        return index < column1_->samples_count()
                   ? column1_->request_sample(index, source_shape, options)
                   : column2_->request_sample(index - column1_->samples_count(), source_shape, options);
    }

    async::promise<nd::array>
    request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) override
    {
        return index < column1_->samples_count()
                   ? column1_->request_sample(index, result_shape, options)
                   : column2_->request_sample(index - column1_->samples_count(), result_shape, options);
    }

    async::promise<nd::array> request_sample_(int64_t index,
                                              const icm::indexable_vector& source_shape,
                                              std::span<int64_t> result_shape,
                                              storage::fetch_options options) override
    {
        return index < column1_->samples_count()
                   ? column1_->request_sample(index, source_shape, result_shape, options)
                   : column2_->request_sample(index - column1_->samples_count(), source_shape, result_shape, options);
    }

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override
    {
        return index < column1_->samples_count()
                   ? column1_->request_sample_shape(index, options)
                   : column2_->request_sample_shape(index - column1_->samples_count(), options);
    }

    async::promise<nd::array>
    request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array>
    request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override
    {
        return index < column1_->samples_count() ? column1_->request_bytes(index, options)
                                                 : column2_->request_bytes(index - column1_->samples_count(), options);
    }

    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

private:
    heimdall::column_view_ptr column1_;
    heimdall::column_view_ptr column2_;
    icm::shape min_shape_;
    icm::shape max_shape_;
    icm::json info_;
};

} // namespace heimdall_common::impl
