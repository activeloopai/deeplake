#pragma once

#include "../chained_column_view.hpp"

#include <heimdall/exceptions.hpp>
#include <heimdall/sample_info_holder.hpp>
#include <icm/bit_vector.hpp>
#include <query_core/index_holder.hpp>

namespace heimdall_common::impl {

class filtered_tensor : public chained_column_view, public query_core::index_holder
{
public:
    filtered_tensor(heimdall::column_view_ptr s, icm::index_mapping_t<int64_t> i)
        : chained_column_view()
        , source_(std::move(s))
        , indices_(std::move(i))
    {
        ASSERT(source_ != nullptr);
    }

public:
    icm::shape max_shape() const noexcept override
    {
        return source_->max_shape();
    }

    icm::shape min_shape() const noexcept override
    {
        return source_->min_shape();
    }

    int64_t samples_count() const noexcept override
    {
        return indices_.size();
    }

public:
    bool is_sequence() const noexcept override
    {
        return source_->is_sequence();
    }

    int64_t sequence_length(int64_t index) const override
    {
        return source_->sequence_length(translate_index(index));
    }

public:
    std::shared_ptr<query_core::index_holder> index_holder() override
    {
        if (source_->index_holder()) {
            return std::dynamic_pointer_cast<query_core::index_holder>(shared_from_this());
        }
        return nullptr;
    }

private:
    std::shared_ptr<filtered_tensor> shared_from_this_()
    {
        return std::static_pointer_cast<filtered_tensor>(heimdall::column_view::shared_from_this());
    }

    void reset_index_data() override
    {
        auto h = source_->index_holder();
        ASSERT(h);
        h->reset_index_data();
    }

    std::string to_string() const override
    {
        auto h = source_->index_holder();
        if (h) {
            return h->to_string();
        }
        return "";
    }

    std::vector<deeplake_core::index_type> get_indexes() const override
    {
        auto h = source_->index_holder();
        if (h) {
            return h->get_indexes();
        }
        return {};
    }

    bool can_run_query(const query_core::top_k_search_info& info) const override
    {
        ASSERT(source_->index_holder());
        if (!static_cast<bool>(info.filter_expr) && static_cast<bool>(info.order_expr)) {
            return source_->index_holder()->can_run_query(info);
        }
        return false;
    }

    bool can_run_query(const query_core::text_search_info& info) const override
    {
        return false;
    }

    bool can_run_query(const query_core::inverted_index_search_info& info) const override
    {
        return false;
    }

    async::promise<query_core::query_results> run_query(const query_core::top_k_search_info& info,
                                                        const query_core::static_data_t& data,
                                                        std::shared_ptr<const icm::roaring> filter) override
    {
        // Convert indices_ (index_mapping) to bit_vector for the source universe size
        // TODO: We need the source dataset size here - this is the tricky part!
        // For now, we'll need to get it from the source somehow
        auto source_size = source_->samples_count();

        // Create a bit_vector representing our current indices_
        auto current_filter = std::make_shared<icm::roaring>();
        for (auto idx : indices_) {
            if (idx < source_size) {
                current_filter->add(idx);
            }
        }

        // If there's an additional filter, we need to apply it to our current indices
        std::shared_ptr<const icm::roaring> combined_filter;
        if (filter) {
            // The incoming filter operates on our filtered space, so we need to expand it
            // to the source space by applying it through our index mapping
            auto expanded_filter = std::make_shared<icm::roaring>();
            for (int64_t i = 0; i < filter->cardinality() && i < indices_.size(); ++i) {
                if (filter->contains(i)) {
                    auto source_idx = indices_[i];
                    if (source_idx < source_size) {
                        expanded_filter->add(source_idx);
                    }
                }
            }
            combined_filter = expanded_filter;
        } else {
            combined_filter = current_filter;
        }

        return source_->index_holder()
            ->run_query(info, data, combined_filter)
            .then([indices = indices_](auto&& results) {
                // Convert results back to our filtered index space
                for (auto& result : results) {
                    std::vector<int64_t> new_indices;
                    new_indices.reserve(result.indices.size());
                    for (auto source_idx : result.indices) {
                        // Find where this source index maps to in our filtered space
                        for (int64_t i = 0; i < indices.size(); ++i) {
                            if (indices[i] == source_idx) {
                                new_indices.push_back(i);
                                break;
                            }
                        }
                    }
                    result.indices = icm::index_mapping_t<int64_t>::list(std::move(new_indices));
                }
                return results;
            });
    }

    async::promise<std::vector<icm::roaring>> run_query(const query_core::text_search_info& info) override
    {
        throw heimdall::invalid_operation("Text search is not supported for virtual columns.");
    }

    async::promise<std::vector<icm::roaring>> run_query(const query_core::inverted_index_search_info& info) override
    {
        throw heimdall::invalid_operation("Inverted index search is not supported for virtual columns.");
    }

public:
    heimdall::column_view_ptr source() override
    {
        return source_;
    }

    heimdall::const_column_view_ptr source() const override
    {
        return source_;
    }

    const auto& indices() const
    {
        return indices_;
    }

public:
    async::promise<nd::array> request_full(storage::fetch_options options) override;
    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override;

public:
    bool can_fetch_bytes() const noexcept override;
    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override;

public:
    int64_t translate_index(int64_t index) const
    {
        return indices_[index];
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

    async::promise<nd::array>
    request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

    async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array>
    request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;
    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override;
    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override;

private:
    heimdall::column_view_ptr source_;
    icm::index_mapping_t<int64_t> indices_;
};

class filtered_sample_info_holder : public filtered_tensor, public heimdall::sample_info_holder
{
public:
    filtered_sample_info_holder(heimdall::column_view_ptr s,
                                std::shared_ptr<heimdall::sample_info_holder> h,
                                icm::index_mapping_t<int64_t> i)
        : filtered_tensor(std::move(s), std::move(i))
        , sample_info_source_(h)
    {
        ASSERT(h != nullptr);
        ASSERT(source()->sample_info_holder() == h);
    }

    std::shared_ptr<heimdall::sample_info_holder> sample_info_holder() override
    {
        return std::dynamic_pointer_cast<heimdall::sample_info_holder>(shared_from_this());
    }

    async::promise<icm::const_json> request_sample_info(int64_t index, storage::fetch_options options) override;
    async::promise<std::vector<icm::const_json>>
    request_sample_info_range(int64_t start_index, int64_t end_index, storage::fetch_options options) override;
    async::promise<std::vector<icm::const_json>> request_sample_info_full(storage::fetch_options options) override;

private:
    std::shared_ptr<heimdall::sample_info_holder> sample_info_source_ = nullptr;
};

} // namespace heimdall_common::impl
