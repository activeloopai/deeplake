#pragma once

#include "exceptions.hpp"

#include <async/promise.hpp>
#include <heimdall/column_view.hpp>
#include <heimdall/dataset_view.hpp>
#include <heimdall/exceptions.hpp>
#include <icm/bit_vector.hpp>
#include <query_core/functor.hpp>
#include <query_core/index_holder.hpp>

#include <nd/adapt.hpp>
#include <nd/type.hpp>

namespace heimdall_common {

class unary_transformed_column : public heimdall::column_view, public query_core::index_holder
{
public:
    unary_transformed_column(heimdall::column_view_ptr source,
                             query_core::static_data_t data,
                             const query_core::array_generic_functor& functor,
                             std::string name)
        : source_(std::move(source))
        , data_(std::move(data))
        , functor_(functor.batch_f())
        , name_(std::move(name))
    {
    }

    unary_transformed_column(const unary_transformed_column&) = delete;
    unary_transformed_column& operator=(const unary_transformed_column&) = delete;
    unary_transformed_column(unary_transformed_column&&) = default;
    unary_transformed_column& operator=(unary_transformed_column&&) = default;
    ~unary_transformed_column() override = default;

public:
    inline const std::string& name() const noexcept override
    {
        return name_;
    }

    const deeplake_core::type type() const noexcept override
    {
        return deeplake_core::type::generic(functor_.type());
    }

    inline const auto& functor() const noexcept
    {
        return functor_;
    }

    inline base::htype htype() const noexcept override
    {
        return functor_.htype();
    }

    inline codecs::compression compression() const noexcept override
    {
        return codecs::compression::null;
    }

    inline icm::shape min_shape() const noexcept override
    {
        return {};
    }

    inline icm::shape max_shape() const noexcept override
    {
        return {};
    }

    inline int64_t samples_count() const noexcept override
    {
        return source_->samples_count();
    }

    const icm::json& metadata() const noexcept override;

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        throw exception("Links info is not available for virtual columns.");
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
    std::shared_ptr<query_core::index_holder> index_holder() override
    {
        if (source_->index_holder()) {
            return std::static_pointer_cast<query_core::index_holder>(shared_from_this_());
        }
        return nullptr;
    }

private:
    std::shared_ptr<unary_transformed_column> shared_from_this_()
    {
        return std::static_pointer_cast<unary_transformed_column>(heimdall::column_view::shared_from_this());
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
        if (!static_cast<bool>(info.filter_expr) && static_cast<bool>(info.order_expr) &&
            info.order_expr.get_type() == query_core::expr_type::column_ref) {
            auto ci = info;
            ci.order_expr = functor_.get_expr();
            return source_->index_holder()->can_run_query(ci);
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
        auto ci = info;
        ci.order_expr = functor_.get_expr();
        auto new_data = data;
        new_data.insert(data_.begin(), data_.end());
        return source_->index_holder()->run_query(ci, new_data, filter);
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
    async::promise<nd::array> request_full(storage::fetch_options options) override;
    async::promise<nd::array> request_shapes_full(storage::fetch_options options) override;

public:
    bool can_fetch_bytes() const noexcept override
    {
        return functor_.is_identity() && source_->can_fetch_bytes();
    }

    async::promise<nd::array> request_bytes_full(storage::fetch_options options) override
    {
        if (!can_fetch_bytes()) {
            throw heimdall::invalid_operation("Can't fetch bytes of the virtual column.");
        }
        return source_->request_bytes_full(options);
    }

public:
    bool is_chunked() const noexcept override
    {
        return source_->is_chunked();
    }

    int64_t chunk_size_hint() const override
    {
        return source_->chunk_size_hint();
    }

    std::vector<int64_t> chunk_ranges() const override
    {
        return source_->chunk_ranges();
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

    async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) override
    {
        if (!can_fetch_bytes()) {
            throw heimdall::invalid_operation("Can't fetch bytes of the virtual column.");
        }
        return source_->request_bytes(index, options);
    }

    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        if (!can_fetch_bytes()) {
            throw heimdall::invalid_operation("Can't fetch bytes of the virtual column.");
        }
        return source_->request_range_bytes(start_index, end_index, options);
    }

private:
    void initialize();

private:
    heimdall::column_view_ptr source_;
    query_core::static_data_t data_;
    query_core::array_batch_functor functor_;
    std::string name_;
    icm::shape min_shape_;
    icm::shape max_shape_;
};

} // namespace heimdall_common
