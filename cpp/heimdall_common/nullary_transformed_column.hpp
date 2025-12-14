#pragma once

#include "exceptions.hpp"

#include <async/promise.hpp>
#include <heimdall/column_view.hpp>
#include <heimdall/exceptions.hpp>
#include <query_core/functor.hpp>
#include <nd/adapt.hpp>
#include <nd/type.hpp>

namespace heimdall_common {

class nullary_transformed_column : public heimdall::column_view
{
public:
    nullary_transformed_column(query_core::static_data_t&& data,
                               const query_core::array_generic_functor& functor,
                               const std::string& name,
                               int64_t samples_count)
        : heimdall::column_view()
        , data_(std::move(data))
        , functor_(functor.batch_f())
        , name_(name)
        , samples_count_(samples_count)
    {
        initialize();
    }

    nullary_transformed_column(const nullary_transformed_column&) = delete;
    nullary_transformed_column& operator=(const nullary_transformed_column&) = delete;
    nullary_transformed_column(nullary_transformed_column&&) = default;
    nullary_transformed_column& operator=(nullary_transformed_column&&) = default;
    ~nullary_transformed_column() override = default;

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
        return min_shape_;
    }

    inline icm::shape max_shape() const noexcept override
    {
        return max_shape_;
    }

    inline int64_t samples_count() const noexcept override
    {
        return samples_count_;
    }

    const icm::json& metadata() const noexcept override;

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        throw heimdall_common::exception("Links info is not available for virtual columns.");
    }

public:
    inline bool is_sequence() const noexcept override
    {
        return false;
    }

    inline int64_t sequence_length(int64_t index) const override
    {
        throw exception("Virtual columns are not sequences.");
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
        throw heimdall::invalid_operation("Can't fetch bytes of the virtual column.");
    }

public:
    bool is_chunked() const noexcept override
    {
        return false;
    }

    int64_t chunk_size_hint() const override
    {
        throw heimdall::invalid_operation("Virtual column is not chunked.");
    }

    std::vector<int64_t> chunk_ranges() const override
    {
        throw heimdall::invalid_operation("Virtual column is not chunked.");
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
        throw heimdall::invalid_operation("Can't fetch bytes of the virtual column.");
    }

    async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) override
    {
        throw heimdall::invalid_operation("Can't fetch bytes of the virtual column.");
    }

private:
    void initialize();

private:
    query_core::static_data_t data_;
    query_core::array_batch_functor functor_;
    std::string name_;
    icm::shape max_shape_;
    icm::shape min_shape_;
    int64_t samples_count_;
};

} // namespace heimdall_common
