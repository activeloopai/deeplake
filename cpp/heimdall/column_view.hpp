#pragma once

#include "sample_info_holder.hpp"

#include <base/htype.hpp>
#include <codecs/compression.hpp>
#include <deeplake_core/type.hpp>
#include <heimdall/links_info_holder.hpp>
#include <icm/indexable.hpp>
#include <icm/json.hpp>
#include <icm/parse_negative_index.hpp>
#include <icm/roaring.hpp>
#include <icm/shape.hpp>
#include <nd/dtype.hpp>
#include <query_core/index_holder.hpp>
#include <storage/fetch_options.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace async {
template <typename T>
class promise;
}

namespace nd {
class array;
}

namespace heimdall {

/**
 * @brief Abstraction over columns.
 *
 * This class represents abstract column with the interface to get metadata of the column, such as
 * `dtype`, `htype`, `shape`, etc. As well as request the specific samples and sample ranges.
 *
 * Column view should always be managed by shared pointer.
 *
 */
class column_view : public std::enable_shared_from_this<column_view>
{
public:
    virtual ~column_view() = default;

public:
    /// @name Metadata access
    /// @{

    /// @brief Name of column.
    virtual const std::string& name() const noexcept = 0;

    virtual const deeplake_core::type type() const noexcept = 0;

    /// @brief Sample compression
    virtual codecs::compression compression() const noexcept = 0;

    /// @brief Min shape along all samples.
    virtual icm::shape min_shape() const noexcept = 0;

    /// @brief Max shape along all samples.
    virtual icm::shape max_shape() const noexcept = 0;

    /// @brief Number of samples.
    virtual int64_t samples_count() const noexcept = 0;

    /// @brief Additional meta info, packed in json.
    virtual const icm::json& metadata() const noexcept = 0;
    /// @}

public:
    /// @name Sequence info.
    /// @{

    /// @brief Check if the samples are sequences.
    virtual bool is_sequence() const noexcept = 0;

    /// @brief Sequence length for the given sample.
    virtual int64_t sequence_length(int64_t index) const = 0;
    /// @}

    /// @name Helper functions.
    /// @{
    [[nodiscard]] std::string to_string() const;

    /// @brief DType
    nd::dtype dtype() const noexcept
    {
        return type().data_type().get_dtype();
    }

    /// @brief HType
    virtual base::htype htype() const noexcept
    {
        return type().htype();
    }
    /// @}

    /// @name Views.
    /// @{
    virtual std::shared_ptr<heimdall::links_info_holder> links_holder() const = 0;

    virtual std::shared_ptr<heimdall::sample_info_holder> sample_info_holder()
    {
        return nullptr;
    }

    virtual std::shared_ptr<query_core::index_holder> index_holder()
    {
        return nullptr;
    }
    /// @}

public:
    /// @name Data fetching.
    /// @{

    /**
     * @brief Request the sample by the given index.
     * @param index Index of the sample.
     * @param options Fetch options.
     * @return Promise with the sample.
     */
    async::promise<nd::array> request_sample(int64_t index, storage::fetch_options options)
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_sample_(index, options);
    }

    /**
     * @brief Request the sample by the given source shape.
     * @param index Index of the sample.
     * @param source_shape Source shape of the sample.
     * @param options Fetch options.
     * @return Promise with the sample.
     */
    async::promise<nd::array>
    request_sample(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options)
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_sample_(index, source_shape, options);
    }

    /**
     * @brief Request sample with the given index and reshape it to the given shape.
     * @param index Index of the sample.
     * @param result_shape Shape to reshape the sample.
     * @param options Fetch options.
     * @return Promise with the reshaped sample.
     */
    async::promise<nd::array>
    request_sample(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options)
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_sample_(index, result_shape, options);
    }

    /**
     * @brief Request the range of samples by the given source shape and reshape them to the given shape.
     * @param index Index of the sample.
     * @param source_shape Source shape of the sample.
     * @param result_shape Shape to reshape the sample.
     * @param options Fetch options.
     * @return Promise with the reshaped range of samples.
     */
    async::promise<nd::array> request_sample(int64_t index,
                                             const icm::indexable_vector& source_shape,
                                             std::span<int64_t> result_shape,
                                             storage::fetch_options options)
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_sample_(index, source_shape, result_shape, options);
    }

    /**
     * @brief Request the range of samples.
     * @param start_index Start index of the range.
     * @param end_index End index of the range.
     * @param options Fetch options.
     * @return Promise with the range of samples.
     */
    async::promise<nd::array> request_range(int64_t start_index, int64_t end_index, storage::fetch_options options)
    {
        std::tie(start_index, end_index) = icm::parse_negative_index(start_index, end_index, samples_count());
        return request_range_(start_index, end_index, options);
    }

    /**
     * @brief Request all samples in the column.
     * @param options Fetch options.
     * @return Promise with all samples.
     */
    virtual async::promise<nd::array> request_full(storage::fetch_options options) = 0;
    /// @}

    /// @name Shapes fetching.
    /// @{

    /**
     * @brief Request the shape of the sample by the given index.
     * @param index Index of the sample.
     * @param options Fetch options.
     * @return Promise with the shape of the sample.
     */
    async::promise<nd::array> request_sample_shape(int64_t index, storage::fetch_options options)
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_sample_shape_(index, options);
    }

    /**
     * @brief Request Shapes of the samples in the given range.
     * @param start_index Start index of the range.
     * @param end_index End index of the range.
     * @param options Fetch options.
     * @return Promise with the shapes of the samples.
     */
    async::promise<nd::array>
    request_range_shape(int64_t start_index, int64_t end_index, storage::fetch_options options)
    {
        std::tie(start_index, end_index) = icm::parse_negative_index(start_index, end_index, samples_count());
        return request_range_shape_(start_index, end_index, options);
    }

    /**
     * @brief Request the shapes of all samples in the column.
     * @param options Fetch options.
     * @return Promise with the shapes of all samples.
     */
    virtual async::promise<nd::array> request_shapes_full(storage::fetch_options options) = 0;
    /// @}

public:
    /// @name Bytes fetching.
    /// @{

    /// @brief Check if the column can fetch bytes.
    virtual bool can_fetch_bytes() const noexcept = 0;

    /**
     * @brief Request the bytes of the sample by the given index.
     * @param index Index of the sample.
     * @param options Fetch options.
     * @return Promise with the bytes of the sample.
     */
    async::promise<nd::array> request_bytes(int64_t index, storage::fetch_options options)
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_bytes_(index, options);
    }

    /**
     * @brief Request the bytes of the given range of samples.
     * @param start_index Start index of the range.
     * @param end_index End index of the range.
     * @param options Fetch options.
     * @return Promise with the bytes of the samples.
     */
    async::promise<nd::array>
    request_range_bytes(int64_t start_index, int64_t end_index, storage::fetch_options options)
    {
        std::tie(start_index, end_index) = icm::parse_negative_index(start_index, end_index, samples_count());
        return request_range_bytes_(start_index, end_index, options);
    }

    /**
     * @brief Request the bytes of all samples in the column.
     * @param options Fetch options.
     * @return Promise with the bytes of all samples.
     */
    virtual async::promise<nd::array> request_bytes_full(storage::fetch_options options) = 0;
    /// @}

public:
    /// @name Chunking.
    /// @{

    /// @brief Check if the column is chunked.
    virtual bool is_chunked() const noexcept = 0;

    /// @brief Get the number of chunks.
    /// @return the number of chunks.
    virtual int64_t chunks_count_hint() const;

    /// @brief get overall understanding of the chunk size.
    /// @return the size of the chunk in bytes.
    virtual int64_t chunk_size_hint() const = 0;

    /// @brief Get the chunk ranges.
    virtual std::vector<int64_t> chunk_ranges() const = 0;
    /// @}

    /// @name Stats.
    /// @{

    /// @brief Get the approximate total bytes of the column.
    virtual uint64_t approximate_total_bytes() const;
    /// @}

public:
    /// @name Utilities for data fetching.
    /// @{

    /**
     * @brief Request the multiple samples by the given indices.
     * @param indices Indices of the samples.
     * @param options Fetch options.
     * @return Promise with the samples.
     */
    async::promise<nd::array> request_multiple_rows(const icm::index_mapping_t<int64_t>& indices,
                                                    storage::fetch_options options);

    async::promise<nd::array> request_multiple_rows(const icm::roaring& indices, storage::fetch_options options);

    /**
     * @brief Request the multiple sample shapes by the given indices.
     * @param indices Indices of the samples.
     * @param options Fetch options.
     * @return Promise with the sample shapes.
     */
    async::promise<nd::array> request_multiple_row_shapes(const icm::index_mapping_t<int64_t>& indices,
                                                          storage::fetch_options options);

    /**
     * @brief Request the multiple sample bytes by the given indices.
     * @param indices Indices of the samples.
     * @param options Fetch options.
     * @return Promise with the sample bytes.
     */
    async::promise<nd::array> request_multiple_row_bytes(const icm::index_mapping_t<int64_t>& indices,
                                                         storage::fetch_options options);
    /// @}

protected:
    friend class links_info_holder;
    async::promise<std::string> creds_key() const
    {
        return creds_key_();
    }

    async::promise<nd::array> request_links_info(int64_t index, storage::fetch_options options) const
    {
        index = icm::parse_negative_index(index, samples_count());
        return request_links_info_(index, options);
    }

    async::promise<nd::array>
    request_links_info_range(int64_t start_index, int64_t end_index, storage::fetch_options options) const
    {
        std::tie(start_index, end_index) = icm::parse_negative_index(start_index, end_index, samples_count());
        return request_links_info_range_(start_index, end_index, options);
    }

    async::promise<nd::array> request_links_info_full(storage::fetch_options options) const
    {
        return request_links_info_range(0, samples_count(), options);
    }

protected:
    virtual async::promise<nd::array> request_sample_(int64_t index, storage::fetch_options options) = 0;
    virtual async::promise<nd::array>
    request_sample_(int64_t index, const icm::indexable_vector& source_shape, storage::fetch_options options) = 0;
    virtual async::promise<nd::array>
    request_sample_(int64_t index, std::span<int64_t> result_shape, storage::fetch_options options) = 0;
    virtual async::promise<nd::array> request_sample_(int64_t index,
                                                      const icm::indexable_vector& source_shape,
                                                      std::span<int64_t> result_shape,
                                                      storage::fetch_options options) = 0;
    virtual async::promise<nd::array>
    request_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) = 0;
    virtual async::promise<nd::array> request_sample_shape_(int64_t index, storage::fetch_options options) = 0;
    virtual async::promise<nd::array>
    request_range_shape_(int64_t start_index, int64_t end_index, storage::fetch_options options) = 0;
    virtual async::promise<nd::array> request_bytes_(int64_t index, storage::fetch_options options) = 0;
    virtual async::promise<nd::array>
    request_range_bytes_(int64_t start_index, int64_t end_index, storage::fetch_options options) = 0;

    virtual async::promise<std::string> creds_key_() const
    {
        throw std::runtime_error("creds_key is not implemented.");
    }

    virtual async::promise<nd::array> request_links_info_(int64_t index, storage::fetch_options options) const
    {
        throw std::runtime_error("request_links_info is not implemented.");
    }

    virtual async::promise<nd::array>
    request_links_info_range_(int64_t start_index, int64_t end_index, storage::fetch_options options) const
    {
        throw std::runtime_error("request_links_info_range is not implemented.");
    }
};

using const_column_view_ptr = std::shared_ptr<const column_view>;

using column_view_ptr = std::shared_ptr<column_view>;

} // namespace heimdall
