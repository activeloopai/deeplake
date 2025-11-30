#pragma once

#include "exceptions.hpp"
#include "row_range_view.hpp"
#include "row_view.hpp"
#include "schema_view.hpp"

#include <icm/index_based_iterator.hpp>
#include <icm/parse_negative_index.hpp>

#include <memory>

namespace heimdall {

class column_view;

namespace impl {
class dataset_iterator;
class const_dataset_iterator;
} // namespace impl

/**
 * @brief Abstraction over the dataset
 *
 * Provides minimal interface dataset should have:
 * - Tensors count
 * - Access to tensors
 *
 * There are two main groups of "things" called dataset:
 * - materialized datasets or storage level datasets
 * - dataset views, which are usually result of query or dataset slicing or dataset projection
 *   by specific column list.
 *
 * In addition to this, each group has different types of datasets, based on specific features such as:
 * For storage datasets:
 *     - tiling
 *     - downsampling
 *     - chunk compression
 *
 * For dataset views:
 *     - slicing/projection
 *     - ordering/shuffling
 *     - transformation
 *
 * This class along with class `column_view` abstracts all these types of datasets and
 * provides unified interface to the underlying data. The derived implementations of these classes
 * hide implementation details under the hood such as:
 * - decision on downsampling level and specific tiles
 * - downloading links
 * - decompression
 * - resizing
 * - query results handling
 * - query transformations apply
 *
 * The unified interface is used by client level code such as visualizer and query, to avoid focusing on
 * implementation specifics and focus on the logic only. They get `dataset_view` as an input. In addition,
 * query engine provides `dataset_view` as an output too.
 *
 * JS and Python apis rely on the implementation of these classes, although they have pieces of code which
 * depends on the dataset implementation specifics.
 *
 */
class dataset_view : public std::enable_shared_from_this<dataset_view>
{
public:
    dataset_view() = default;

    virtual ~dataset_view() = default;

public:
    class row_views_adaptor
    {
    public:
        row_views_adaptor(std::shared_ptr<dataset_view> ds)
            : ds_(std::move(ds))
        {
        }

        int64_t size() const
        {
            return ds_->num_rows();
        }

        bool empty() const
        {
            return size() == 0;
        }

        row_view operator[](int64_t index) const
        {
            return row_view(ds_, index);
        }

        row_view at(int64_t index) const
        {
            index = icm::parse_negative_index(index, size());
            return row_view(ds_, index);
        }

        row_range_view operator[](icm::index_mapping_t<int64_t> indices) const
        {
            return row_range_view(ds_, std::move(indices));
        }

        row_range_view at(icm::index_mapping_t<int64_t> indices) const
        {
            return row_range_view(ds_, std::move(indices));
        }

        impl::row_view_based_iterator begin() const
        {
            return impl::row_view_based_iterator(*ds_, 0L, impl::row_view_constructor);
        }

        impl::row_view_based_iterator end() const
        {
            return impl::row_view_based_iterator(*ds_, ds_->num_rows(), impl::row_view_constructor);
        }

        impl::row_view_based_iterator cbegin() const
        {
            return impl::row_view_based_iterator(*ds_, 0L, impl::row_view_constructor);
        }

        impl::row_view_based_iterator cend() const
        {
            return impl::row_view_based_iterator(*ds_, ds_->num_rows(), impl::row_view_constructor);
        }

    private:
        std::shared_ptr<dataset_view> ds_;
    };

    row_views_adaptor row_views()
    {
        return row_views_adaptor(shared_from_this());
    }

    /// @brief Returns the iterator to the first column of the dataset.
    impl::dataset_iterator begin();

    /// @brief Returns the iterator to the column after the last tensor of the dataset.
    impl::dataset_iterator end();

    /// @brief Returns the iterator to the first column of the dataset.
    impl::const_dataset_iterator begin() const;

    /// @brief Returns the iterator to the column after the last tensor of the dataset.
    impl::const_dataset_iterator end() const;

    /// @brief Returns the iterator to the first column of the dataset.
    impl::const_dataset_iterator cbegin() const;

    /// @brief Returns the iterator to the column after the last tensor of the dataset.
    impl::const_dataset_iterator cend() const;

    [[nodiscard]] virtual std::shared_ptr<schema_view> get_schema_view() const;

public:
    /// @brief Returns the column by the given index.
    column_view& operator[](int index)
    {
        return get_column_view(index);
    }

    /// @brief Returns the column by the given index.
    const column_view& operator[](int index) const
    {
        return const_cast<dataset_view*>(this)->get_column_view(index);
    }

    /// @brief Returns the column by the given name.
    column_view& operator[](std::string_view name);

    /// @brief Returns the column by the given name.
    const column_view& operator[](std::string_view name) const;

    /// @brief Returns the size of the dataset.
    int size() const
    {
        return columns_count();
    }

    /// @brief Returns the number of rows in the dataset.
    /// If the tensors have different number of rows, the maximal number of rows is returned.
    virtual int64_t num_rows() const;

    /// @brief Returns true if the dataset is empty.
    bool empty() const
    {
        return size() == 0;
    }

    virtual icm::json to_json() const = 0;

    std::string to_string() const;

    std::string summary() const;

    virtual const icm::json& metadata() const
    {
        static icm::json empty_json;
        return empty_json;
    }

protected:
    virtual int columns_count() const = 0;
    virtual column_view& get_column_view(int index) = 0;
};

using const_dataset_view_ptr = std::shared_ptr<const dataset_view>;

using dataset_view_ptr = std::shared_ptr<dataset_view>;

} // namespace heimdall

#include "impl/dataset_iterator.hpp"

namespace heimdall {

inline auto begin(dataset_view& ds)
{
    return impl::dataset_iterator(ds, 0);
}

inline auto end(dataset_view& ds)
{
    return impl::dataset_iterator(ds, ds.size());
}

inline auto begin(const dataset_view& ds)
{
    return impl::const_dataset_iterator(ds, 0);
}

inline auto end(const dataset_view& ds)
{
    return impl::const_dataset_iterator(ds, ds.size());
}

inline auto cbegin(const dataset_view& ds)
{
    return impl::const_dataset_iterator(ds, 0);
}

inline auto cend(const dataset_view& ds)
{
    return impl::const_dataset_iterator(ds, ds.size());
}

inline impl::dataset_iterator dataset_view::begin()
{
    return impl::dataset_iterator(*this, 0);
}

inline impl::dataset_iterator dataset_view::end()
{
    return impl::dataset_iterator(*this, size());
}

inline impl::const_dataset_iterator dataset_view::begin() const
{
    return impl::const_dataset_iterator(*this, 0);
}

inline impl::const_dataset_iterator dataset_view::end() const
{
    return impl::const_dataset_iterator(*this, size());
}

inline impl::const_dataset_iterator dataset_view::cbegin() const
{
    return impl::const_dataset_iterator(*this, 0);
}

inline impl::const_dataset_iterator dataset_view::cend() const
{
    return impl::const_dataset_iterator(*this, size());
}
/**
 * @brief Calculates approximate sample size in bytes for the given dataset.
 *
 * @param d dataset_view
 * @return Average bytes.
 */
uint64_t dataset_sample_bytes(const dataset_view& d);

/**
 * @brief Calculates approximate total size of the dataset in bytes.
 *
 * @param d dataset_view
 * @return Average bytes.
 */
uint64_t dataset_total_bytes(const dataset_view& d);

/**
 * @brief Returns the minimal length across all the tensors.
 */
int64_t min_size(const dataset_view& d);

/**
 * @brief Returns the maximal length across all the tensors.
 */
int64_t max_size(const dataset_view& d);

} // namespace heimdall
