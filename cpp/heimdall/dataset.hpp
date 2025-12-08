#pragma once

#include "column.hpp"
#include "dataset_view.hpp"
#include "row.hpp"
#include "row_range.hpp"

#include <icm/index_based_iterator.hpp>
#include <icm/parse_negative_index.hpp>
#include <icm/string_map.hpp>

namespace heimdall {

/**
 * @brief Dataset represents the mutable dataset, while dataset_view represents the immutable dataset.
 * It has the same interface as dataset_view, but with additional methods to modify the dataset. The modifications are
 * the following:
 * - Add a new column
 * - Remove a column
 * - Rename a column
 * - Change the type of a column
 * - Add a new row
 * - Remove a row
 * - Update a row
 * - Add range of rows
 * - Remove range of rows
 * - Update range of rows
 */
class dataset : public dataset_view
{
public:
    class rows_adaptor
    {
    public:
        rows_adaptor(std::shared_ptr<dataset> ds)
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

        row operator[](int64_t index) const
        {
            return row(ds_, index);
        }

        row at(int64_t index) const
        {
            index = icm::parse_negative_index(index, size());
            return row(ds_, index);
        }

        row_range operator[](icm::index_mapping_t<int64_t> rows) const
        {
            return row_range(ds_, std::move(rows));
        }

        row_range at(icm::index_mapping_t<int64_t> rows) const
        {
            return row_range(ds_, std::move(rows));
        }

        impl::row_based_iterator begin() const
        {
            return impl::row_based_iterator(*ds_, 0L, impl::row_constructor);
        }

        impl::row_based_iterator end() const
        {
            return impl::row_based_iterator(*ds_, ds_->num_rows(), impl::row_constructor);
        }

        impl::row_based_iterator cbegin() const
        {
            return impl::row_based_iterator(*ds_, 0L, impl::row_constructor);
        }

        impl::row_based_iterator cend() const
        {
            return impl::row_based_iterator(*ds_, ds_->num_rows(), impl::row_constructor);
        }

    private:
        std::shared_ptr<dataset> ds_;
    };

    rows_adaptor rows()
    {
        return rows_adaptor(std::static_pointer_cast<dataset>(shared_from_this()));
    }

    class columns_adaptor
    {
    public:
        columns_adaptor(dataset* ds)
            : ds_(ds)
        {
        }

        int64_t size() const
        {
            return ds_->size();
        }

        bool empty() const
        {
            return size() == 0;
        }

        const column& operator[](int64_t index) const
        {
            return ds_->get_column(index);
        }

        const column& at(int64_t index) const
        {
            if (index < 0 || index >= size()) {
                throw column_out_of_range(index, size());
            }
            return ds_->get_column(index);
        }

        column& operator[](int64_t index)
        {
            return ds_->get_column(index);
        }

        column& at(int64_t index)
        {
            if (index < 0 || index >= size()) {
                throw column_out_of_range(index, size());
            }
            return ds_->get_column(index);
        }

        column& operator[](std::string_view column_name);

        const column& operator[](std::string_view column_name) const;

        using iterator = icm::mutable_index_based_iterator<columns_adaptor, column&, icm::use_container_index_tag>;
        using const_iterator = icm::index_based_iterator<columns_adaptor, const column&, icm::use_container_index_tag>;

        iterator begin()
        {
            return iterator(*this, 0L);
        }

        iterator end()
        {
            return iterator(*this, ds_->size());
        }

        const_iterator cbegin() const
        {
            return const_iterator(*this, 0L);
        }

        const_iterator cend() const
        {
            return const_iterator(*this, ds_->size());
        }

    private:
        dataset* ds_;
    };

public:
    /**
     * @brief Appends the given row at the end of the dataset.
     * @param row The row to append.
     */
    [[nodiscard]] virtual async::promise<void> append_row(const icm::string_map<nd::array>& row) = 0;

    /**
     * @brief Appends the given rows at the end of the dataset.
     * @param rows The rows to append.
     */
    [[nodiscard]] virtual async::promise<void> append_rows(const icm::string_map<nd::array>& rows) = 0;

    /**
     * @brief Updates the cell at the given row_id and column_name with the new value.
     * @param row_id The row_id of the cell to update.
     * @param column_name The column_name of the cell to update.
     * @param new_value The new value to set.
     */
    [[nodiscard]] virtual async::promise<void>
    update_row(int64_t row_id, const std::string& column_name, const nd::array& new_value) = 0;

    /**
     * @brief Updates the cells in the given range with the new value.
     * @param start_row_id The start row_id of the range to update.
     * @param end_row_id The end row_id of the range to update.
     * @param column_name The column_name of the cells to update.
     * @param new_value The new value to set.
     */
    [[nodiscard]] virtual async::promise<void> update_rows(int64_t start_row_id,
                                                           int64_t end_row_id,
                                                           const std::string& column_name,
                                                           const nd::array& new_value) = 0;

    /**
     * @brief Updates the given row with the new values.
     * If any of the columns is missing in the new_values, the corresponding column in the row will not be updated.
     * @param row_id The row_id of the row to update.
     * @param new_values The new values to set.
     */
    [[nodiscard]] virtual async::promise<void> update_row(int64_t row_id,
                                                          const icm::string_map<nd::array>& new_values) = 0;

    /**
     * @brief Updates the rows in the given range with the new values.
     * If any of the columns is missing in the new_values, the corresponding column in the row will not be updated.
     * @param start_row_id The start row_id of the range to update.
     * @param end_row_id The end row_id of the range to update.
     * @param new_values The new values to set.
     */
    [[nodiscard]] virtual async::promise<void>
    update_rows(int64_t start_row_id, int64_t end_row_id, const icm::string_map<nd::array>& new_values) = 0;

    /**
     * @brief Deletes the given row.
     * @param row_id The row_id of the row to delete.
     */
    virtual void delete_row(int64_t row_id) = 0;

    /**
     * @brief Deletes the given rows.
     * @param row_ids The row_ids of the rows to delete.
     */
    virtual void delete_rows(const std::vector<int64_t>& row_ids) = 0;

    /**
     * @brief Deletes the rows in the given range.
     * @param start_row_id The start row_id of the range to delete.
     * @param end_row_id The end row_id of the range to delete.
     */
    virtual void delete_rows(int64_t start_row_id, int64_t end_row_id) = 0;

    columns_adaptor columns()
    {
        return columns_adaptor(this);
    }

protected:
    virtual column& get_column(int index) = 0;

    column_view& get_column_view(int index) override
    {
        return get_column(index);
    }
};

} // namespace heimdall
