#pragma once

#include <heimdall/column_view.hpp>
#include <heimdall/dataset_view.hpp>

#include <memory>

namespace heimdall_common {

/**
 * @brief An abstract dataset view that is based on another dataset view.
 */
class chained_dataset_view : public heimdall::dataset_view
{
public:
    /**
     * @brief Constructor.
     * @param source The source dataset view.
     */
    explicit chained_dataset_view(heimdall::dataset_view_ptr source)
        : dataset_view()
        , source_(source)
    {
    }

public:
    /// @brief Returns the source dataset view.
    const heimdall::dataset_view_ptr& source() const
    {
        return source_;
    }

    const std::string& query_string() const noexcept
    {
        return query_string_;
    }

    void set_query_string(const std::string& query_string)
    {
        query_string_ = query_string;
    }

protected:
    void assign(heimdall::dataset_view_ptr source)
    {
        source_ = source;
    }

private:
    icm::json to_json() const override;

private:
    heimdall::dataset_view_ptr source_;
    std::string query_string_;
};

/**
 * @brief A chained dataset view that is based on a source dataset and list of columns.
 */
class chained_dataset : public chained_dataset_view
{
public:
    chained_dataset(heimdall::dataset_view_ptr source, std::vector<heimdall::column_view_ptr>&& columns)
        : chained_dataset_view(source)
        , columns_(std::move(columns))
    {
    }

public:
    int columns_count() const override
    {
        return static_cast<int>(columns_.size());
    }

    heimdall::column_view& get_column_view(int index) override
    {
        return *columns_[index];
    }

private:
    std::vector<heimdall::column_view_ptr> columns_;
};

/// @brief Dataset that chains multiple dataset views.
class multiple_chained_dataset : public heimdall::dataset_view
{
public:
    multiple_chained_dataset(std::vector<heimdall::dataset_view_ptr> sources,
                             std::vector<heimdall::column_view_ptr>&& columns)
        : sources_(std::move(sources))
        , columns_(std::move(columns))
    {
    }

public:
    int columns_count() const override
    {
        return static_cast<int>(columns_.size());
    }

    heimdall::column_view& get_column_view(int index) override
    {
        return *columns_[index];
    }

    const std::vector<heimdall::dataset_view_ptr>& sources() const noexcept
    {
        return sources_;
    }

    const std::string& query_string() const noexcept
    {
        return query_string_;
    }

    void set_query_string(const std::string& query_string)
    {
        query_string_ = query_string;
    }

private:
    icm::json to_json() const override;

private:
    std::vector<heimdall::dataset_view_ptr> sources_;
    std::vector<heimdall::column_view_ptr> columns_;
    std::string query_string_;
};

/**
 * @brief Finds the original dataset view associated with the given dataset over the chain.
 *
 * @param d
 * @return std::vector<dataset_view_ptr> Original dataset.
 */
std::vector<heimdall::dataset_view_ptr> original_dataset_view_over_chain(heimdall::dataset_view_ptr d);

} // namespace heimdall_common
