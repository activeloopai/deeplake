#pragma once

#include "exceptions.hpp"

#include <heimdall/column_view.hpp>
#include <heimdall/links_info_holder.hpp>

namespace heimdall_common {

/// @brief An abstract chained column view from another column view.
class chained_column_view : public heimdall::column_view
{
public:
    const std::string& name() const noexcept override
    {
        return source()->name();
    }

    const deeplake_core::type type() const noexcept override
    {
        return source()->type();
    }

    base::htype htype() const noexcept override
    {
        return source()->htype();
    }

    codecs::compression compression() const noexcept override
    {
        return source()->compression();
    }

    const icm::json& metadata() const noexcept override
    {
        return source()->metadata();
    }

    std::shared_ptr<heimdall::links_info_holder> links_holder() const override
    {
        return source()->links_holder();
    }

    bool is_chunked() const noexcept override
    {
        return false;
    }

    int64_t chunk_size_hint() const override
    {
        throw invalid_operation("The column is not chunked.");
    }

    std::vector<int64_t> chunk_ranges() const override
    {
        throw invalid_operation("The column is not chunked.");
    }

public:
    virtual heimdall::column_view_ptr source() = 0;
    virtual heimdall::const_column_view_ptr source() const = 0;
};

/**
 * @brief Finds the original column associated with the given column over the chain.
 *
 * @param t
 * @return column_view& Original column.
 */
heimdall::column_view& original_column_over_chain(heimdall::column_view& t);

/**
 * @brief Finds the original column associated with the given column over the chain.
 *
 * @param t
 * @return column_view& Original column.
 */
const heimdall::column_view& original_column_over_chain(const heimdall::column_view& t);

/**
 * @brief Finds the index_mapping applied to the original dataset.
 *
 * @param t
 * @return index_mapping
 */
icm::index_mapping_t<int64_t> index_mapping_on_column(const heimdall::column_view& t);

} // namespace heimdall_common
