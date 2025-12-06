#pragma once

#include "impl/scalar_column.hpp"

#include <memory>
#include <vector>

namespace heimdall_common {

class column_view;

/**
 * @brief Creates a scalar column with the given name and data.
 *
 * @tparam T The type of the data.
 * @param name The name of the column.
 * @param data The data of the column.
 * @return column_view_ptr The created column.
 */
template <typename T>
heimdall::column_view_ptr make_scalar_column(std::string name, std::vector<T>&& data)
{
    return std::make_shared<impl::scalar_column<T>>(std::move(name), std::move(data));
}

}
