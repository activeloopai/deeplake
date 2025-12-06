#pragma once

/**
 * @file schema.hpp
 * @brief Declaration of the schema interface.
 */

#include "schema_field.hpp"

#include <icm/schema.hpp>

namespace nd {

class type;
using schema = icm::schema<type>;

} // namespace nd
