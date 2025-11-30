#pragma once

#include <deeplake_core/type.hpp>
#include <nd/array.hpp>

#include <memory>

namespace duckdb {
class Value;
class LogicalType;
class Vector;
}

namespace pg {

deeplake_core::type to_deeplake_type(const duckdb::LogicalType& duckdb_type);
nd::array to_deeplake_value(const duckdb::Value& value);
nd::array to_deeplake_value(std::shared_ptr<duckdb::Vector>&& vector, size_t row_count);

} // namespace pg
