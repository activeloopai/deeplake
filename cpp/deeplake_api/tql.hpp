#pragma once

#include <tql/executor.hpp>
#include <tql/explain_query_result.hpp>
#include <tql/tql.hpp>

namespace deeplake_api::tql {

using ::tql::query;

using ::tql::prepare_query;

using ::tql::explain_query;

using ::tql::register_function;

using ::tql::unregister_function;

using ::tql::options;

using executor = ::tql::executor;

using explain_query_result = ::tql::explain_query_result;

} // namespace deeplake_api::tql
