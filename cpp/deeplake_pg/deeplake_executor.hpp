#pragma once

// NOTE: postgres.h and libintl.h must be included first in the compilation unit before this header

#ifdef __cplusplus
extern "C" {
#endif

#include <nodes/execnodes.h>
#include <nodes/pathnodes.h>
#include <nodes/plannodes.h>
#include <optimizer/optimizer.h>

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Register the DeepLake direct query executor
PGDLLEXPORT void register_deeplake_executor();

// Create a planned statement for direct deeplake execution
PGDLLEXPORT PlannedStmt* deeplake_create_direct_execution_plan(
    Query* parse, 
    const char* query_string, 
    int32_t cursorOptions, 
    ParamListInfo boundParams
);

#ifdef __cplusplus
}
#endif

namespace pg {
void analyze_plan(PlannedStmt* plan);
} // namespace pg
