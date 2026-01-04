#pragma once

#ifdef __cplusplus
#define typeof __typeof__
extern "C" {
#endif

#include <postgres.h>
#include <nodes/nodes.h>
#include <nodes/parsenodes.h>
#include <nodes/primnodes.h>
#include <nodes/pg_list.h>
#include <utils/fmgroids.h>

#ifdef __cplusplus
} /// extern "C"
#endif


namespace pg {

// Helper function to detect COUNT(*) queries without WHERE/GROUP BY/HAVING
// Declared in deeplake_executor.hpp for use in deeplake_executor.cpp
bool is_pure_count_star_query(Query* parse)
{
    if (parse == nullptr) {
        return false;
    }

    // Must be SELECT
    if (parse->commandType != CMD_SELECT) {
        return false;
    }

    // Must have exactly one table
    if (parse->rtable == NIL || list_length(parse->rtable) != 1) {
        return false;
    }

    // Must not have WHERE clause
    if (parse->jointree && parse->jointree->quals != nullptr) {
        return false;
    }

    // Must not have GROUP BY
    if (parse->groupClause != NIL) {
        return false;
    }

    // Must not have HAVING
    if (parse->havingQual != nullptr) {
        return false;
    }

    // Must have exactly one target: COUNT(*)
    if (parse->targetList == NIL || list_length(parse->targetList) != 1) {
        return false;
    }

    TargetEntry* tle = (TargetEntry*)linitial(parse->targetList);

    // Check if target is COUNT(*) using helper from anonymous namespace
    if (tle == nullptr || tle->expr == nullptr || !IsA(tle->expr, Aggref)) {
        return false;
    }

    Aggref* agg = (Aggref*)tle->expr;
    return ((agg->aggfnoid == F_COUNT_ANY || agg->aggfnoid == F_COUNT_) && (agg->args == NIL || agg->aggstar));
}

}
