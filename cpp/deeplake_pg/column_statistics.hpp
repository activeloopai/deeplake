#pragma once

#include <postgres.h>
#include <utils/rel.h>

#include <cstdint>

namespace pg {

/**
 * @brief Injects pre-computed DeepLake column statistics into PostgreSQL's pg_statistic.
 *
 * This function retrieves column statistics from the DeepLake dataset and inserts/updates
 * the corresponding pg_statistic rows. This is more efficient than PostgreSQL's sampling-based
 * ANALYZE since DeepLake already computes statistics incrementally during writes.
 *
 * The following statistics are injected:
 * - stanullfrac: Fraction of NULL values (0.0 to 1.0)
 * - stawidth: Average width in bytes
 * - stadistinct: Number of distinct values (PostgreSQL convention: positive=count, negative=fraction)
 * - stakind1=1 (MCV): Most common values and their frequencies
 *
 * @param rel The relation to inject statistics for
 * @return true if statistics were successfully injected, false otherwise
 */
bool inject_deeplake_statistics(Relation rel);

/**
 * @brief Injects statistics for a single column.
 *
 * @param rel The relation
 * @param attnum The attribute number (1-based)
 * @return true if successful
 */
bool inject_column_statistics(Relation rel, int16_t attnum);

} // namespace pg
