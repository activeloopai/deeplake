# PostgreSQL to DuckDB JSON Query Translator

## Overview

This translator converts PostgreSQL JSON queries to DuckDB-compatible syntax, specifically handling differences in JSON operators, type casts, timestamp functions, and date operations.

## Files

- **[pg_to_duckdb_translator.hpp](pg_to_duckdb_translator.hpp)** - Header file with class declaration
- **[pg_to_duckdb_translator.cpp](pg_to_duckdb_translator.cpp)** - Implementation
- **[pg_to_duckdb_translator_test.cpp](pg_to_duckdb_translator_test.cpp)** - Test cases

## Translation Rules

### 1. JSON Operators
- PostgreSQL: `json -> 'key'` or `json ->> 'key'`
- DuckDB: `json->>'$.key'`
- Chained: `json -> 'a' -> 'b' ->> 'c'` → `json->>'$.a.b.c'`

### 2. Type Casts
- PostgreSQL: `(expr)::TYPE`, `identifier::TYPE`, or `'literal'::TYPE`
- DuckDB: `CAST(expr AS TYPE)`
- Supports compound types like `DOUBLE PRECISION`
- Supports string literals: `'text'::varchar` → `CAST('text' AS varchar)`

### 3. Timestamp Conversions
- PostgreSQL: `TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (json->>'time_us')::BIGINT`
- DuckDB: `TO_TIMESTAMP(CAST(json->>'$.time_us' AS BIGINT) / 1000000)`

### 4. Date Extraction
- PostgreSQL: `EXTRACT(HOUR FROM expr)`
- DuckDB: `hour(expr)`
- Supported fields: HOUR, DAY, MONTH, YEAR, MINUTE, SECOND, DOW, DOY, WEEK

### 5. Date Differences
- PostgreSQL: `EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) * 1000`
- DuckDB: `date_diff('milliseconds', MIN(ts), MAX(ts))`
- Multipliers: 1=seconds, 1000=milliseconds, 1000000=microseconds

### 6. IN Clauses
- PostgreSQL: `expr IN ('a', 'b', 'c')`
- DuckDB: `expr in ['a', 'b', 'c']`

### 7. WHERE Clause Predicates
- PostgreSQL: `WHERE a = 1 AND b = 2`
- DuckDB: `WHERE (a = 1) AND (b = 2)`
- Each predicate separated by AND/OR is wrapped in parentheses

### 8. Date Formatting Functions
- PostgreSQL: `to_date('2024-01-01', 'YYYY-MM-DD')`
- DuckDB: `strptime('2024-01-01', 'YYYY-MM-DD')`

### 9. Integer Division
- PostgreSQL: `DIV(a, b)`
- DuckDB: `(a // b)`

### 10. Regular Expression Functions
- PostgreSQL: `regexp_substr(string, pattern)`
- DuckDB: `regexp_extract(string, pattern)`

## Usage

```cpp
#include "pg_to_duckdb_translator.hpp"

std::string pg_query = "SELECT json -> 'commit' ->> 'collection' FROM table";
std::string duckdb_query = pg::pg_to_duckdb_translator::translate(pg_query);
```

## Integration

### Option 1: Integrate into DuckDB Executor

Add the translator to your existing query execution path:

```cpp
// In duckdb_executor.cpp
#include "pg_to_duckdb_translator.hpp"

std::unique_ptr<duckdb::MaterializedQueryResult>
execute_query(duckdb::Connection& conn, const std::string& pg_query) {
    // Translate PostgreSQL syntax to DuckDB syntax
    std::string duckdb_query = pg::pg_to_duckdb_translator::translate(pg_query);

    // Optional: Log the translation for debugging
    // elog(DEBUG1, "Original query: %s", pg_query.c_str());
    // elog(DEBUG1, "Translated query: %s", duckdb_query.c_str());

    // Execute the translated query
    auto result = conn.Query(duckdb_query);

    if (result->HasError()) {
        elog(ERROR, "DuckDB query failed: %s", result->GetError().c_str());
    }

    return result;
}
```

### Option 2: Add as a PostgreSQL Hook

Create a query rewrite hook:

```cpp
// In your extension initialization
static ProcessUtility_hook_type prev_ProcessUtility = NULL;

static void deeplake_ProcessUtility(
    PlannedStmt *pstmt,
    const char *queryString,
    ProcessUtilityContext context,
    ParamListInfo params,
    QueryEnvironment *queryEnv,
    DestReceiver *dest,
    QueryCompletion *qc)
{
    // Check if this is a SELECT query targeting your tables
    if (is_deeplake_query(queryString)) {
        // Translate the query
        std::string translated = pg::pg_to_duckdb_translator::translate(queryString);

        // Execute via DuckDB
        execute_via_duckdb(translated);
        return;
    }

    // Fall through to standard processing
    if (prev_ProcessUtility)
        prev_ProcessUtility(pstmt, queryString, context, params, queryEnv, dest, qc);
    else
        standard_ProcessUtility(pstmt, queryString, context, params, queryEnv, dest, qc);
}

void _PG_init(void) {
    // Install the hook
    prev_ProcessUtility = ProcessUtility_hook;
    ProcessUtility_hook = deeplake_ProcessUtility;
}
```

### Option 3: Automatic Translation Flag

Add an environment variable or GUC to enable automatic translation:

```cpp
// In extension_init.cpp or pg_deeplake.cpp
static bool enable_pg_to_duckdb_translation = true;

void _PG_init(void) {
    // Define GUC variable
    DefineCustomBoolVariable(
        "deeplake.enable_query_translation",
        "Enable automatic PostgreSQL to DuckDB query translation",
        NULL,
        &enable_pg_to_duckdb_translation,
        true,
        PGC_USERSET,
        0,
        NULL, NULL, NULL
    );
}

// Then in your query execution:
std::string prepare_query(const std::string& query) {
    if (enable_pg_to_duckdb_translation) {
        return pg::pg_to_duckdb_translator::translate(query);
    }
    return query;
}
```

## Example Queries

### Date Formatting with to_date

**PostgreSQL:**
```sql
SELECT id, name
FROM users
WHERE signup_date >= to_date('2024-01-01', 'YYYY-MM-DD');
```

**DuckDB:**
```sql
SELECT id, name
FROM users
WHERE (signup_date >= strptime('2024-01-01', 'YYYY-MM-DD'));
```

### Integer Division with DIV

**PostgreSQL:**
```sql
SELECT id, DIV(total_amount, 100) AS dollars
FROM transactions
WHERE DIV(quantity, 10) > 5;
```

**DuckDB:**
```sql
SELECT id, (total_amount // 100) AS dollars
FROM transactions
WHERE ((quantity // 10) > 5);
```

### Regular Expression Extraction

**PostgreSQL:**
```sql
SELECT id, regexp_substr(email, '[^@]+') AS username
FROM users
WHERE regexp_substr(domain, '[a-z]+') = 'example';
```

**DuckDB:**
```sql
SELECT id, regexp_extract(email, '[^@]+') AS username
FROM users
WHERE (regexp_extract(domain, '[a-z]+') = 'example');
```

### Simple WHERE Clauses

**PostgreSQL:**
```sql
SELECT COUNT(*) FROM bluesky
WHERE json ->> 'kind' = 'commit'
  AND json -> 'commit' ->> 'operation' = 'create';
```

**DuckDB:**
```sql
SELECT COUNT(*) FROM bluesky
WHERE json->>'$.kind' = 'commit'
  AND json->>'$.commit.operation' = 'create';
```

### Type Casting in WHERE

**PostgreSQL:**
```sql
SELECT id, json ->> 'repo' as repo
FROM bluesky
WHERE (json ->> 'seq')::BIGINT > 1000000;
```

**DuckDB:**
```sql
SELECT id, json->>'$.repo' as repo
FROM bluesky
WHERE CAST(json->>'$.seq' AS BIGINT) > 1000000;
```

### IN Clauses

**PostgreSQL:**
```sql
SELECT COUNT(*) FROM bluesky
WHERE json ->> 'kind' = 'commit'
  AND json -> 'commit' ->> 'collection' IN ('app.bsky.feed.post', 'app.bsky.feed.like');
```

**DuckDB:**
```sql
SELECT COUNT(*) FROM bluesky
WHERE json->>'$.kind' = 'commit'
  AND json->>'$.commit.collection' in ['app.bsky.feed.post', 'app.bsky.feed.like'];
```

### Complex Aggregation with Timestamps

**PostgreSQL:**
```sql
SELECT json->>'did' AS user_id,
       MIN(TIMESTAMP WITH TIME ZONE 'epoch' +
           INTERVAL '1 microsecond' * (json->>'time_us')::BIGINT) AS first_post_ts
FROM bluesky
WHERE json->>'kind' = 'commit'
GROUP BY user_id
LIMIT 3;
```

**DuckDB:**
```sql
SELECT json->>'$.did' AS user_id,
       MIN(TO_TIMESTAMP(CAST(json->>'$.time_us' AS BIGINT) / 1000000) ) AS first_post_ts
FROM bluesky
WHERE json->>'$.kind' = 'commit'
GROUP BY user_id
LIMIT 3;
```

## Testing

Run the standalone test:
```bash
cd cpp/deeplake_pg
chmod +x test_translator.sh
./test_translator.sh
```

All 24 test cases should pass:
- ✓ Test 1: Simple JSON access with GROUP BY
- ✓ Test 2: Multiple JSON access with WHERE
- ✓ Test 3: EXTRACT HOUR with IN clause
- ✓ Test 4: MIN with TIMESTAMP epoch conversion
- ✓ Test 5: Date difference with EXTRACT EPOCH
- ✓ Test 6: Simple WHERE with COUNT
- ✓ Test 7: WHERE with nested JSON and string comparison
- ✓ Test 8: WHERE with type cast
- ✓ Test 9: WHERE with IN clause and multiple JSON operators
- ✓ Test 10: Subquery with WHERE clause (regression test)
- ✓ Test 11: to_date to strptime conversion
- ✓ Test 12: DIV function to // operator
- ✓ Test 13: regexp_substr to regexp_extract
- ✓ Test 14: Combined date operations
- ✓ Test 15: Arithmetic with DIV and type casts
- ✓ Test 16: Pattern matching and string operations
- ✓ Test 17: Case insensitive TO_DATE
- ✓ Test 18: Nested functions with type casts
- ✓ Test 19: Complex WHERE with mixed operators
- ✓ Test 20: Multiple regexp_substr in SELECT and WHERE
- ✓ Test 21: DOUBLE PRECISION compound type
- ✓ Test 22: VARCHAR and TEXT type casts (real test from test_array.py)
- ✓ Test 23: String concatenation with TEXT cast (real test from test_string_types.py)
- ✓ Test 24: JSON column to TEXT cast (real test from test_json.py)

## Build Integration

The translator is automatically included in the PostgreSQL extension build via CMake's glob pattern in [CMakeLists.pg.cmake](../../CMakeLists.pg.cmake):

```cmake
file(GLOB_RECURSE PG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${PG_MODULE}/*.cpp" ...)
```

No additional build configuration is needed - just rebuild the extension:
```bash
python3 scripts/build_pg_ext.py dev
```

## Debugging

If queries fail after translation, enable logging:

```cpp
#include "pg_to_duckdb_translator.hpp"

std::string query = "...";
std::string translated = pg::pg_to_duckdb_translator::translate(query);

elog(INFO, "Original PostgreSQL query:");
elog(INFO, "%s", query.c_str());
elog(INFO, "Translated DuckDB query:");
elog(INFO, "%s", translated.c_str());
```

## Performance Considerations

- **Translation overhead:** ~0.1-1ms per query (depends on query complexity)
- **Caching:** For queries executed multiple times, consider caching translations
- **Execution impact:** No impact on query execution time (translation happens once before execution)
- **Implementation:** Uses regex-based pattern matching with manual parenthesis balancing for complex expressions
- **Memory:** Minimal allocation - mostly string operations

## Rollback Plan

If you encounter issues:
1. Set `deeplake.enable_query_translation = false` in postgresql.conf
2. Or remove the translation call from your code temporarily
3. Queries will be passed to DuckDB unchanged (may require manual DuckDB syntax)

## Limitations

- Does not handle all PostgreSQL syntax - focused on common JSON query patterns
- Nested CAST expressions beyond 2 levels may need testing
- Comments in SQL queries are not preserved
- Multi-statement queries are translated as a whole (no statement separation)

## Future Enhancements

Possible improvements:
1. Add support for more PostgreSQL-specific functions
2. Handle JSON array indexing: `json->'array'->0`
3. Support JSONB operators
4. Add query validation/linting
5. Performance profiling and optimization
