#pragma once

#include <string>
#include <string_view>
#include <regex>
#include <vector>

namespace pg {

/**
 * Translates PostgreSQL JSON queries to DuckDB JSON queries.
 *
 * Key translation rules:
 * 1. JSON operators:
 *    - json -> 'key'      => json->>'$.key'
 *    - json ->> 'key'     => json->>'$.key'
 *    - json -> 'a' -> 'b' => json->>'$.a.b'
 *
 * 2. Type casts:
 *    - (expr)::TYPE       => CAST(expr AS TYPE)
 *
 * 3. Timestamp conversions:
 *    - TO_TIMESTAMP((json->>'key')::BIGINT / divisor)
 *      => TO_TIMESTAMP(CAST(json->>'$.key' AS BIGINT) / divisor)
 *    - TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (json->>'key')::BIGINT
 *      => TO_TIMESTAMP(CAST(json->>'$.key' AS BIGINT) / 1000000)
 *
 * 4. Date extraction:
 *    - EXTRACT(HOUR FROM expr)      => hour(expr)
 *    - EXTRACT(DAY FROM expr)       => day(expr)
 *    - EXTRACT(MONTH FROM expr)     => month(expr)
 *    - EXTRACT(YEAR FROM expr)      => year(expr)
 *    - EXTRACT(EPOCH FROM expr)     => epoch(expr)
 *
 * 5. Date differences:
 *    - EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) * 1000
 *      => date_diff('milliseconds', MIN(ts), MAX(ts))
 *
 * 6. Aggregate functions:
 *    - COUNT(*)           => count() or COUNT(*)
 *
 * 7. IN clauses:
 *    - IN ('a', 'b', 'c') => in ['a', 'b', 'c']
 *
 * 8. WHERE clause predicates:
 *    - WHERE a = 1 AND b = 2 => WHERE (a = 1) AND (b = 2)
 *    - Each predicate is wrapped in parentheses
 *
 * 9. Date formatting functions:
 *    - to_date(string, format) => strptime(string, format)
 *
 * 10. Integer division:
 *    - DIV(a, b) => (a // b)
 *
 * 11. Regular expressions:
 *    - regexp_substr(string, pattern) => regexp_extract(string, pattern)
 */
class pg_to_duckdb_translator {
public:
    /**
     * Translates a PostgreSQL query to DuckDB syntax.
     * @param pg_query The PostgreSQL query string
     * @return The translated DuckDB query string
     */
    static std::string translate(const std::string& pg_query);

private:
    // Individual translation steps
    static std::string translate_json_operators(const std::string& query);
    static std::string translate_type_casts(const std::string& query);
    static std::string translate_timestamp_functions(const std::string& query);
    static std::string translate_extract_functions(const std::string& query);
    static std::string translate_date_diff(const std::string& query);
    static std::string translate_in_clauses(const std::string& query);
    static std::string translate_count_star(const std::string& query);
    static std::string translate_to_date(const std::string& query);
    static std::string translate_div_function(const std::string& query);
    static std::string translate_regexp_substr(const std::string& query);
    static std::string wrap_where_predicates(const std::string& query);

    // Helper functions
    static std::string build_json_path(const std::vector<std::string>& keys);
    static bool is_within_quotes(const std::string& str, size_t pos);
};

} // namespace pg
