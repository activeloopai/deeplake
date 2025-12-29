#include "../pg_to_duckdb_translator.hpp"
#include <iostream>
#include <cassert>
#include <string>

namespace {

void test_query(const std::string& name,
                const std::string& pg_query,
                const std::string& expected_duckdb_query) {
    std::cout << "Testing: " << name << std::endl;
    std::cout << "PostgreSQL Query:" << std::endl;
    std::cout << "  " << pg_query << std::endl;

    std::string result = pg::pg_to_duckdb_translator::translate(pg_query);

    std::cout << "Expected DuckDB Query:" << std::endl;
    std::cout << "  " << expected_duckdb_query << std::endl;
    std::cout << "Actual DuckDB Query:" << std::endl;
    std::cout << "  " << result << std::endl;

    if (result == expected_duckdb_query) {
        std::cout << "✓ PASSED" << std::endl;
    } else {
        std::cout << "✗ FAILED" << std::endl;
        std::cout << "Differences found!" << std::endl;
    }
    std::cout << std::endl;
}

} // anonymous namespace

int main() {
    std::cout << "=== PostgreSQL to DuckDB Translator Tests ===" << std::endl << std::endl;

    // Test 1: Simple JSON access with GROUP BY
    test_query(
        "Test 1: Simple JSON access",
        "SELECT json -> 'commit' ->> 'collection' AS event, COUNT(*) as count FROM bluesky GROUP BY event ORDER BY count DESC;",
        "SELECT json->>'$.commit.collection' AS event, count() as count FROM bluesky GROUP BY event ORDER BY count DESC;"
    );

    // Test 2: Multiple JSON access with WHERE clause
    test_query(
        "Test 2: Multiple JSON access with WHERE",
        "SELECT json -> 'commit' ->> 'collection' AS event, COUNT(*) as count, COUNT(DISTINCT json ->> 'did') AS users FROM bluesky WHERE json ->> 'kind' = 'commit' AND json -> 'commit' ->> 'operation' = 'create' GROUP BY event ORDER BY count DESC;",
        "SELECT json->>'$.commit.collection' AS event, count() as count, COUNT(DISTINCT json->>'$.did') AS users FROM bluesky WHERE (json->>'$.kind' = 'commit') AND (json->>'$.commit.operation' = 'create') GROUP BY event ORDER BY count DESC;"
    );

    // Test 3: EXTRACT(HOUR FROM) with TO_TIMESTAMP and IN clause
    test_query(
        "Test 3: EXTRACT HOUR with IN clause",
        "SELECT json->'commit'->>'collection' AS event, EXTRACT(HOUR FROM TO_TIMESTAMP((json->>'time_us')::BIGINT / 1000000)) AS hour_of_day, COUNT(*) AS count FROM bluesky WHERE json->>'kind' = 'commit' AND json->'commit'->>'operation' = 'create' AND json->'commit'->>'collection' IN ('app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like') GROUP BY event, hour_of_day ORDER BY hour_of_day, event;",
        "SELECT json->>'$.commit.collection' AS event, hour(TO_TIMESTAMP(CAST(json->>'$.time_us' AS BIGINT) / 1000000)) AS hour_of_day, count() AS count FROM bluesky WHERE (json->>'$.kind' = 'commit') AND (json->>'$.commit.operation' = 'create') AND (json->>'$.commit.collection' in ['app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like']) GROUP BY event, hour_of_day ORDER BY hour_of_day, event;"
    );

    // Test 4: MIN with TIMESTAMP WITH TIME ZONE 'epoch'
    test_query(
        "Test 4: MIN with TIMESTAMP epoch conversion",
        "SELECT json->>'did' AS user_id, MIN( TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (json->>'time_us')::BIGINT ) AS first_post_ts FROM bluesky WHERE json->>'kind' = 'commit' AND json->'commit'->>'operation' = 'create' AND json->'commit'->>'collection' = 'app.bsky.feed.post' GROUP BY user_id ORDER BY first_post_ts ASC LIMIT 3;",
        "SELECT json->>'$.did' AS user_id, MIN( TO_TIMESTAMP(CAST(json->>'$.time_us' AS BIGINT) / 1000000) ) AS first_post_ts FROM bluesky WHERE (json->>'$.kind' = 'commit') AND (json->>'$.commit.operation' = 'create') AND (json->>'$.commit.collection' = 'app.bsky.feed.post') GROUP BY user_id ORDER BY first_post_ts ASC LIMIT 3;"
    );

    // Test 5: EXTRACT(EPOCH FROM ...) with date difference
    test_query(
        "Test 5: Date difference with EXTRACT EPOCH",
        "SELECT json->>'did' AS user_id, EXTRACT(EPOCH FROM ( MAX( TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (json->>'time_us')::BIGINT ) - MIN( TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (json->>'time_us')::BIGINT ) )) * 1000 AS activity_span FROM bluesky WHERE json->>'kind' = 'commit' AND json->'commit'->>'operation' = 'create' AND json->'commit'->>'collection' = 'app.bsky.feed.post' GROUP BY user_id ORDER BY activity_span DESC LIMIT 3;",
        "SELECT json->>'$.did' AS user_id, date_diff('milliseconds', MIN( TO_TIMESTAMP(CAST(json->>'$.time_us' AS BIGINT) / 1000000) ), MAX( TO_TIMESTAMP(CAST(json->>'$.time_us' AS BIGINT) / 1000000) )) AS activity_span FROM bluesky WHERE (json->>'$.kind' = 'commit') AND (json->>'$.commit.operation' = 'create') AND (json->>'$.commit.collection' = 'app.bsky.feed.post') GROUP BY user_id ORDER BY activity_span DESC LIMIT 3;"
    );

    // Test 6: Simple WHERE with JSON operators
    test_query(
        "Test 6: Simple WHERE with COUNT",
        "SELECT COUNT(*) FROM bluesky WHERE json ->> 'kind' = 'commit' AND json -> 'commit' ->> 'operation' = 'create';",
        "SELECT count() FROM bluesky WHERE (json->>'$.kind' = 'commit') AND (json->>'$.commit.operation' = 'create');"
    );

    // Test 7: WHERE with multiple JSON paths and comparison
    test_query(
        "Test 7: WHERE with nested JSON and string comparison",
        "SELECT * FROM bluesky WHERE json -> 'commit' ->> 'collection' = 'app.bsky.feed.post' AND json ->> 'did' IS NOT NULL;",
        "SELECT * FROM bluesky WHERE (json->>'$.commit.collection' = 'app.bsky.feed.post') AND (json->>'$.did' IS NOT NULL);"
    );

    // Test 8: WHERE with type cast and comparison
    test_query(
        "Test 8: WHERE with type cast",
        "SELECT id, json ->> 'repo' as repo FROM bluesky WHERE (json ->> 'seq')::BIGINT > 1000000;",
        "SELECT id, json->>'$.repo' as repo FROM bluesky WHERE (CAST(json->>'$.seq' AS BIGINT) > 1000000);"
    );

    // Test 9: Complex WHERE with IN clause and multiple JSON operators
    test_query(
        "Test 9: WHERE with IN clause",
        "SELECT COUNT(*) FROM bluesky WHERE json ->> 'kind' = 'commit' AND json -> 'commit' ->> 'collection' IN ('app.bsky.feed.post', 'app.bsky.feed.like') AND json -> 'commit' ->> 'operation' = 'create';",
        "SELECT count() FROM bluesky WHERE (json->>'$.kind' = 'commit') AND (json->>'$.commit.collection' in ['app.bsky.feed.post', 'app.bsky.feed.like']) AND (json->>'$.commit.operation' = 'create');"
    );

    // Test 10: Subquery with WHERE clause (regression test for subquery parenthesis bug)
    test_query(
        "Test 10: Subquery with WHERE clause",
        "SELECT count(*) FROM (SELECT * FROM people WHERE age = 4) AS subquery",
        "SELECT count() FROM (SELECT * FROM people WHERE (age = 4)) AS subquery"
    );

    // Test 11: to_date function conversion
    test_query(
        "Test 11: to_date to strptime conversion",
        "SELECT id, name FROM users WHERE signup_date >= to_date('2024-01-01', 'YYYY-MM-DD')",
        "SELECT id, name FROM users WHERE (signup_date >= strptime('2024-01-01', 'YYYY-MM-DD'))"
    );

    // Test 12: DIV() function conversion
    test_query(
        "Test 12: DIV function to // operator",
        "SELECT id, DIV(total_amount, 100) AS dollars FROM transactions WHERE DIV(quantity, 10) > 5",
        "SELECT id, (total_amount // 100) AS dollars FROM transactions WHERE ((quantity // 10) > 5)"
    );

    // Test 13: regexp_substr conversion
    test_query(
        "Test 13: regexp_substr to regexp_extract",
        "SELECT id, regexp_substr(email, '[^@]+') AS username FROM users WHERE regexp_substr(domain, '[a-z]+') = 'example'",
        "SELECT id, regexp_extract(email, '[^@]+') AS username FROM users WHERE (regexp_extract(domain, '[a-z]+') = 'example')"
    );

    // Test 14: Multiple date functions combined
    test_query(
        "Test 14: Combined date operations",
        "SELECT name, EXTRACT(YEAR FROM to_date(birth_date, 'DD/MM/YYYY')) AS birth_year FROM people WHERE EXTRACT(MONTH FROM signup_date) = 1",
        "SELECT name, year(strptime(birth_date, 'DD/MM/YYYY')) AS birth_year FROM people WHERE (month(signup_date) = 1)"
    );

    // Test 15: Complex arithmetic with DIV
    test_query(
        "Test 15: Arithmetic with DIV and type casts",
        "SELECT id, (revenue - costs)::DECIMAL / 100 AS profit, DIV((quantity)::BIGINT, 12) AS dozens FROM sales WHERE DIV(price, 10) < 100",
        "SELECT id, CAST(revenue - costs AS DECIMAL) / 100 AS profit, (CAST(quantity AS BIGINT) // 12) AS dozens FROM sales WHERE ((price // 10) < 100)"
    );

    // Test 16: String pattern matching with regex
    test_query(
        "Test 16: Pattern matching and string operations",
        "SELECT COUNT(*) FROM logs WHERE regexp_substr(message, 'ERROR|WARN') IS NOT NULL AND status = 'failed'",
        "SELECT count() FROM logs WHERE (regexp_extract(message, 'ERROR|WARN') IS NOT NULL) AND (status = 'failed')"
    );

    // Test 17: Case-insensitive function names
    test_query(
        "Test 17: Case insensitive TO_DATE",
        "SELECT * FROM events WHERE event_date = TO_DATE('15-Mar-2024', 'DD-Mon-YYYY')",
        "SELECT * FROM events WHERE (event_date = strptime('15-Mar-2024', 'DD-Mon-YYYY'))"
    );

    // Test 18: Nested function calls
    test_query(
        "Test 18: Nested functions with type casts",
        "SELECT user_id, EXTRACT(DAY FROM to_date(date_string, 'YYYY-MM-DD')) AS day_of_month FROM user_actions WHERE (user_id)::INTEGER > 1000",
        "SELECT user_id, day(strptime(date_string, 'YYYY-MM-DD')) AS day_of_month FROM user_actions WHERE (CAST(user_id AS INTEGER) > 1000)"
    );

    // Test 19: Complex WHERE with multiple operators
    test_query(
        "Test 19: Complex WHERE with mixed operators",
        "SELECT * FROM products WHERE (price)::DECIMAL > 50.00 AND category IN ('electronics', 'appliances') AND DIV(stock, 10) >= 2",
        "SELECT * FROM products WHERE (CAST(price AS DECIMAL) > 50.00) AND (category in ['electronics', 'appliances']) AND ((stock // 10) >= 2)"
    );

    // Test 20: Multiple regex operations
    test_query(
        "Test 20: Multiple regexp_substr in SELECT and WHERE",
        "SELECT regexp_substr(full_name, '^[^ ]+') AS first_name, regexp_substr(email, '@(.+)$') AS domain FROM users WHERE regexp_substr(phone, '^[0-9]{3}') = '415'",
        "SELECT regexp_extract(full_name, '^[^ ]+') AS first_name, regexp_extract(email, '@(.+)$') AS domain FROM users WHERE (regexp_extract(phone, '^[0-9]{3}') = '415')"
    );

    // Test 21: DOUBLE PRECISION type cast
    test_query(
        "Test 21: DOUBLE PRECISION compound type",
        "SELECT id, (value)::DOUBLE PRECISION AS double_val, amount::DOUBLE PRECISION FROM numbers WHERE (price)::DOUBLE PRECISION > 10.5",
        "SELECT id, CAST(value AS DOUBLE PRECISION) AS double_val, CAST(amount AS DOUBLE PRECISION) FROM numbers WHERE (CAST(price AS DOUBLE PRECISION) > 10.5)"
    );

    // Test 22: VARCHAR and TEXT type casts (from real PostgreSQL test: test_array.py)
    test_query(
        "Test 22: VARCHAR and TEXT type casts",
        "SELECT * FROM array_test WHERE text_array_1d[1] = 'text1'::text AND varchar_array_1d[2] = 'varchar2'::varchar",
        "SELECT * FROM array_test WHERE (text_array_1d[1] = CAST('text1' AS text)) AND (varchar_array_1d[2] = CAST('varchar2' AS varchar))"
    );

    // Test 23: String concatenation with TEXT casts (from real PostgreSQL test: test_string_types.py)
    test_query(
        "Test 23: String concatenation with TEXT cast",
        "SELECT 'bpchar_' || LPAD(i::text, 4, '0') AS bpchar_val FROM generate_series(1, 10) i WHERE (i)::text = '5'",
        "SELECT 'bpchar_' || LPAD(CAST(i AS text), 4, '0') AS bpchar_val FROM generate_series(1, 10) i WHERE (CAST(i AS text) = '5')"
    );

    // Test 24: JSON to TEXT cast (from real PostgreSQL test: test_json.py)
    test_query(
        "Test 24: JSON column to TEXT cast",
        "SELECT * FROM json_test WHERE json_col::text = '{\"name\": \"Alice\", \"age\": 30}'",
        "SELECT * FROM json_test WHERE (CAST(json_col AS text) = '{\"name\": \"Alice\", \"age\": 30}')"
    );

    std::cout << "=== All tests completed ===" << std::endl;

    return 0;
}
