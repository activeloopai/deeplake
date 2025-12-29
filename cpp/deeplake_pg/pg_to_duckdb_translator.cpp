#include "pg_to_duckdb_translator.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace pg {

namespace {

// Helper to check if a position is within single quotes
bool is_within_quotes(const std::string& str, size_t pos) {
    size_t quote_count = 0;
    for (size_t i = 0; i < pos; ++i) {
        if (str[i] == '\'') {
            quote_count++;
        }
    }
    return (quote_count % 2) == 1;
}

// Helper to trim whitespace
std::string trim(const std::string& str) {
    auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char ch) {
        return std::isspace(ch);
    });
    auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char ch) {
        return std::isspace(ch);
    }).base();
    return (start < end) ? std::string(start, end) : std::string();
}

// Helper to check if string starts with a prefix (case insensitive)
bool starts_with_ci(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) return false;
    return std::equal(prefix.begin(), prefix.end(), str.begin(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

} // anonymous namespace

std::string pg_to_duckdb_translator::translate(const std::string& pg_query) {
    std::string result = pg_query;

    // Order matters! Process in this sequence:
    // 1. Translate JSON operators FIRST (before type casts)
    result = translate_json_operators(result);

    // 2. Translate type casts (::TYPE to CAST)
    result = translate_type_casts(result);

    // 3. Handle complex timestamp conversions
    result = translate_timestamp_functions(result);

    // 4. Handle date_diff (EXTRACT(EPOCH FROM ...))
    result = translate_date_diff(result);

    // 5. Handle other EXTRACT functions
    result = translate_extract_functions(result);

    // 6. Handle IN clauses
    result = translate_in_clauses(result);

    // 7. Optional: translate COUNT(*) to count()
    result = translate_count_star(result);

    // 8. Translate to_date to strptime
    result = translate_to_date(result);

    // 9. Translate DIV() function
    result = translate_div_function(result);

    // 10. Translate regexp_substr to regexp_extract
    result = translate_regexp_substr(result);

    // 11. Wrap WHERE clause predicates in parentheses (must be last!)
    result = wrap_where_predicates(result);

    return result;
}

std::string pg_to_duckdb_translator::translate_json_operators(const std::string& query) {
    std::string result = query;

    // Pattern: identifier followed by one or more (-> 'key') or (->> 'key')
    // Must handle spaces: json -> 'commit' -> 'collection'
    // The key is to match -> or ->> followed by optional spaces and a quoted string
    std::regex json_access_pattern(
        R"((\w+)(\s*(?:->|->>\s*)\s*'[^']+')+)"
    );

    std::smatch match;
    std::string temp = result;
    result.clear();

    size_t last_pos = 0;
    auto search_start = temp.cbegin();

    while (std::regex_search(search_start, temp.cend(), match, json_access_pattern)) {
        size_t match_pos = std::distance(temp.cbegin(), match[0].first);

        // Copy everything before the match
        result.append(temp.substr(last_pos, match_pos - last_pos));

        // Extract column name and the FULL chain of operators
        // Note: match[2] only captures the last repetition, so we need to use the full match
        std::string column_name = match[1].str();
        std::string full_match = match[0].str();
        std::string operators_chain = full_match.substr(column_name.length());

        // Parse the operators chain to extract keys
        // Updated pattern to handle spaces properly
        std::vector<std::string> keys;
        std::regex key_pattern(R"(->>?\s*'([^']+)')");
        auto keys_begin = std::sregex_iterator(operators_chain.begin(), operators_chain.end(), key_pattern);
        auto keys_end = std::sregex_iterator();

        for (std::sregex_iterator i = keys_begin; i != keys_end; ++i) {
            keys.push_back((*i)[1].str());
        }

        // Build DuckDB JSON path: column->>'$.key1.key2.key3'
        if (!keys.empty()) {
            result.append(column_name);
            result.append("->>'");
            result.append(build_json_path(keys));
            result.append("'");
        } else {
            // Shouldn't happen, but fallback to original
            result.append(match[0].str());
        }

        // Update position
        last_pos = match_pos + match[0].length();
        search_start = match[0].second;
    }

    // Append remaining text
    result.append(temp.substr(last_pos));

    return result;
}

std::string pg_to_duckdb_translator::translate_type_casts(const std::string& query) {
    std::string result = query;

    // Pattern: (expression)::TYPE or identifier::TYPE
    // Convert to: CAST(expression AS TYPE)
    // But skip if already inside a CAST() - to avoid double conversion

    // First, handle simple JSON path casts (must be before complex patterns)
    // json->>'key'::BIGINT => CAST(json->>'key' AS BIGINT)
    // Note: DOUBLE PRECISION is handled as a compound type in parenthesized casts
    std::regex json_cast_pattern(R"((\w+->>'[^']+')::(BIGINT|INTEGER|VARCHAR|TEXT|DOUBLE PRECISION|DOUBLE|FLOAT|TIMESTAMP|DATE|BOOLEAN|DECIMAL)\b)",
                                std::regex::icase);
    result = std::regex_replace(result, json_cast_pattern, "CAST($1 AS $2)");

    // Handle parenthesized expression casts with manual parenthesis balancing
    // (expr)::BIGINT => CAST(expr AS BIGINT)
    std::regex paren_cast_marker(R"(\(.*?\)::(BIGINT|INTEGER|VARCHAR|TEXT|DOUBLE|FLOAT|TIMESTAMP|DATE|BOOLEAN|DECIMAL|String)\b)",
                                 std::regex::icase);

    std::smatch match;
    std::string temp = result;
    result.clear();
    size_t last_pos = 0;

    // Process each potential cast
    size_t pos = 0;
    while (pos < temp.length()) {
        // Look for opening parenthesis
        if (temp[pos] == '(') {
            // Find matching closing parenthesis
            size_t start = pos;
            int depth = 0;
            bool in_quotes = false;
            size_t close_pos = pos;

            for (size_t i = pos; i < temp.length(); ++i) {
                char c = temp[i];
                if (c == '\'') {
                    in_quotes = !in_quotes;
                } else if (!in_quotes) {
                    if (c == '(') {
                        depth++;
                    } else if (c == ')') {
                        depth--;
                        if (depth == 0) {
                            close_pos = i;
                            break;
                        }
                    }
                }
            }

            if (close_pos > start && close_pos + 2 < temp.length() && temp[close_pos + 1] == ':' && temp[close_pos + 2] == ':') {
                // Found a cast! Extract the type
                size_t type_start = close_pos + 3;
                size_t type_end = type_start;

                // Skip leading spaces
                while (type_start < temp.length() && std::isspace(temp[type_start])) {
                    type_start++;
                }
                type_end = type_start;

                // Extract first word of type
                while (type_end < temp.length() && (std::isalnum(temp[type_end]) || temp[type_end] == '_')) {
                    type_end++;
                }

                std::string type_str = temp.substr(type_start, type_end - type_start);

                // Check for compound types: if type is DOUBLE, check if followed by PRECISION
                if (type_end < temp.length()) {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    if (type_upper == "DOUBLE") {
                        // Skip spaces
                        size_t next_word_start = type_end;
                        while (next_word_start < temp.length() && std::isspace(temp[next_word_start])) {
                            next_word_start++;
                        }

                        // Check if next word is PRECISION
                        if (next_word_start + 9 <= temp.length()) {
                            std::string next_word = temp.substr(next_word_start, 9);
                            std::transform(next_word.begin(), next_word.end(), next_word.begin(), ::toupper);

                            if (next_word == "PRECISION") {
                                type_str = "DOUBLE PRECISION";
                                type_end = next_word_start + 9;
                            }
                        }
                    }
                }

                // Check if it's a valid type (case insensitive)
                std::string type_upper = type_str;
                std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                // Check for compound types first
                bool is_valid_type = false;
                if (type_upper == "DOUBLE PRECISION") {
                    is_valid_type = true;
                } else if (type_upper == "BIGINT" || type_upper == "INTEGER" || type_upper == "VARCHAR" ||
                           type_upper == "TEXT" || type_upper == "DOUBLE" || type_upper == "FLOAT" ||
                           type_upper == "TIMESTAMP" || type_upper == "DATE" || type_upper == "BOOLEAN" ||
                           type_upper == "DECIMAL") {
                    is_valid_type = true;
                }

                if (is_valid_type) {

                    // Copy everything before this cast
                    result.append(temp.substr(last_pos, start - last_pos));

                    // Extract the expression inside parentheses
                    std::string expr = temp.substr(start + 1, close_pos - start - 1);

                    // Build CAST
                    result.append("CAST(");
                    result.append(expr);
                    result.append(" AS ");
                    result.append(type_str);
                    result.append(")");

                    last_pos = type_end;
                    pos = type_end;
                    continue;
                }
            }
        }
        pos++;
    }

    result.append(temp.substr(last_pos));

    // Also handle simple identifier casts
    // Note: For DOUBLE PRECISION with identifiers, match the compound type first
    std::regex simple_cast_double_prec(R"((\w+)::(DOUBLE\s+PRECISION)\b)", std::regex::icase);
    result = std::regex_replace(result, simple_cast_double_prec, "CAST($1 AS $2)");

    // Handle string literal casts: 'string'::TYPE
    // Pattern matches: 'text'::varchar or 'some''escaped''text'::text
    std::regex string_literal_cast(R"(('(?:[^']|'')*')::(BIGINT|INTEGER|VARCHAR|TEXT|DOUBLE|FLOAT|TIMESTAMP|DATE|BOOLEAN|DECIMAL)\b)",
                                   std::regex::icase);
    result = std::regex_replace(result, string_literal_cast, "CAST($1 AS $2)");

    std::regex simple_cast_pattern(R"((\w+)::(BIGINT|INTEGER|VARCHAR|TEXT|DOUBLE|FLOAT|TIMESTAMP|DATE|BOOLEAN|DECIMAL)\b)",
                                   std::regex::icase);
    result = std::regex_replace(result, simple_cast_pattern, "CAST($1 AS $2)");

    return result;
}

std::string pg_to_duckdb_translator::translate_timestamp_functions(const std::string& query) {
    std::string result = query;

    // Pattern: Find TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL...
    // and manually extract the following expression with balanced parentheses
    std::regex epoch_start_pattern(
        R"(TIMESTAMP\s+WITH\s+TIME\s+ZONE\s+'epoch'\s*\+\s*INTERVAL\s+'1\s+microsecond'\s*\*\s*)",
        std::regex::icase
    );

    std::smatch match;
    std::string temp = result;
    result.clear();

    size_t last_pos = 0;
    auto search_start = temp.cbegin();

    while (std::regex_search(search_start, temp.cend(), match, epoch_start_pattern)) {
        size_t match_pos = std::distance(temp.cbegin(), match[0].first);
        size_t expr_start = match_pos + match[0].length();

        // Copy everything before the match
        result.append(temp.substr(last_pos, match_pos - last_pos));

        // Extract expression with balanced parentheses
        size_t expr_end = expr_start;
        int paren_depth = 0;
        bool in_quotes = false;

        for (size_t i = expr_start; i < temp.length(); ++i) {
            char c = temp[i];
            if (c == '\'') {
                in_quotes = !in_quotes;
            } else if (!in_quotes) {
                if (c == '(') {
                    paren_depth++;
                } else if (c == ')') {
                    if (paren_depth == 0) {
                        expr_end = i;
                        break;
                    }
                    paren_depth--;
                }
            }
        }

        std::string expr = trim(temp.substr(expr_start, expr_end - expr_start));

        // If it's already a CAST(...), extract the inner expression
        // Use non-greedy match for the content
        std::regex cast_wrapper(R"(^CAST\s*\((.+?)\s+AS\s+BIGINT\)$)", std::regex::icase);
        std::smatch cast_match;
        if (std::regex_match(expr, cast_match, cast_wrapper)) {
            expr = trim(cast_match[1].str());
        }

        // Build the DuckDB equivalent
        result.append("TO_TIMESTAMP(CAST(");
        result.append(expr);
        result.append(" AS BIGINT) / 1000000) ");

        // Update position
        last_pos = expr_end;
        search_start = temp.cbegin() + expr_end;
    }

    // Append remaining text
    result.append(temp.substr(last_pos));

    return result;
}

std::string pg_to_duckdb_translator::translate_extract_functions(const std::string& query) {
    std::string result = query;

    // Pattern: EXTRACT(field FROM expr)
    // Common fields: HOUR, DAY, MONTH, YEAR, MINUTE, SECOND
    // Convert to: field(expr)

    std::regex extract_pattern(
        R"(EXTRACT\s*\(\s*(HOUR|DAY|MONTH|YEAR|MINUTE|SECOND|DOW|DOY|WEEK)\s+FROM\s+([^)]+)\))",
        std::regex::icase
    );

    std::smatch match;
    std::string temp = result;
    result.clear();

    size_t last_pos = 0;
    auto search_start = temp.cbegin();

    while (std::regex_search(search_start, temp.cend(), match, extract_pattern)) {
        size_t match_pos = std::distance(temp.cbegin(), match[0].first);

        // Copy everything before the match
        result.append(temp.substr(last_pos, match_pos - last_pos));

        // Extract field and expression
        std::string field = match[1].str();
        std::string expr = match[2].str();

        // Convert field name to lowercase for DuckDB function
        std::transform(field.begin(), field.end(), field.begin(), ::tolower);

        // Build DuckDB function call
        result.append(field);
        result.append("(");
        result.append(expr);
        result.append(")");

        // Update position
        last_pos = match_pos + match[0].length();
        search_start = match[0].second;
    }

    // Append remaining text
    result.append(temp.substr(last_pos));

    return result;
}

std::string pg_to_duckdb_translator::translate_date_diff(const std::string& query) {
    std::string result = query;

    // Pattern: EXTRACT(EPOCH FROM ( ... - ... )) * multiplier
    // Need to handle nested parentheses manually
    // Common multipliers: 1 => seconds, 1000 => milliseconds, 1000000 => microseconds

    std::regex extract_start_pattern(
        R"(EXTRACT\s*\(\s*EPOCH\s+FROM\s*\(\s*)",
        std::regex::icase
    );

    std::smatch match;
    std::string temp = result;
    result.clear();

    size_t last_pos = 0;
    auto search_start = temp.cbegin();

    while (std::regex_search(search_start, temp.cend(), match, extract_start_pattern)) {
        size_t match_pos = std::distance(temp.cbegin(), match[0].first);
        size_t expr_start = match_pos + match[0].length();

        // Manually find the matching close parens and extract expr1 - expr2
        int paren_depth = 0;
        bool in_quotes = false;
        size_t expr_end = expr_start;
        size_t minus_pos = std::string::npos;

        for (size_t i = expr_start; i < temp.length(); ++i) {
            char c = temp[i];
            if (c == '\'') {
                in_quotes = !in_quotes;
            } else if (!in_quotes) {
                if (c == '(') {
                    paren_depth++;
                } else if (c == ')') {
                    if (paren_depth == 0) {
                        expr_end = i;
                        break;
                    }
                    paren_depth--;
                } else if (c == '-' && paren_depth == 0 && minus_pos == std::string::npos) {
                    // Found the top-level minus
                    minus_pos = i;
                }
            }
        }

        // Check if this is a date diff pattern (has - and * multiplier after)
        if (minus_pos == std::string::npos || expr_end == expr_start) {
            // Not a date diff, skip
            result.append(temp.substr(last_pos, match_pos + match[0].length() - last_pos));
            last_pos = match_pos + match[0].length();
            search_start = temp.cbegin() + last_pos;
            continue;
        }

        // Look for )) * number after expr_end
        std::regex multiplier_pattern(R"(\s*\)\s*\)\s*\*\s*(\d+))");
        std::smatch mult_match;
        std::string remaining = temp.substr(expr_end);
        if (!std::regex_search(remaining, mult_match, multiplier_pattern)) {
            // Not a date diff pattern
            result.append(temp.substr(last_pos, match_pos + match[0].length() - last_pos));
            last_pos = match_pos + match[0].length();
            search_start = temp.cbegin() + last_pos;
            continue;
        }

        // Extract expressions
        std::string expr1 = trim(temp.substr(expr_start, minus_pos - expr_start));
        std::string expr2 = trim(temp.substr(minus_pos + 1, expr_end - minus_pos - 1));
        std::string multiplier = mult_match[1].str();

        // Copy everything before the match
        result.append(temp.substr(last_pos, match_pos - last_pos));

        // Determine unit based on multiplier
        std::string unit;
        if (multiplier == "1000") {
            unit = "milliseconds";
        } else if (multiplier == "1000000") {
            unit = "microseconds";
        } else if (multiplier == "1") {
            unit = "seconds";
        } else {
            unit = "seconds"; // default
        }

        // Build date_diff call
        result.append("date_diff('");
        result.append(unit);
        result.append("', ");
        result.append(expr2);
        result.append(", ");
        result.append(expr1);
        result.append(")");

        // Update position - skip past the entire EXTRACT(...)) * number
        last_pos = expr_end + mult_match[0].length();
        search_start = temp.cbegin() + last_pos;
    }

    // Append remaining text
    result.append(temp.substr(last_pos));

    return result;
}

std::string pg_to_duckdb_translator::translate_in_clauses(const std::string& query) {
    std::string result = query;

    // Pattern: expr IN ('val1', 'val2', 'val3')
    // Convert to: expr in ['val1', 'val2', 'val3']

    std::regex in_clause_pattern(
        R"(\bIN\s*\(\s*('[^']+'\s*(?:,\s*'[^']+')*)\s*\))",
        std::regex::icase
    );

    std::smatch match;
    std::string temp = result;
    result.clear();

    size_t last_pos = 0;
    auto search_start = temp.cbegin();

    while (std::regex_search(search_start, temp.cend(), match, in_clause_pattern)) {
        size_t match_pos = std::distance(temp.cbegin(), match[0].first);

        // Copy everything before the match
        result.append(temp.substr(last_pos, match_pos - last_pos));

        // Extract values list
        std::string values_list = match[1].str();

        // Build DuckDB IN clause with square brackets
        result.append("in [");
        result.append(values_list);
        result.append("]");

        // Update position
        last_pos = match_pos + match[0].length();
        search_start = match[0].second;
    }

    // Append remaining text
    result.append(temp.substr(last_pos));

    return result;
}

std::string pg_to_duckdb_translator::translate_count_star(const std::string& query) {
    // Optional: Convert COUNT(*) to count()
    // DuckDB supports both, so this is optional
    std::regex count_star_pattern(R"(\bCOUNT\s*\(\s*\*\s*\))", std::regex::icase);
    return std::regex_replace(query, count_star_pattern, "count()");
}

std::string pg_to_duckdb_translator::wrap_where_predicates(const std::string& query) {
    // Find WHERE clause and wrap each predicate (separated by AND/OR) in parentheses
    // Example: WHERE a = 1 AND b = 2 => WHERE (a = 1) AND (b = 2)

    std::regex where_pattern(R"(\bWHERE\s+)", std::regex::icase);
    std::smatch match;

    if (!std::regex_search(query, match, where_pattern)) {
        return query; // No WHERE clause
    }

    size_t where_start = match.position() + match.length();
    std::string before_where = query.substr(0, where_start);
    std::string after_where = query.substr(where_start);

    // Find the end of the WHERE clause by manually tracking parentheses depth
    // We need to handle:
    // 1. Subquery closing parentheses: WHERE x = 1) AS subquery
    // 2. Keywords: GROUP BY, ORDER BY, LIMIT, etc.
    // 3. Semicolons

    size_t where_end = std::string::npos;
    int paren_depth = 0;
    bool in_quotes = false;

    for (size_t i = 0; i < after_where.length(); ++i) {
        char c = after_where[i];

        if (c == '\'') {
            in_quotes = !in_quotes;
        } else if (!in_quotes) {
            if (c == '(') {
                paren_depth++;
            } else if (c == ')') {
                if (paren_depth == 0) {
                    // Closing paren at depth 0 means we're at the end of a subquery
                    where_end = i;
                    break;
                }
                paren_depth--;
            } else if (paren_depth == 0) {
                // Check for keywords or semicolon at top level
                if (c == ';') {
                    where_end = i;
                    break;
                }

                // Check for SQL keywords
                std::string remaining = after_where.substr(i);
                std::regex keyword_pattern(R"(^\s*\b(GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|HAVING|UNION|INTERSECT|EXCEPT)\b)",
                                          std::regex::icase);
                std::smatch keyword_match;
                if (std::regex_search(remaining, keyword_match, keyword_pattern)) {
                    where_end = i;
                    break;
                }
            }
        }
    }

    std::string where_clause;
    std::string after_clause;

    if (where_end != std::string::npos) {
        where_clause = after_where.substr(0, where_end);
        after_clause = after_where.substr(where_end);
    } else {
        where_clause = after_where;
        after_clause = "";
    }

    // Trim trailing whitespace from where_clause
    size_t end_trim = where_clause.find_last_not_of(" \t\n\r");
    if (end_trim != std::string::npos) {
        where_clause = where_clause.substr(0, end_trim + 1);
    }

    // Now split the WHERE clause by AND/OR (but not within parentheses or quotes)
    // and wrap each predicate in parentheses if not already wrapped

    std::string result = before_where;
    std::vector<std::string> predicates;
    std::vector<std::string> operators; // AND or OR

    size_t pos = 0;
    size_t start = 0;
    // Reuse paren_depth and in_quotes variables from above, reset them
    paren_depth = 0;
    in_quotes = false;

    // Parse the WHERE clause
    while (pos < where_clause.length()) {
        char c = where_clause[pos];

        if (c == '\'') {
            in_quotes = !in_quotes;
        } else if (!in_quotes) {
            if (c == '(') {
                paren_depth++;
            } else if (c == ')') {
                paren_depth--;
            } else if (paren_depth == 0) {
                // Check for AND or OR at top level
                if (pos + 3 <= where_clause.length()) {
                    std::string next_3 = where_clause.substr(pos, 3);
                    std::string next_3_upper = next_3;
                    for (char& ch : next_3_upper) ch = std::toupper(ch);

                    bool is_and = (next_3_upper == "AND" &&
                                  (pos + 3 >= where_clause.length() || std::isspace(where_clause[pos + 3])));
                    bool is_or = (pos + 2 < where_clause.length() &&
                                 next_3_upper.substr(0, 2) == "OR" &&
                                 (pos + 2 >= where_clause.length() || std::isspace(where_clause[pos + 2])));

                    if (is_and) {
                        predicates.push_back(where_clause.substr(start, pos - start));
                        operators.push_back("AND");
                        pos += 3;
                        start = pos;
                        continue;
                    } else if (is_or) {
                        predicates.push_back(where_clause.substr(start, pos - start));
                        operators.push_back("OR");
                        pos += 2;
                        start = pos;
                        continue;
                    }
                }
            }
        }
        pos++;
    }

    // Add the last predicate
    predicates.push_back(where_clause.substr(start));

    // Now wrap each predicate in parentheses if needed
    for (size_t i = 0; i < predicates.size(); ++i) {
        std::string pred = predicates[i];

        // Trim whitespace
        size_t first = pred.find_first_not_of(" \t\n\r");
        size_t last = pred.find_last_not_of(" \t\n\r");
        if (first == std::string::npos) {
            continue; // Empty predicate
        }
        pred = pred.substr(first, last - first + 1);

        // Check if already wrapped in parentheses
        bool already_wrapped = false;
        if (pred.length() > 0 && pred[0] == '(' && pred[pred.length() - 1] == ')') {
            // Verify these parens match
            int depth = 0;
            bool valid = true;
            for (size_t j = 0; j < pred.length() - 1; ++j) {
                if (pred[j] == '(') depth++;
                else if (pred[j] == ')') depth--;
                if (depth == 0) {
                    valid = false;
                    break;
                }
            }
            already_wrapped = valid && depth == 1;
        }

        if (!already_wrapped) {
            result += "(";
            result += pred;
            result += ")";
        } else {
            result += pred;
        }

        if (i < operators.size()) {
            result += " ";
            result += operators[i];
            result += " ";
        }
    }

    // Add space before next clause if it's a keyword (not a closing paren or semicolon)
    if (!after_clause.empty() && after_clause[0] != ')' && after_clause[0] != ';') {
        // Trim leading whitespace from after_clause
        size_t first_non_space = after_clause.find_first_not_of(" \t\n\r");
        if (first_non_space != std::string::npos) {
            after_clause = after_clause.substr(first_non_space);
        }
        // Add single space before keyword
        result += " ";
    }
    result += after_clause;

    return result;
}

bool pg_to_duckdb_translator::is_within_quotes(const std::string& str, size_t pos) {
    return ::pg::is_within_quotes(str, pos);
}

std::string pg_to_duckdb_translator::build_json_path(const std::vector<std::string>& keys) {
    std::ostringstream oss;
    oss << "$.";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ".";
        oss << keys[i];
    }
    return oss.str();
}

std::string pg_to_duckdb_translator::translate_to_date(const std::string& query) {
    // PostgreSQL: to_date(string, format)
    // DuckDB: strptime(string, format)
    std::regex to_date_pattern(
        R"(\bto_date\s*\()",
        std::regex::icase
    );

    return std::regex_replace(query, to_date_pattern, "strptime(");
}

std::string pg_to_duckdb_translator::translate_div_function(const std::string& query) {
    // PostgreSQL: DIV(a, b)
    // DuckDB: (a // b)
    std::regex div_pattern(
        R"(\bDIV\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\))",
        std::regex::icase
    );

    std::string result = query;
    std::smatch match;
    std::string temp = result;
    result.clear();

    size_t last_pos = 0;
    auto search_start = temp.cbegin();

    while (std::regex_search(search_start, temp.cend(), match, div_pattern)) {
        size_t match_pos = std::distance(temp.cbegin(), match[0].first);

        // Copy everything before the match
        result.append(temp.substr(last_pos, match_pos - last_pos));

        // Replace DIV(a, b) with (a // b)
        result.append("(");
        result.append(match[1].str());
        result.append(" // ");
        result.append(match[2].str());
        result.append(")");

        last_pos = match_pos + match[0].length();
        search_start = match[0].second;
    }

    result.append(temp.substr(last_pos));
    return result;
}

std::string pg_to_duckdb_translator::translate_regexp_substr(const std::string& query) {
    // PostgreSQL: regexp_substr(string, pattern)
    // DuckDB: regexp_extract(string, pattern)
    std::regex regexp_substr_pattern(
        R"(\bregexp_substr\s*\()",
        std::regex::icase
    );

    return std::regex_replace(query, regexp_substr_pattern, "regexp_extract(");
}

} // namespace pg
