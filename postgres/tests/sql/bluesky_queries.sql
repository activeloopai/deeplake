-- ==========================================
-- BLUESKY JSONB INDEX TEST SUITE
-- ==========================================

-- Setup
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
SET pg_deeplake.use_deeplake_executor = false;
SET pg_deeplake.support_json_index = true;

-- Create table
DROP TABLE IF EXISTS bluesky CASCADE;
CREATE TABLE bluesky (
    id SERIAL PRIMARY KEY,
    data JSONB
) USING deeplake;

-- Insert realistic Bluesky data
INSERT INTO bluesky (data) VALUES
-- Posts
('{"kind": "commit", "did": "did:plc:user1", "time_us": 1704067200000000, "commit": {"operation": "create", "collection": "app.bsky.feed.post", "rkey": "post1"}}'),
('{"kind": "commit", "did": "did:plc:user2", "time_us": 1704070800000000, "commit": {"operation": "create", "collection": "app.bsky.feed.post", "rkey": "post2"}}'),
('{"kind": "commit", "did": "did:plc:user1", "time_us": 1704074400000000, "commit": {"operation": "create", "collection": "app.bsky.feed.post", "rkey": "post3"}}'),
('{"kind": "commit", "did": "did:plc:user3", "time_us": 1704078000000000, "commit": {"operation": "create", "collection": "app.bsky.feed.post", "rkey": "post4"}}'),
-- Reposts
('{"kind": "commit", "did": "did:plc:user2", "time_us": 1704081600000000, "commit": {"operation": "create", "collection": "app.bsky.feed.repost", "rkey": "repost1"}}'),
('{"kind": "commit", "did": "did:plc:user3", "time_us": 1704085200000000, "commit": {"operation": "create", "collection": "app.bsky.feed.repost", "rkey": "repost2"}}'),
-- Likes
('{"kind": "commit", "did": "did:plc:user1", "time_us": 1704088800000000, "commit": {"operation": "create", "collection": "app.bsky.feed.like", "rkey": "like1"}}'),
('{"kind": "commit", "did": "did:plc:user2", "time_us": 1704092400000000, "commit": {"operation": "create", "collection": "app.bsky.feed.like", "rkey": "like2"}}'),
('{"kind": "commit", "did": "did:plc:user3", "time_us": 1704096000000000, "commit": {"operation": "create", "collection": "app.bsky.feed.like", "rkey": "like3"}}'),
('{"kind": "commit", "did": "did:plc:user1", "time_us": 1704099600000000, "commit": {"operation": "create", "collection": "app.bsky.feed.like", "rkey": "like4"}}'),
-- Follows
('{"kind": "commit", "did": "did:plc:user2", "time_us": 1704103200000000, "commit": {"operation": "create", "collection": "app.bsky.graph.follow", "rkey": "follow1"}}'),
('{"kind": "commit", "did": "did:plc:user3", "time_us": 1704106800000000, "commit": {"operation": "create", "collection": "app.bsky.graph.follow", "rkey": "follow2"}}'),
-- Updates and deletes
('{"kind": "commit", "did": "did:plc:user1", "time_us": 1704110400000000, "commit": {"operation": "update", "collection": "app.bsky.feed.post", "rkey": "post1"}}'),
('{"kind": "commit", "did": "did:plc:user2", "time_us": 1704114000000000, "commit": {"operation": "delete", "collection": "app.bsky.feed.post", "rkey": "post2"}}'),
-- Identity events
('{"kind": "identity", "did": "did:plc:user1", "time_us": 1704117600000000}'),
('{"kind": "identity", "did": "did:plc:user2", "time_us": 1704121200000000}'),
-- Account events
('{"kind": "account", "did": "did:plc:user3", "time_us": 1704124800000000, "active": true}'),
('{"kind": "account", "did": "did:plc:user1", "time_us": 1704128400000000, "active": false}');

\echo '\n========================================='
\echo 'Data loaded. Creating index...'
\echo '========================================='

-- Create JSONB index
CREATE INDEX idx_bluesky_data ON bluesky USING deeplake_index(data);

\echo '\n========================================='
\echo 'QUERY 1: Event counts by collection'
\echo '========================================='
EXPLAIN SELECT
    data -> 'commit' ->> 'collection' AS event,
    COUNT(*) as count
FROM bluesky
GROUP BY event
ORDER BY count DESC;

SELECT
    data -> 'commit' ->> 'collection' AS event,
    COUNT(*) as count
FROM bluesky
GROUP BY event
ORDER BY count DESC;

\echo '\n========================================='
\echo 'QUERY 2: Events with user counts (filtered)'
\echo '========================================='
EXPLAIN SELECT
    data -> 'commit' ->> 'collection' AS event,
    COUNT(*) as count,
    COUNT(DISTINCT data ->> 'did') AS users
FROM bluesky
WHERE data ->> 'kind' = 'commit'
    AND data -> 'commit' ->> 'operation' = 'create'
GROUP BY event
ORDER BY count DESC;

SELECT
    data -> 'commit' ->> 'collection' AS event,
    COUNT(*) as count,
    COUNT(DISTINCT data ->> 'did') AS users
FROM bluesky
WHERE data ->> 'kind' = 'commit'
    AND data -> 'commit' ->> 'operation' = 'create'
GROUP BY event
ORDER BY count DESC;

\echo '\n========================================='
\echo 'QUERY 3: Activity by hour and event type'
\echo '========================================='
EXPLAIN SELECT
    data->'commit'->>'collection' AS event,
    EXTRACT(HOUR FROM TO_TIMESTAMP((data->>'time_us')::BIGINT / 1000000)) AS hour_of_day,
    COUNT(*) AS count
FROM bluesky
WHERE data->>'kind' = 'commit'
    AND data->'commit'->>'operation' = 'create'
    AND data->'commit'->>'collection' IN ('app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like')
GROUP BY event, hour_of_day
ORDER BY hour_of_day, event;

SELECT
    data->'commit'->>'collection' AS event,
    EXTRACT(HOUR FROM TO_TIMESTAMP((data->>'time_us')::BIGINT / 1000000)) AS hour_of_day,
    COUNT(*) AS count
FROM bluesky
WHERE data->>'kind' = 'commit'
    AND data->'commit'->>'operation' = 'create'
    AND data->'commit'->>'collection' IN ('app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like')
GROUP BY event, hour_of_day
ORDER BY hour_of_day, event;

\echo '\n========================================='
\echo 'QUERY 4: First post timestamp by user'
\echo '========================================='
EXPLAIN SELECT
    data->>'did' AS user_id,
    MIN(TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (data->>'time_us')::BIGINT) AS first_post_ts
FROM bluesky
WHERE data->>'kind' = 'commit'
    AND data->'commit'->>'operation' = 'create'
    AND data->'commit'->>'collection' = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC
LIMIT 3;

SELECT
    data->>'did' AS user_id,
    MIN(TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (data->>'time_us')::BIGINT) AS first_post_ts
FROM bluesky
WHERE data->>'kind' = 'commit'
    AND data->'commit'->>'operation' = 'create'
    AND data->'commit'->>'collection' = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC
LIMIT 3;

\echo '\n========================================='
\echo 'QUERY 5: User activity span (milliseconds)'
\echo '========================================='
EXPLAIN SELECT
    data->>'did' AS user_id,
    EXTRACT(EPOCH FROM (
        MAX(TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (data->>'time_us')::BIGINT) -
        MIN(TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (data->>'time_us')::BIGINT)
    )) * 1000 AS activity_span_ms
FROM bluesky
WHERE data->>'kind' = 'commit'
    AND data->'commit'->>'operation' = 'create'
    AND data->'commit'->>'collection' = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span_ms DESC
LIMIT 3;

SELECT
    data->>'did' AS user_id,
    EXTRACT(EPOCH FROM (
        MAX(TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (data->>'time_us')::BIGINT) -
        MIN(TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * (data->>'time_us')::BIGINT)
    )) * 1000 AS activity_span_ms
FROM bluesky
WHERE data->>'kind' = 'commit'
    AND data->'commit'->>'operation' = 'create'
    AND data->'commit'->>'collection' = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span_ms DESC
LIMIT 3;

\echo '\n========================================='
\echo 'SUMMARY: Check EXPLAIN outputs above'
\echo '========================================='
\echo 'All queries should show:'
\echo '  - Index Scan (not Seq Scan)'
\echo '  - Filters with @> containment operator'
\echo '  - DEBUG logs showing transformation depth'
\echo '  - DEBUG logs showing search values used'
\echo '========================================='
