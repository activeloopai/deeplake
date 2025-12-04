\i sql/utils.psql

-- Cleanup any existing objects
DROP SCHEMA IF EXISTS schema_a CASCADE;
DROP SCHEMA IF EXISTS schema_b CASCADE;
DROP SCHEMA IF EXISTS schema_c CASCADE;
DROP SCHEMA IF EXISTS "default" CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_deeplake;

DO $$
DECLARE
    row_count INTEGER;
BEGIN
    -- Test 1: Create multiple schemas
    RAISE NOTICE 'Test 1: Creating multiple schemas';
    CREATE SCHEMA schema_a;
    CREATE SCHEMA schema_b;
    CREATE SCHEMA schema_c;
    CREATE SCHEMA "default";
    RAISE NOTICE 'Created schemas: schema_a, schema_b, schema_c, default';

    -- Test 2: Create same-named tables in different schemas
    RAISE NOTICE 'Test 2: Creating tables with same name in different schemas';
    CREATE TABLE public.users (id int, name text, age int) USING deeplake;
    CREATE TABLE schema_a.users (id int, name text, age int) USING deeplake;
    CREATE TABLE schema_b.users (id int, name text, age int) USING deeplake;
    CREATE TABLE "default".users (id int, name text, age int) USING deeplake;
    RAISE NOTICE 'Created users table in public, schema_a, schema_b and default schemas';

    -- Test 3: Insert different data into each schema's table
    RAISE NOTICE 'Test 3: Inserting data into tables in different schemas';
    INSERT INTO public.users VALUES (1, 'public_user_1', 25), (2, 'public_user_2', 30), (3, 'public_user_3', 35);
    INSERT INTO schema_a.users VALUES (10, 'schema_a_user_1', 20), (11, 'schema_a_user_2', 22);
    INSERT INTO schema_b.users VALUES (100, 'schema_b_user_1', 40), (101, 'schema_b_user_2', 45),
                                       (102, 'schema_b_user_3', 50), (103, 'schema_b_user_4', 55);
    INSERT INTO "default".users VALUES (1000, 'default_user_1', 60), (1001, 'default_user_2', 65);
    RAISE NOTICE 'Data inserted into all schemas';

    -- Test 4: Verify row counts in each schema
    RAISE NOTICE 'Test 4: Verifying row counts in each schema';
    SELECT COUNT(*) INTO row_count FROM public.users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'Expected 3 rows in public.users, got %', row_count;
    END IF;
    RAISE NOTICE 'public.users has 3 rows';

    SELECT COUNT(*) INTO row_count FROM schema_a.users;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'Expected 2 rows in schema_a.users, got %', row_count;
    END IF;
    RAISE NOTICE 'schema_a.users has 2 rows';

    SELECT COUNT(*) INTO row_count FROM schema_b.users;
    IF row_count != 4 THEN
        RAISE EXCEPTION 'Expected 4 rows in schema_b.users, got %', row_count;
    END IF;
    RAISE NOTICE 'schema_b.users has 4 rows';

    SELECT COUNT(*) INTO row_count FROM "default".users;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'Expected 2 rows in default.users, got %', row_count;
    END IF;
    RAISE NOTICE 'default.users has 2 rows';

    -- Test 5: Query from different schemas in a single query (UNION)
    RAISE NOTICE 'Test 5: Querying from multiple schemas in a single query';
    SELECT COUNT(*) INTO row_count FROM (
        SELECT * FROM public.users
        UNION ALL
        SELECT * FROM schema_a.users
        UNION ALL
        SELECT * FROM schema_b.users
        UNION ALL
        SELECT * FROM "default".users
    ) combined;
    IF row_count != 11 THEN
        RAISE EXCEPTION 'Expected 11 rows from UNION ALL, got %', row_count;
    END IF;
    RAISE NOTICE 'UNION ALL query returned 11 rows correctly';

    -- Test 6: Query with JOIN across schemas
    RAISE NOTICE 'Test 6: Joining tables from different schemas';
    SELECT COUNT(*) INTO row_count FROM public.users p CROSS JOIN schema_a.users a;
    IF row_count != 6 THEN
        RAISE EXCEPTION 'Expected 6 rows from CROSS JOIN, got %', row_count;
    END IF;
    RAISE NOTICE 'Cross-schema CROSS JOIN returned 6 rows correctly';

    -- Test 7: Change search_path and query without schema qualification
    RAISE NOTICE 'Test 7: Testing search_path changes (single schema only)';

    -- Default search_path (public)
    EXECUTE 'SET LOCAL search_path TO public';
    SELECT COUNT(*) INTO row_count FROM users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'With search_path=public, expected 3 rows from users, got %', row_count;
    END IF;
    RAISE NOTICE 'search_path=public: queries public.users (3 rows)';

    -- Change search_path to schema_a only
    EXECUTE 'SET LOCAL search_path TO schema_a';
    SELECT COUNT(*) INTO row_count FROM users;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'With search_path=schema_a, expected 2 rows from users, got %', row_count;
    END IF;
    RAISE NOTICE 'search_path=schema_a: queries schema_a.users (2 rows)';

    -- Change search_path to schema_b only
    EXECUTE 'SET LOCAL search_path TO schema_b';
    SELECT COUNT(*) INTO row_count FROM users;
    IF row_count != 4 THEN
        RAISE EXCEPTION 'With search_path=schema_b, expected 4 rows from users, got %', row_count;
    END IF;
    RAISE NOTICE 'search_path=schema_b: queries schema_b.users (4 rows)';

    -- Change search_path to "default" only (requires quoting)
    EXECUTE 'SET LOCAL search_path TO "default"';
    SELECT COUNT(*) INTO row_count FROM users;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'With search_path=default, expected 2 rows from users, got %', row_count;
    END IF;
    RAISE NOTICE 'search_path=default: queries default.users (2 rows)';

    -- Reset search_path to public
    EXECUTE 'SET LOCAL search_path TO public';
    SELECT COUNT(*) INTO row_count FROM users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'After reset to public, expected 3 rows from users, got %', row_count;
    END IF;
    RAISE NOTICE 'search_path=public: queries public.users (3 rows)';

    -- Test 8: Create table in non-default schema using search_path
    RAISE NOTICE 'Test 8: Creating table in schema_c using search_path';
    EXECUTE 'SET LOCAL search_path TO schema_c';
    CREATE TABLE products (id int, name text, price float) USING deeplake;
    INSERT INTO products VALUES (1, 'Product A', 10.5), (2, 'Product B', 20.0);

    SELECT COUNT(*) INTO row_count FROM products;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'Expected 2 rows in products, got %', row_count;
    END IF;

    SELECT COUNT(*) INTO row_count FROM schema_c.products;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'Expected 2 rows in schema_c.products, got %', row_count;
    END IF;
    RAISE NOTICE 'Table created in schema_c via search_path (2 rows)';

    -- Test 9: Verify qualified vs unqualified names with different search_path
    RAISE NOTICE 'Test 9: Testing qualified vs unqualified table references';
    EXECUTE 'SET LOCAL search_path TO schema_a';

    -- Unqualified name uses schema_a (first in search_path)
    SELECT COUNT(*) INTO row_count FROM users;
    IF row_count != 2 THEN
        RAISE EXCEPTION 'Unqualified users should have 2 rows, got %', row_count;
    END IF;

    -- Qualified names work regardless of search_path
    SELECT COUNT(*) INTO row_count FROM public.users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'public.users should have 3 rows, got %', row_count;
    END IF;

    SELECT COUNT(*) INTO row_count FROM schema_b.users;
    IF row_count != 4 THEN
        RAISE EXCEPTION 'schema_b.users should have 4 rows, got %', row_count;
    END IF;
    RAISE NOTICE 'Qualified vs unqualified names working correctly';

    -- Test 10: Update and delete operations across schemas
    RAISE NOTICE 'Test 10: Testing UPDATE and DELETE across schemas';
    EXECUTE 'SET LOCAL search_path TO public';

    -- Update in public.users
    UPDATE public.users SET age = 26 WHERE id = 1;
    SELECT COUNT(*) INTO row_count FROM public.users WHERE age = 26;
    IF row_count != 1 THEN
        RAISE EXCEPTION 'Expected 1 updated row in public.users, got %', row_count;
    END IF;

    -- Update in schema_a.users
    UPDATE schema_a.users SET name = 'updated_name' WHERE id = 10;
    SELECT COUNT(*) INTO row_count FROM schema_a.users WHERE name = 'updated_name';
    IF row_count != 1 THEN
        RAISE EXCEPTION 'Expected 1 updated row in schema_a.users, got %', row_count;
    END IF;

    -- Delete from schema_b.users
    DELETE FROM schema_b.users WHERE id = 100;
    SELECT COUNT(*) INTO row_count FROM schema_b.users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'Expected 3 rows in schema_b.users after delete, got %', row_count;
    END IF;
    RAISE NOTICE 'UPDATE and DELETE operations successful across schemas';

    -- Test 11: Aggregations across multiple schemas
    RAISE NOTICE 'Test 11: Testing aggregations across schemas';
    SELECT COUNT(*) INTO row_count FROM (
        SELECT * FROM public.users
        UNION ALL
        SELECT * FROM schema_a.users
        UNION ALL
        SELECT * FROM schema_b.users
        UNION ALL
        SELECT * FROM "default".users
    ) all_users;
    IF row_count != 10 THEN
        RAISE EXCEPTION 'Expected 10 total users across all schemas, got %', row_count;
    END IF;
    RAISE NOTICE 'Cross-schema aggregation successful (10 total users)';

    -- Test 12: DROP TABLE from specific schema
    RAISE NOTICE 'Test 12: Testing DROP TABLE from specific schema';
    DROP TABLE schema_a.users;

    -- Verify schema_a.users is dropped but others remain
    SELECT COUNT(*) INTO row_count FROM public.users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'public.users should still have 3 rows, got %', row_count;
    END IF;

    SELECT COUNT(*) INTO row_count FROM schema_b.users;
    IF row_count != 3 THEN
        RAISE EXCEPTION 'schema_b.users should still have 3 rows, got %', row_count;
    END IF;
    RAISE NOTICE 'Dropped table from schema_a, others remain intact';

    -- Test 13: Create table with same name as dropped one
    RAISE NOTICE 'Test 13: Recreating table with different structure';
    CREATE TABLE schema_a.users (id int, name text, department text) USING deeplake;
    INSERT INTO schema_a.users VALUES (200, 'new_user', 'Engineering');

    SELECT COUNT(*) INTO row_count FROM schema_a.users;
    IF row_count != 1 THEN
        RAISE EXCEPTION 'Expected 1 row in recreated schema_a.users, got %', row_count;
    END IF;

    SELECT COUNT(*) INTO row_count FROM schema_a.users WHERE department = 'Engineering';
    IF row_count != 1 THEN
        RAISE EXCEPTION 'Expected 1 row with department=Engineering, got %', row_count;
    END IF;
    RAISE NOTICE 'Table recreated with different structure';

    RAISE NOTICE 'All schema tests passed successfully!';

EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
        RAISE;
END $$;

-- Cleanup
DROP SCHEMA IF EXISTS schema_a CASCADE;
DROP SCHEMA IF EXISTS schema_b CASCADE;
DROP SCHEMA IF EXISTS schema_c CASCADE;
DROP SCHEMA IF EXISTS "default" CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
