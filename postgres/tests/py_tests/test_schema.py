"""
Test PostgreSQL schema operations with deeplake storage.

Ported from: postgres/tests/sql/schema_test.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_schema_operations(db_conn: asyncpg.Connection):
    """
    Test comprehensive schema operations with deeplake storage.

    Tests:
    - Creating multiple schemas
    - Same-named tables in different schemas
    - Data isolation between schemas
    - UNION ALL across schemas
    - CROSS JOIN across schemas
    - search_path behavior
    - Qualified vs unqualified table names
    - UPDATE/DELETE operations across schemas
    - DROP TABLE from specific schema
    - Table recreation with different structure
    """
    assertions = Assertions(db_conn)

    try:
        # Test 1: Create multiple schemas
        await db_conn.execute("CREATE SCHEMA schema_a")
        await db_conn.execute("CREATE SCHEMA schema_b")
        await db_conn.execute("CREATE SCHEMA schema_c")
        await db_conn.execute('CREATE SCHEMA "default"')

        # Test 2: Create same-named tables in different schemas
        await db_conn.execute("""
            CREATE TABLE public.users (id int, name text, age int) USING deeplake
        """)
        await db_conn.execute("""
            CREATE TABLE schema_a.users (id int, name text, age int) USING deeplake
        """)
        await db_conn.execute("""
            CREATE TABLE schema_b.users (id int, name text, age int) USING deeplake
        """)
        await db_conn.execute("""
            CREATE TABLE "default".users (id int, name text, age int) USING deeplake
        """)

        # Test 3: Insert different data into each schema's table
        await db_conn.execute("""
            INSERT INTO public.users VALUES
                (1, 'public_user_1', 25),
                (2, 'public_user_2', 30),
                (3, 'public_user_3', 35)
        """)

        await db_conn.execute("""
            INSERT INTO schema_a.users VALUES
                (10, 'schema_a_user_1', 20),
                (11, 'schema_a_user_2', 22)
        """)

        await db_conn.execute("""
            INSERT INTO schema_b.users VALUES
                (100, 'schema_b_user_1', 40),
                (101, 'schema_b_user_2', 45),
                (102, 'schema_b_user_3', 50),
                (103, 'schema_b_user_4', 55)
        """)

        await db_conn.execute("""
            INSERT INTO "default".users VALUES
                (1000, 'default_user_1', 60),
                (1001, 'default_user_2', 65)
        """)

        # Test 4: Verify row counts in each schema
        public_count = await db_conn.fetchval("SELECT COUNT(*) FROM public.users")
        assert public_count == 3, f"Expected 3 rows in public.users, got {public_count}"

        schema_a_count = await db_conn.fetchval("SELECT COUNT(*) FROM schema_a.users")
        assert schema_a_count == 2, f"Expected 2 rows in schema_a.users, got {schema_a_count}"

        schema_b_count = await db_conn.fetchval("SELECT COUNT(*) FROM schema_b.users")
        assert schema_b_count == 4, f"Expected 4 rows in schema_b.users, got {schema_b_count}"

        default_count = await db_conn.fetchval('SELECT COUNT(*) FROM "default".users')
        assert default_count == 2, f"Expected 2 rows in default.users, got {default_count}"

        # Test 5: Query from different schemas in a single query (UNION ALL)
        union_count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM (
                SELECT * FROM public.users
                UNION ALL
                SELECT * FROM schema_a.users
                UNION ALL
                SELECT * FROM schema_b.users
                UNION ALL
                SELECT * FROM "default".users
            ) combined
        """)
        assert union_count == 11, f"Expected 11 rows from UNION ALL, got {union_count}"

        # Test 6: CROSS JOIN across schemas
        cross_join_count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM public.users p CROSS JOIN schema_a.users a
        """)
        assert cross_join_count == 6, f"Expected 6 rows from CROSS JOIN, got {cross_join_count}"

        # Test 7: search_path changes
        # Default search_path (public)
        await db_conn.execute("SET search_path TO public")
        unqualified_public = await db_conn.fetchval("SELECT COUNT(*) FROM users")
        assert unqualified_public == 3, \
            f"With search_path=public, expected 3 rows, got {unqualified_public}"

        # Change to schema_a
        await db_conn.execute("SET search_path TO schema_a")
        unqualified_a = await db_conn.fetchval("SELECT COUNT(*) FROM users")
        assert unqualified_a == 2, \
            f"With search_path=schema_a, expected 2 rows, got {unqualified_a}"

        # Change to schema_b
        await db_conn.execute("SET search_path TO schema_b")
        unqualified_b = await db_conn.fetchval("SELECT COUNT(*) FROM users")
        assert unqualified_b == 4, \
            f"With search_path=schema_b, expected 4 rows, got {unqualified_b}"

        # Change to "default" (requires quoting)
        await db_conn.execute('SET search_path TO "default"')
        unqualified_default = await db_conn.fetchval("SELECT COUNT(*) FROM users")
        assert unqualified_default == 2, \
            f"With search_path=default, expected 2 rows, got {unqualified_default}"

        # Reset to public
        await db_conn.execute("SET search_path TO public")
        unqualified_public_reset = await db_conn.fetchval("SELECT COUNT(*) FROM users")
        assert unqualified_public_reset == 3, \
            f"After reset to public, expected 3 rows, got {unqualified_public_reset}"

        # Test 8: Create table in schema_c using search_path
        await db_conn.execute("SET search_path TO schema_c")
        await db_conn.execute("""
            CREATE TABLE products (id int, name text, price float) USING deeplake
        """)
        await db_conn.execute("""
            INSERT INTO products VALUES (1, 'Product A', 10.5), (2, 'Product B', 20.0)
        """)

        products_count = await db_conn.fetchval("SELECT COUNT(*) FROM products")
        assert products_count == 2, f"Expected 2 rows in products, got {products_count}"

        products_qualified = await db_conn.fetchval("SELECT COUNT(*) FROM schema_c.products")
        assert products_qualified == 2, \
            f"Expected 2 rows in schema_c.products, got {products_qualified}"

        # Test 9: Qualified vs unqualified names
        await db_conn.execute("SET search_path TO schema_a")

        # Unqualified uses schema_a
        unqualified = await db_conn.fetchval("SELECT COUNT(*) FROM users")
        assert unqualified == 2, f"Unqualified users should have 2 rows, got {unqualified}"

        # Qualified names work regardless of search_path
        public_qualified = await db_conn.fetchval("SELECT COUNT(*) FROM public.users")
        assert public_qualified == 3, f"public.users should have 3 rows, got {public_qualified}"

        schema_b_qualified = await db_conn.fetchval("SELECT COUNT(*) FROM schema_b.users")
        assert schema_b_qualified == 4, \
            f"schema_b.users should have 4 rows, got {schema_b_qualified}"

        # Test 10: UPDATE and DELETE operations across schemas
        await db_conn.execute("SET search_path TO public")

        # Update in public.users
        await db_conn.execute("UPDATE public.users SET age = 26 WHERE id = 1")
        updated_count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM public.users WHERE age = 26
        """)
        assert updated_count == 1, f"Expected 1 updated row, got {updated_count}"

        # Update in schema_a.users
        await db_conn.execute("""
            UPDATE schema_a.users SET name = 'updated_name' WHERE id = 10
        """)
        updated_name_count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM schema_a.users WHERE name = 'updated_name'
        """)
        assert updated_name_count == 1, f"Expected 1 updated row, got {updated_name_count}"

        # Delete from schema_b.users
        await db_conn.execute("DELETE FROM schema_b.users WHERE id = 100")
        after_delete = await db_conn.fetchval("SELECT COUNT(*) FROM schema_b.users")
        assert after_delete == 3, f"Expected 3 rows after delete, got {after_delete}"

        # Test 11: Aggregations across multiple schemas
        total_users = await db_conn.fetchval("""
            SELECT COUNT(*) FROM (
                SELECT * FROM public.users
                UNION ALL
                SELECT * FROM schema_a.users
                UNION ALL
                SELECT * FROM schema_b.users
                UNION ALL
                SELECT * FROM "default".users
            ) all_users
        """)
        assert total_users == 10, f"Expected 10 total users, got {total_users}"

        # Test 12: DROP TABLE from specific schema
        await db_conn.execute("DROP TABLE schema_a.users")

        # Verify others remain
        public_after_drop = await db_conn.fetchval("SELECT COUNT(*) FROM public.users")
        assert public_after_drop == 3, f"public.users should still have 3 rows, got {public_after_drop}"

        schema_b_after_drop = await db_conn.fetchval("SELECT COUNT(*) FROM schema_b.users")
        assert schema_b_after_drop == 3, \
            f"schema_b.users should still have 3 rows, got {schema_b_after_drop}"

        # Test 13: Recreate table with different structure
        await db_conn.execute("""
            CREATE TABLE schema_a.users (id int, name text, department text) USING deeplake
        """)
        await db_conn.execute("""
            INSERT INTO schema_a.users VALUES (200, 'new_user', 'Engineering')
        """)

        recreated_count = await db_conn.fetchval("SELECT COUNT(*) FROM schema_a.users")
        assert recreated_count == 1, f"Expected 1 row in recreated table, got {recreated_count}"

        dept_count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM schema_a.users WHERE department = 'Engineering'
        """)
        assert dept_count == 1, f"Expected 1 row with department=Engineering, got {dept_count}"

        print("✓ Test passed: Schema operations work correctly with deeplake storage")

    finally:
        # Cleanup
        await db_conn.execute("DROP SCHEMA IF EXISTS schema_a CASCADE")
        await db_conn.execute("DROP SCHEMA IF EXISTS schema_b CASCADE")
        await db_conn.execute("DROP SCHEMA IF EXISTS schema_c CASCADE")
        await db_conn.execute('DROP SCHEMA IF EXISTS "default" CASCADE')
        await db_conn.execute("DROP TABLE IF EXISTS public.users CASCADE")
        await db_conn.execute("RESET search_path")
