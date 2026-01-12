"""
Test exact_text index with multiple tables in custom schema using separate connections.

Tests:
- Creating custom schema
- Creating multiple tables with exact_text indexes in custom schema
- Inserting data into tables with parallel connections
- Querying with exact_text index
- Each operation uses a separate connection
"""
import pytest
import asyncpg
import asyncio
import os


async def get_connection():
    """Helper to create a new database connection."""
    user = os.environ.get("USER", "postgres")
    return await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0
    )


@pytest.mark.asyncio
async def test_exact_text_multiple_tables_separate_connections(pg_server):
    """
    Test exact_text index with multiple tables using separate connections for each operation.

    Tests:
    - Creating custom schema 'test_schema'
    - Creating table 'coco' with exact_text index in custom schema
    - Inserting 10,000 rows into coco using 10 parallel connections (1000 rows each)
    - Querying coco with exact text match
    - Creating table 'coco2' with exact_text index in custom schema
    - Inserting 10,000 rows into coco2 using 10 parallel connections (1000 rows each)
    - Querying coco2 with exact text match
    - Creating table 'coco3' with exact_text index in custom schema
    - Inserting 10,000 rows into coco3 using 10 parallel connections (1000 rows each)
    - Querying coco3 with exact text match
    """

    try:
        # Operation 1: Create custom schema
        conn1 = await get_connection()
        try:
            await conn1.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
            await conn1.execute("CREATE EXTENSION pg_deeplake")
            await conn1.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
            print("✓ Created schema 'test_schema'")
        finally:
            await conn1.close()

        # Operation 2: Create first table 'coco'
        conn2 = await get_connection()
        try:
            await conn2.execute("""
                CREATE TABLE "test_schema".coco (
                    id INTEGER,
                    name TEXT
                ) USING deeplake
            """)

            # Create exact_text index on name column
            await conn2.execute("""
                CREATE INDEX idx_coco_name_exact ON "test_schema".coco
                USING deeplake_index (name) WITH (index_type = 'exact_text')
            """)
            print("✓ Created table 'test_schema.coco' with exact_text index")
        finally:
            await conn2.close()

        # Operation 3: Insert data into coco table (10 connections, 1000 rows each in parallel)
        async def insert_batch(batch_num, start_id):
            """Insert 1000 rows from a single connection."""
            conn = await get_connection()
            try:
                values = []
                for i in range(start_id, start_id + 1000):
                    values.append(f"({i}, 'Name_{i}')")

                insert_query = f"""
                    INSERT INTO "test_schema".coco (id, name) VALUES
                    {', '.join(values)}
                """
                await conn.execute(insert_query)
                print(f"  ✓ Batch {batch_num}: Inserted 1000 rows (id {start_id} to {start_id + 999})")
            finally:
                await conn.close()

        # Run 10 inserts in parallel
        print("Inserting 10,000 rows into 'test_schema.coco' using 10 parallel connections...")
        insert_tasks = [
            insert_batch(i + 1, i * 1000 + 1)
            for i in range(10)
        ]
        await asyncio.gather(*insert_tasks)
        print("✓ Completed parallel inserts into 'test_schema.coco'")

        # Operation 4: Query coco table with exact text match (verify row count and search)
        conn4 = await get_connection()
        try:
            # First verify we have 10,000 rows
            count_result = await conn4.fetchval("""
                SELECT COUNT(*) FROM "test_schema".coco
            """)
            assert count_result == 10000, f"Expected 10000 rows, got {count_result}"
            print(f"✓ Verified 'test_schema.coco' has {count_result} rows")

            # Test exact text search
            result = await conn4.fetch("""
                SELECT id, name FROM "test_schema".coco
                WHERE name = 'Name_5500'
            """)
            assert len(result) == 1, f"Expected 1 result, got {len(result)}"
            assert result[0]['id'] == 5500, f"Expected id=5500, got {result[0]['id']}"
            assert result[0]['name'] == 'Name_5500', f"Expected name='Name_5500', got {result[0]['name']}"
            print(f"✓ Query on 'test_schema.coco' returned: id={result[0]['id']}, name={result[0]['name']}")
        finally:
            await conn4.close()

        # Operation 5: Create second table 'coco2'
        conn5 = await get_connection()
        try:
            await conn5.execute("""
                CREATE TABLE "test_schema".coco2 (
                    id INTEGER,
                    name TEXT
                ) USING deeplake
            """)

            # Create exact_text index on name column
            await conn5.execute("""
                CREATE INDEX idx_coco2_name_exact ON "test_schema".coco2
                USING deeplake_index (name) WITH (index_type = 'exact_text')
            """)
            print("✓ Created table 'test_schema.coco2' with exact_text index")
        finally:
            await conn5.close()

        # Operation 6: Insert data into coco2 table (10 connections, 1000 rows each in parallel)
        async def insert_batch_coco2(batch_num, start_id):
            """Insert 1000 rows from a single connection into coco2."""
            conn = await get_connection()
            try:
                values = []
                for i in range(start_id, start_id + 1000):
                    values.append(f"({i}, 'Person_{i}')")

                insert_query = f"""
                    INSERT INTO "test_schema".coco2 (id, name) VALUES
                    {', '.join(values)}
                """
                await conn.execute(insert_query)
                print(f"  ✓ Batch {batch_num}: Inserted 1000 rows (id {start_id} to {start_id + 999})")
            finally:
                await conn.close()

        # Run 10 inserts in parallel
        print("Inserting 10,000 rows into 'test_schema.coco2' using 10 parallel connections...")
        insert_tasks_coco2 = [
            insert_batch_coco2(i + 1, 10000 + i * 1000 + 1)
            for i in range(10)
        ]
        await asyncio.gather(*insert_tasks_coco2)
        print("✓ Completed parallel inserts into 'test_schema.coco2'")

        # Operation 7: Query coco2 table with exact text match (verify row count and search)
        conn7 = await get_connection()
        try:
            # First verify we have 10,000 rows
            count_result = await conn7.fetchval("""
                SELECT COUNT(*) FROM "test_schema".coco2
            """)
            assert count_result == 10000, f"Expected 10000 rows, got {count_result}"
            print(f"✓ Verified 'test_schema.coco2' has {count_result} rows")

            # Test exact text search
            result = await conn7.fetch("""
                SELECT id, name FROM "test_schema".coco2
                WHERE name = 'Person_15500'
            """)
            assert len(result) == 1, f"Expected 1 result, got {len(result)}"
            assert result[0]['id'] == 15500, f"Expected id=15500, got {result[0]['id']}"
            assert result[0]['name'] == 'Person_15500', f"Expected name='Person_15500', got {result[0]['name']}"
            print(f"✓ Query on 'test_schema.coco2' returned: id={result[0]['id']}, name={result[0]['name']}")
        finally:
            await conn7.close()

        # Operation 8: Create third table 'coco3'
        conn8 = await get_connection()
        try:
            await conn8.execute("""
                CREATE TABLE "test_schema".coco3 (
                    id INTEGER,
                    name TEXT
                ) USING deeplake
            """)

            # Create exact_text index on name column
            await conn8.execute("""
                CREATE INDEX idx_coco3_name_exact ON "test_schema".coco3
                USING deeplake_index (name) WITH (index_type = 'exact_text')
            """)
            print("✓ Created table 'test_schema.coco3' with exact_text index")
        finally:
            await conn8.close()

        # Operation 9: Insert data into coco3 table (10 connections, 1000 rows each in parallel)
        async def insert_batch_coco3(batch_num, start_id):
            """Insert 1000 rows from a single connection into coco3."""
            conn = await get_connection()
            try:
                values = []
                for i in range(start_id, start_id + 1000):
                    values.append(f"({i}, 'User_{i}')")

                insert_query = f"""
                    INSERT INTO "test_schema".coco3 (id, name) VALUES
                    {', '.join(values)}
                """
                await conn.execute(insert_query)
                print(f"  ✓ Batch {batch_num}: Inserted 1000 rows (id {start_id} to {start_id + 999})")
            finally:
                await conn.close()

        # Run 10 inserts in parallel
        print("Inserting 10,000 rows into 'test_schema.coco3' using 10 parallel connections...")
        insert_tasks_coco3 = [
            insert_batch_coco3(i + 1, 20000 + i * 1000 + 1)
            for i in range(10)
        ]
        await asyncio.gather(*insert_tasks_coco3)
        print("✓ Completed parallel inserts into 'test_schema.coco3'")

        # Operation 10: Query coco3 table with exact text match (verify row count and search)
        conn9 = await get_connection()
        try:
            # First verify we have 10,000 rows
            count_result = await conn9.fetchval("""
                SELECT COUNT(*) FROM "test_schema".coco3
            """)
            assert count_result == 10000, f"Expected 10000 rows, got {count_result}"
            print(f"✓ Verified 'test_schema.coco3' has {count_result} rows")

            # Test exact text search
            result = await conn9.fetch("""
                SELECT id, name FROM "test_schema".coco3
                WHERE name = 'User_25500'
            """)
            assert len(result) == 1, f"Expected 1 result, got {len(result)}"
            assert result[0]['id'] == 25500, f"Expected id=25500, got {result[0]['id']}"
            assert result[0]['name'] == 'User_25500', f"Expected name='User_25500', got {result[0]['name']}"
            print(f"✓ Query on 'test_schema.coco3' returned: id={result[0]['id']}, name={result[0]['name']}")
        finally:
            await conn9.close()

        print("\n✓ Test passed: exact_text index works correctly with multiple tables in custom schema using separate connections")

    finally:
        # Cleanup
        cleanup_conn = await get_connection()
        try:
            await cleanup_conn.execute("DROP SCHEMA IF EXISTS test_schema CASCADE")
            print("✓ Cleaned up test_schema")
        finally:
            await cleanup_conn.close()
