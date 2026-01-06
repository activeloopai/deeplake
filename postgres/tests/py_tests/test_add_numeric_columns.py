"""
Test adding multiple NUMERIC columns to a deeplake table and updating them.

This test verifies that:
- Creating a table with SERIAL PRIMARY KEY and TEXT columns works
- Adding NUMERIC columns dynamically works correctly
- Inserting rows with NULL numeric values works (stored as 0 in deeplake)
- Querying after multiple schema changes works
- Multiple NUMERIC columns can be added sequentially
- UPDATE operations work correctly on numeric columns
- Multi-column updates work correctly
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_add_multiple_numeric_columns_with_null(db_conn: asyncpg.Connection):
    """
    Test adding multiple NUMERIC columns, inserting, and updating values.

    Tests:
    - Create table with SERIAL PRIMARY KEY and TEXT column using deeplake
    - Query with ORDER BY and LIMIT/OFFSET
    - Add first NUMERIC column via ALTER TABLE
    - Verify column is added and query works
    - Insert row with empty string and NULL NUMERIC value
    - Add second NUMERIC column via ALTER TABLE
    - Verify all columns are present and NULL numeric values are stored as 0
    - UPDATE single row with specific numeric values
    - UPDATE multiple rows in batch
    - UPDATE single column while leaving others unchanged
    - UPDATE multiple columns including TEXT and NUMERIC together
    - Verify final state of all rows after updates
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with id (SERIAL) and name (TEXT) columns using deeplake
        await db_conn.execute("""
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
            ) USING deeplake
        """)

        # Query on users table (should be empty initially)
        rows = await db_conn.fetch("""
            SELECT * FROM (SELECT * FROM users ORDER BY "id") sub
            LIMIT 20 OFFSET 0
        """)
        assert len(rows) == 0, f"Expected 0 rows initially, got {len(rows)}"

        # Add first numeric column to users
        await db_conn.execute("""
            ALTER TABLE users ADD COLUMN "uu" NUMERIC
        """)

        # Verify column was added to PostgreSQL catalog
        column_info = await db_conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'users'
            ORDER BY ordinal_position
        """)
        column_names = [col['column_name'] for col in column_info]
        assert 'uu' in column_names, \
            f"Column 'uu' should exist in catalog. Found: {column_names}"

        # Query users table after first schema change (still empty)
        rows = await db_conn.fetch("""
            SELECT * FROM (SELECT * FROM users ORDER BY "id") sub
            LIMIT 20 OFFSET 0
        """)
        assert len(rows) == 0, f"Expected 0 rows after adding column, got {len(rows)}"

        # Insert row with empty name and NULL UUID (numeric column)
        await db_conn.execute("""
            INSERT INTO users ("name", "uu") VALUES ('', NULL)
        """)

        # Query users table to verify insert
        rows = await db_conn.fetch("""
            SELECT * FROM (SELECT * FROM users ORDER BY "id") sub
            LIMIT 20 OFFSET 0
        """)
        assert len(rows) == 1, f"Expected 1 row after insert, got {len(rows)}"

        # Verify first row values
        row = rows[0]
        assert row['id'] == 1, f"Expected id=1, got {row['id']}"
        assert row['name'] == '', f"Expected empty string, got '{row['name']}'"
        # Numeric NULL values are stored as 0 in deeplake
        assert row['uu'] == 0, f"Expected uu=0 (NULL stored as 0), got {row['uu']}"

        # Insert another row with same values
        await db_conn.execute("""
            INSERT INTO users ("name", "uu") VALUES ('', NULL)
        """)

        # Query to verify both rows
        rows = await db_conn.fetch("""
            SELECT * FROM (SELECT * FROM users ORDER BY "id") sub
            LIMIT 20 OFFSET 0
        """)
        assert len(rows) == 2, f"Expected 2 rows after second insert, got {len(rows)}"

        # Add second numeric column
        await db_conn.execute("""
            ALTER TABLE users ADD COLUMN "uu23" NUMERIC
        """)

        # Verify second column was added
        column_info = await db_conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'users'
            ORDER BY ordinal_position
        """)
        column_names = [col['column_name'] for col in column_info]
        assert 'uu23' in column_names, \
            f"Column 'uu23' should exist in catalog. Found: {column_names}"
        assert column_names == ['id', 'name', 'uu', 'uu23'], \
            f"Expected ['id', 'name', 'uu', 'uu23'], got {column_names}"

        # Query users table after second schema change
        rows = await db_conn.fetch("SELECT * FROM users ORDER BY id")
        assert len(rows) == 2, f"Expected 2 rows after adding uu23, got {len(rows)}"

        # Verify both rows have all columns with correct values
        for i, row in enumerate(rows, start=1):
            assert row['id'] == i, f"Expected id={i}, got {row['id']}"
            assert row['name'] == '', f"Expected empty string, got '{row['name']}'"
            assert row['uu'] == 0, f"Expected uu=0, got {row['uu']}"
            # New column should be 0 for existing rows
            assert row['uu23'] == 0, f"Expected uu23=0 (NULL for new column), got {row['uu23']}"

        # Verify we can insert with the new column
        await db_conn.execute("""
            INSERT INTO users ("name", "uu", "uu23") VALUES ('test', 5, 10)
        """)

        # Verify the new row
        new_row = await db_conn.fetchrow("""
            SELECT * FROM users WHERE name = 'test'
        """)
        assert new_row is not None, "Expected to find row with name='test'"
        assert new_row['name'] == 'test', f"Expected 'test', got '{new_row['name']}'"
        assert new_row['uu'] == 5, f"Expected uu=5, got {new_row['uu']}"
        assert new_row['uu23'] == 10, f"Expected uu23=10, got {new_row['uu23']}"

        # Verify total row count before updates
        await assertions.assert_table_row_count(3, "users")

        # Test UPDATE operations on numeric columns
        # Update single row - set both numeric columns to specific values
        await db_conn.execute("""
            UPDATE users SET "uu" = 100, "uu23" = 200 WHERE id = 1
        """)

        # Verify the update
        updated_row = await db_conn.fetchrow("""
            SELECT * FROM users WHERE id = 1
        """)
        assert updated_row is not None, "Expected to find row with id=1"
        assert updated_row['uu'] == 100, f"Expected uu=100 after update, got {updated_row['uu']}"
        assert updated_row['uu23'] == 200, f"Expected uu23=200 after update, got {updated_row['uu23']}"
        assert updated_row['name'] == '', f"Name should remain unchanged, got '{updated_row['name']}'"

        # Update multiple rows at once
        await db_conn.execute("""
            UPDATE users SET "uu" = 50 WHERE id IN (2, 3)
        """)

        # Verify the batch update
        batch_updated = await db_conn.fetch("""
            SELECT * FROM users WHERE id IN (2, 3) ORDER BY id
        """)
        assert len(batch_updated) == 2, f"Expected 2 updated rows, got {len(batch_updated)}"
        for row in batch_updated:
            assert row['uu'] == 50, f"Expected uu=50 for id={row['id']}, got {row['uu']}"

        # Update single column while leaving other unchanged
        await db_conn.execute("""
            UPDATE users SET "uu23" = 999 WHERE id = 2
        """)

        # Verify partial column update
        partial_updated = await db_conn.fetchrow("""
            SELECT * FROM users WHERE id = 2
        """)
        assert partial_updated['uu'] == 50, \
            f"uu should remain 50 after partial update, got {partial_updated['uu']}"
        assert partial_updated['uu23'] == 999, \
            f"Expected uu23=999 after partial update, got {partial_updated['uu23']}"

        # Update another numeric value for row 3
        await db_conn.execute("""
            UPDATE users SET "uu23" = 333 WHERE id = 3
        """)

        # Verify the update
        row3_updated = await db_conn.fetchrow("""
            SELECT * FROM users WHERE id = 3
        """)
        assert row3_updated['uu'] == 50, \
            f"Expected uu=50 (unchanged), got {row3_updated['uu']}"
        assert row3_updated['uu23'] == 333, \
            f"Expected uu23=333 after update, got {row3_updated['uu23']}"
        assert row3_updated['name'] == 'test', \
            f"Name should remain 'test', got '{row3_updated['name']}'"

        # Update name column along with numeric columns
        await db_conn.execute("""
            UPDATE users SET "name" = 'updated', "uu" = 777, "uu23" = 888 WHERE id = 1
        """)

        # Verify multi-column update
        multi_updated = await db_conn.fetchrow("""
            SELECT * FROM users WHERE id = 1
        """)
        assert multi_updated['name'] == 'updated', \
            f"Expected name='updated', got '{multi_updated['name']}'"
        assert multi_updated['uu'] == 777, \
            f"Expected uu=777 after multi-update, got {multi_updated['uu']}"
        assert multi_updated['uu23'] == 888, \
            f"Expected uu23=888 after multi-update, got {multi_updated['uu23']}"

        # Final validation: query all rows and verify final state
        final_rows = await db_conn.fetch("SELECT * FROM users ORDER BY id")
        assert len(final_rows) == 3, f"Expected 3 rows in final state, got {len(final_rows)}"

        # Verify row 1 (id=1): updated name and both numeric columns
        assert final_rows[0]['id'] == 1
        assert final_rows[0]['name'] == 'updated'
        assert final_rows[0]['uu'] == 777
        assert final_rows[0]['uu23'] == 888

        # Verify row 2 (id=2): empty name, uu=50, uu23=999
        assert final_rows[1]['id'] == 2
        assert final_rows[1]['name'] == ''
        assert final_rows[1]['uu'] == 50
        assert final_rows[1]['uu23'] == 999

        # Verify row 3 (id=3): name='test', uu=50, uu23=333
        assert final_rows[2]['id'] == 3
        assert final_rows[2]['name'] == 'test'
        assert final_rows[2]['uu'] == 50
        assert final_rows[2]['uu23'] == 333

        print("✓ Test passed: Adding multiple NUMERIC columns and inserting NULL values works correctly")

    finally:
        # Cleanup
        try:
            await db_conn.execute("DROP TABLE IF EXISTS users CASCADE")
        except:
            pass  # Connection may be dead after errors
