"""
Test UUID type handling to ensure no data corruption.

Bug Report: UUIDs with leading '00' bytes were being corrupted to '80' when
            selected without explicit ::text cast.

Example:
    Stored:  0085cc89-607e-4009-98f0-89d48d188e2f
    Without cast: 8085cc89-607e-4009-98f0-89d48d188e2f (CORRUPTED!)
    With ::text:  0085cc89-607e-4009-98f0-89d48d188e2f (CORRECT)

This test verifies that UUID-to-string conversion works correctly for all
byte values, especially edge cases like 0x00, 0x0F, 0x80, and 0xFF.
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_uuid_no_corruption(db_conn: asyncpg.Connection):
    """
    Test that UUID values are not corrupted during conversion to string.

    This test specifically checks:
    1. UUIDs starting with 0x00 (the reported bug case)
    2. UUIDs starting with 0x0F (low nibble non-zero)
    3. UUIDs starting with 0x10 (high nibble non-zero)
    4. UUIDs starting with 0x80 (high bit set)
    5. UUIDs starting with 0xFF (all bits set)
    6. NULL UUID values

    For each test case, we verify that:
    - Direct SELECT matches the original value
    - SELECT with ::text cast matches the original value
    - Both methods produce identical results
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table with UUID column
        await db_conn.execute("""
            CREATE TABLE test_uuid_corruption (
                id SERIAL PRIMARY KEY,
                uuid_val UUID,
                description TEXT
            ) USING deeplake
        """)

        # Test cases covering edge cases in UUID first byte
        test_cases = [
            ('0085cc89-607e-4009-98f0-89d48d188e2f', 'UUID starting with 0x00 (reported bug)'),
            ('0f85cc89-607e-4009-98f0-89d48d188e2f', 'UUID starting with 0x0F'),
            ('1085cc89-607e-4009-98f0-89d48d188e2f', 'UUID starting with 0x10'),
            ('8085cc89-607e-4009-98f0-89d48d188e2f', 'UUID starting with 0x80'),
            ('ff85cc89-607e-4009-98f0-89d48d188e2f', 'UUID starting with 0xFF'),
        ]

        # Insert test cases
        for uuid_str, description in test_cases:
            await db_conn.execute(
                "INSERT INTO test_uuid_corruption (uuid_val, description) VALUES ($1, $2)",
                uuid_str, description
            )

        # Insert NULL case
        await db_conn.execute(
            "INSERT INTO test_uuid_corruption (uuid_val, description) VALUES (NULL, $1)",
            'NULL UUID'
        )

        # Verify row count
        await assertions.assert_table_row_count(6, "test_uuid_corruption")

        # Test 1: Fetch UUIDs directly and verify they match original values
        print("\n=== Test 1: Direct UUID fetch (without explicit cast) ===")
        rows = await db_conn.fetch("""
            SELECT id, uuid_val, description
            FROM test_uuid_corruption
            ORDER BY id
        """)

        for i, row in enumerate(rows[:-1]):  # Skip NULL case
            uuid_val = str(row['uuid_val']) if row['uuid_val'] else None
            expected_uuid = test_cases[i][0]
            description = row['description']

            assert uuid_val == expected_uuid, (
                f"UUID corruption detected!\n"
                f"  Test case: {description}\n"
                f"  Expected: {expected_uuid}\n"
                f"  Got:      {uuid_val}\n"
                f"  This indicates a bug in UUID-to-string conversion."
            )
            print(f"✓ {description}: {uuid_val}")

        # Verify NULL case
        assert rows[-1]['uuid_val'] is None, "NULL UUID should remain NULL"
        print(f"✓ NULL UUID: None")

        # Test 2: Fetch UUIDs with explicit ::text cast
        print("\n=== Test 2: UUID fetch with ::text cast ===")
        rows_with_cast = await db_conn.fetch("""
            SELECT id, uuid_val::text as uuid_str, description
            FROM test_uuid_corruption
            ORDER BY id
        """)

        for i, row in enumerate(rows_with_cast[:-1]):  # Skip NULL case
            uuid_str = row['uuid_str']
            expected_uuid = test_cases[i][0]
            description = row['description']

            assert uuid_str == expected_uuid, (
                f"UUID with ::text cast doesn't match!\n"
                f"  Test case: {description}\n"
                f"  Expected: {expected_uuid}\n"
                f"  Got:      {uuid_str}"
            )
            print(f"✓ {description}: {uuid_str}")

        # Verify NULL case with cast
        assert rows_with_cast[-1]['uuid_str'] is None, "NULL UUID with ::text cast should remain NULL"
        print(f"✓ NULL UUID with ::text: None")

        # Test 3: Verify direct fetch and ::text cast produce identical results
        print("\n=== Test 3: Compare direct fetch vs ::text cast ===")
        for i in range(len(test_cases)):
            uuid_direct = str(rows[i]['uuid_val']) if rows[i]['uuid_val'] else None
            uuid_cast = rows_with_cast[i]['uuid_str']
            description = test_cases[i][1]

            assert uuid_direct == uuid_cast, (
                f"Mismatch between direct and cast results!\n"
                f"  Test case: {description}\n"
                f"  Direct:  {uuid_direct}\n"
                f"  Cast:    {uuid_cast}\n"
                f"  These should be identical."
            )
            print(f"✓ {description}: Both methods match")

        # Test 4: Test UUID comparison operations
        print("\n=== Test 4: UUID comparison operations ===")

        # Test equality with string
        count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM test_uuid_corruption
            WHERE uuid_val = '0085cc89-607e-4009-98f0-89d48d188e2f'::uuid
        """)
        assert count == 1, f"Expected 1 row matching UUID, got {count}"
        print(f"✓ UUID equality comparison works correctly")

        # Test UUID ordering
        ordered_rows = await db_conn.fetch("""
            SELECT uuid_val::text as uuid_str
            FROM test_uuid_corruption
            WHERE uuid_val IS NOT NULL
            ORDER BY uuid_val
        """)

        # Verify we got all non-NULL UUIDs back
        assert len(ordered_rows) == 5, f"Expected 5 non-NULL UUIDs, got {len(ordered_rows)}"

        # Verify they are in ascending order
        uuid_strs = [row['uuid_str'] for row in ordered_rows]
        sorted_uuids = sorted(uuid_strs)
        assert uuid_strs == sorted_uuids, (
            f"UUIDs not in correct order:\n"
            f"  Got:      {uuid_strs}\n"
            f"  Expected: {sorted_uuids}"
        )
        print(f"✓ UUID ordering works correctly")

        print("\n=== All UUID tests passed! ===")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_uuid_corruption")


@pytest.mark.asyncio
async def test_uuid_edge_cases(db_conn: asyncpg.Connection):
    """
    Test additional UUID edge cases and operations.
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_uuid_edges (
                id SERIAL PRIMARY KEY,
                uuid_val UUID
            ) USING deeplake
        """)

        # Test all-zeros and all-ones UUIDs
        await db_conn.execute("""
            INSERT INTO test_uuid_edges (uuid_val) VALUES
                ('00000000-0000-0000-0000-000000000000'),
                ('ffffffff-ffff-ffff-ffff-ffffffffffff')
        """)

        # Verify they round-trip correctly
        rows = await db_conn.fetch("SELECT uuid_val::text as uuid_str FROM test_uuid_edges ORDER BY id")

        assert rows[0]['uuid_str'] == '00000000-0000-0000-0000-000000000000', \
            f"All-zeros UUID corrupted: {rows[0]['uuid_str']}"
        print(f"✓ All-zeros UUID: {rows[0]['uuid_str']}")

        assert rows[1]['uuid_str'] == 'ffffffff-ffff-ffff-ffff-ffffffffffff', \
            f"All-ones UUID corrupted: {rows[1]['uuid_str']}"
        print(f"✓ All-ones UUID: {rows[1]['uuid_str']}")

        print("\n=== All UUID edge case tests passed! ===")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_uuid_edges")
