"""
Test JOIN operations between deeplake and PostgreSQL tables.

Ported from: postgres/tests/sql/join.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_join_operations(db_conn: asyncpg.Connection):
    """
    Test various JOIN operations between deeplake and regular PostgreSQL tables.

    Tests:
    - INNER JOIN between deeplake and PostgreSQL tables
    - INNER JOIN between two deeplake tables
    - LEFT JOIN
    - RIGHT JOIN
    - FULL OUTER JOIN
    - CROSS JOIN
    - Complex multi-table JOINs
    - NULL handling in outer joins
    """
    assertions = Assertions(db_conn)

    try:
        # Create deeplake table 1
        await db_conn.execute("""
            CREATE TABLE dl_employees (
                emp_id int,
                name text,
                dept_id int,
                salary numeric
            ) USING deeplake
        """)

        # Create regular PostgreSQL table
        await db_conn.execute("""
            CREATE TABLE pg_departments (
                dept_id int,
                dept_name text
            )
        """)

        # Create deeplake table 2
        await db_conn.execute("""
            CREATE TABLE dl_projects (
                project_id int,
                emp_id int,
                project_name text
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO dl_employees VALUES
                (1, 'John', 1, 50000),
                (2, 'Jane', 2, 60000),
                (3, 'Bob', 1, 55000),
                (4, 'Alice', 3, 65000)
        """)

        await db_conn.execute("""
            INSERT INTO pg_departments VALUES
                (1, 'Engineering'),
                (2, 'Marketing'),
                (5, 'HR')
        """)

        await db_conn.execute("""
            INSERT INTO dl_projects VALUES
                (101, 1, 'Project A'),
                (102, 1, 'Project B'),
                (103, 2, 'Project C')
        """)

        # Test 1: INNER JOIN between deeplake and PostgreSQL tables
        await assertions.assert_query_row_count(
            3,
            """SELECT e.name, d.dept_name
               FROM dl_employees e
               JOIN pg_departments d ON e.dept_id = d.dept_id"""
        )

        # Test 2: INNER JOIN between two deeplake tables
        await assertions.assert_query_row_count(
            3,
            """SELECT e.name, p.project_name
               FROM dl_employees e
               JOIN dl_projects p ON e.emp_id = p.emp_id"""
        )

        # Test 3: LEFT JOIN to include all employees
        await assertions.assert_query_row_count(
            4,
            """SELECT e.name, d.dept_name
               FROM dl_employees e
               LEFT JOIN pg_departments d ON e.dept_id = d.dept_id"""
        )

        # Test 4: RIGHT JOIN to include all departments
        await assertions.assert_query_row_count(
            4,
            """SELECT e.name, d.dept_name
               FROM dl_employees e
               RIGHT JOIN pg_departments d ON e.dept_id = d.dept_id"""
        )

        # Test 5: FULL OUTER JOIN to include all records
        await assertions.assert_query_row_count(
            5,
            """SELECT e.name, d.dept_name
               FROM dl_employees e
               FULL OUTER JOIN pg_departments d ON e.dept_id = d.dept_id"""
        )

        # Test 6: CROSS JOIN between deeplake tables
        # 4 employees × 3 projects = 12 rows
        await assertions.assert_query_row_count(
            12,
            """SELECT e.name, p.project_name
               FROM dl_employees e
               CROSS JOIN dl_projects p"""
        )

        # Test 7: Complex multi-table LEFT JOINs
        await assertions.assert_query_row_count(
            5,
            """SELECT e.name, d.dept_name, p.project_name
               FROM dl_employees e
               LEFT JOIN pg_departments d ON e.dept_id = d.dept_id
               LEFT JOIN dl_projects p ON e.emp_id = p.emp_id
               ORDER BY e.name"""
        )

        # Test 8: Verify NULL results in outer joins
        # Alice has no matching department (dept_id 3 doesn't exist)
        await assertions.assert_query_row_count(
            1,
            """SELECT e.name
               FROM dl_employees e
               LEFT JOIN pg_departments d ON e.dept_id = d.dept_id
               WHERE d.dept_name IS NULL"""
        )

        print("✓ Test passed: All JOIN operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS dl_employees CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS pg_departments CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS dl_projects CASCADE")
