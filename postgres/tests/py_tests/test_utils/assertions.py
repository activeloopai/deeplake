"""
Test assertion utilities - Python equivalents of SQL utils.psql functions.
"""
import asyncpg
from typing import List, Tuple, Any, Optional


class Assertions:
    """
    Assertion helpers for PostgreSQL extension tests.

    This class provides Python equivalents of the PL/pgSQL assertion
    functions defined in utils.psql.
    """

    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def assert_table_row_count(self, expected: int, table_name: str) -> None:
        """
        Assert that a table has the expected number of rows.

        Args:
            expected: Expected row count
            table_name: Name of the table to check

        Raises:
            AssertionError: If row count doesn't match
        """
        query = f"SELECT count(*) FROM {table_name}"
        actual = await self.conn.fetchval(query)

        assert actual == expected, (
            f"Check failed: Expected {expected} rows in {table_name}, "
            f"but got {actual}"
        )
        print(f"✓ Check passed: {table_name} has {expected} rows as expected.")

    async def assert_query_row_count(self, expected: int, query: str) -> None:
        """
        Assert that a query returns the expected number of rows.

        Args:
            expected: Expected row count
            query: SQL query to execute

        Raises:
            AssertionError: If row count doesn't match
        """
        count_query = f"SELECT count(*) FROM ({query}) AS subquery"
        actual = await self.conn.fetchval(count_query)

        assert actual == expected, (
            f"Check failed: Expected row count = {expected}, but got {actual}\n"
            f"Query: {query}"
        )
        print(f"✓ Check passed: Expected row count = {expected}.")

    async def assert_query_result(
        self,
        query: str,
        expected_rows: List[Tuple[Any, ...]],
        ordered: bool = True
    ) -> None:
        """
        Assert that query results match expected rows.

        Args:
            query: SQL query to execute
            expected_rows: List of tuples representing expected rows
            ordered: If True, compare rows in order. If False, treat as sets.

        Raises:
            AssertionError: If results don't match
        """
        actual_rows = await self.conn.fetch(query)
        actual_tuples = [tuple(row.values()) for row in actual_rows]

        if ordered:
            # Compare with order
            assert len(actual_tuples) == len(expected_rows), (
                f"Row count mismatch: expected {len(expected_rows)}, "
                f"got {len(actual_tuples)}"
            )

            for i, (actual, expected) in enumerate(zip(actual_tuples, expected_rows)):
                assert actual == expected, (
                    f"Row {i} mismatch:\n"
                    f"  Expected: {expected}\n"
                    f"  Got:      {actual}"
                )
        else:
            # Compare as sets (order doesn't matter)
            actual_set = set(actual_tuples)
            expected_set = set(expected_rows)

            missing = expected_set - actual_set
            extra = actual_set - expected_set

            assert not missing and not extra, (
                f"Result set mismatch:\n"
                f"  Missing rows: {missing}\n"
                f"  Extra rows: {extra}"
            )

        print("✓ Check passed: Query result matches expected output.")

    async def assert_query_result_from_table(
        self,
        query: str,
        expected_table: str
    ) -> None:
        """
        Assert that query results match rows from an expected table.

        This is equivalent to assert_query_result from utils.psql.

        Args:
            query: SQL query to execute
            expected_table: Name of table containing expected results

        Raises:
            AssertionError: If results don't match
        """
        # Get expected rows from table
        expected_rows = await self.conn.fetch(f"SELECT * FROM {expected_table}")
        expected_tuples = [tuple(row.values()) for row in expected_rows]

        # Compare with query results
        await self.assert_query_result(query, expected_tuples, ordered=True)

    async def assert_aggregate_match(
        self,
        table1: str,
        table2: str,
        column_name: str,
        aggregate_func: str,
        test_description: str = "",
        representation: str = ""
    ) -> None:
        """
        Compare aggregate results between two tables.

        Args:
            table1: First table name
            table2: Second table name
            column_name: Column to aggregate
            aggregate_func: Aggregate function (SUM, AVG, etc.)
            test_description: Optional test description for error messages
            representation: Optional type cast (e.g., 'numeric')

        Raises:
            AssertionError: If aggregates don't match
        """
        # Build column expression
        if representation:
            column_expr = f"{column_name}::{representation}"
        else:
            column_expr = column_name

        query1 = f"SELECT {aggregate_func}({column_expr}) FROM {table1}"
        query2 = f"SELECT {aggregate_func}({column_expr}) FROM {table2}"

        result1 = await self.conn.fetchval(query1)
        result2 = await self.conn.fetchval(query2)

        # Handle NULL values
        result1 = result1 if result1 is not None else -999999
        result2 = result2 if result2 is not None else -999999

        desc = f" ({test_description})" if test_description else ""
        repr_str = f"::{representation}" if representation else ""

        assert result1 == result2, (
            f"Check failed{desc}: {aggregate_func}({column_name}{repr_str}) differs - "
            f"{table1}: {result1}, {table2}: {result2}"
        )

        print(
            f"✓ Check passed{desc}: {aggregate_func}({column_name}{repr_str}) "
            f"matches - both tables: {result1}"
        )

    async def assert_tables_query_match(
        self,
        query1: str,
        query2: str,
        test_description: str = ""
    ) -> None:
        """
        Compare results from two queries (typically on different tables).

        Args:
            query1: First query
            query2: Second query
            test_description: Optional test description

        Raises:
            AssertionError: If results don't match
        """
        desc = f" ({test_description})" if test_description else ""

        # Check for mismatched rows
        mismatch_query = f"""
            WITH q1_results AS ({query1}), q2_results AS ({query2})
            SELECT COUNT(*) FROM (
                SELECT * FROM q1_results EXCEPT SELECT * FROM q2_results
            ) AS differences
        """
        mismatch_count = await self.conn.fetchval(mismatch_query)

        assert mismatch_count == 0, (
            f"Check failed{desc}: First query has {mismatch_count} row(s) "
            f"that differ from second query"
        )

        # Check for missing rows
        missing_query = f"""
            WITH q1_results AS ({query1}), q2_results AS ({query2})
            SELECT COUNT(*) FROM (
                SELECT * FROM q2_results EXCEPT SELECT * FROM q1_results
            ) AS missing_rows
        """
        missing_count = await self.conn.fetchval(missing_query)

        assert missing_count == 0, (
            f"Check failed{desc}: First query is missing {missing_count} row(s) "
            f"that exist in second query"
        )

        print(f"✓ Check passed{desc}: Query results match exactly.")

    async def is_using_index_scan(self, query_text: str) -> bool:
        """
        Check if a query uses an index scan.

        Args:
            query_text: SQL query to analyze

        Returns:
            True if query uses index scan, False otherwise
        """
        explain_query = f"EXPLAIN (FORMAT JSON) {query_text}"
        result = await self.conn.fetchval(explain_query)

        return self._check_plan_node(
            result[0]["Plan"],
            ["Index Scan", "Index Only Scan", "Bitmap Index Scan"]
        )

    async def is_using_seq_scan(self, query_text: str) -> bool:
        """
        Check if a query uses a sequential scan.

        Args:
            query_text: SQL query to analyze

        Returns:
            True if query uses sequential scan, False otherwise
        """
        explain_query = f"EXPLAIN (FORMAT JSON) {query_text}"
        result = await self.conn.fetchval(explain_query)

        return self._check_plan_node(result[0]["Plan"], ["Seq Scan"], False)

    async def is_using_parallel_seq_scan(self, query_text: str) -> bool:
        """
        Check if a query uses a parallel sequential scan.

        Args:
            query_text: SQL query to analyze

        Returns:
            True if query uses parallel sequential scan, False otherwise
        """
        explain_query = f"EXPLAIN (FORMAT JSON) {query_text}"
        result = await self.conn.fetchval(explain_query)

        return self._check_plan_node(result[0]["Plan"], ["Seq Scan"], True)

    def _check_plan_node(
        self,
        node: dict,
        node_types: List[str],
        check_parallel: bool = False
    ) -> bool:
        """
        Recursively check plan node for specific node types.

        Args:
            node: Plan node dictionary
            node_types: List of node types to check for
            check_parallel: If True, also verify node is parallel aware

        Returns:
            True if matching node found, False otherwise
        """
        node_type = node.get("Node Type")

        # Check if this is one of the specified node types
        if node_type in node_types:
            if check_parallel:
                return node.get("Parallel Aware", False)
            else:
                return True

        # Recursively check Plans array
        if "Plans" in node:
            for child in node["Plans"]:
                if self._check_plan_node(child, node_types, check_parallel):
                    return True

        return False

    async def directory_exists(self, dir_path: str) -> bool:
        """
        Check if a directory exists in PostgreSQL data directory.

        Args:
            dir_path: Path to directory

        Returns:
            True if directory exists, False otherwise
        """
        try:
            await self.conn.fetch(f"SELECT pg_ls_dir('{dir_path}')")
            return True
        except Exception:
            return False

    async def directory_empty(self, dir_path: str) -> bool:
        """
        Check if a directory is empty.

        Args:
            dir_path: Path to directory

        Returns:
            True if directory is empty, False otherwise
        """
        try:
            files = await self.conn.fetch(f"SELECT pg_ls_dir('{dir_path}')")
            return len(files) == 0
        except Exception:
            return False
