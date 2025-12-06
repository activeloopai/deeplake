Supported SQL Queries
=====================

This page contains a short list of queries that can be correctly parsed with our parser. If you are interested in finding out if a certain feature is supported, it is probably the easiest to checkout the repository and try the example project or check our [list of known limitations](known-limitations.md). Also the file [queries-good.sql](../test/queries/queries-good.sql) shows a list of queries which are parsable with the current version.


## Select Statements

We implement a broad support for the most common elements for `SELECT` statements. Following are a few examples of basic constructs that are supported.

```sql
SELECT name, city, *
    FROM students AS t1 JOIN students AS t2 ON t1.city = t2.city
    WHERE t1.grade < 2.0 AND
          t2.grade > 2.0 AND
          t1.city = 'Frohnau'
    ORDER BY t1.grade DESC;

SELECT city, AVG(grade) AS average,
    MIN(grade) AS best, MAX(grade) AS worst
    FROM students
    GROUP BY city;
```

## Data Definition & Modification

**Create Tables**
```sql
CREATE TABLE students (
    name TEXT,
    student_number INTEGER,
    city TEXT,
    grade DOUBLE
);
```

**Update and Delete**
```sql
UPDATE students SET name='Max Mustermann' WHERE name = 'Ralf Mustermann';

DELETE FROM students WHERE name = 'Max Mustermann';
```


## Prepared Statements

The definition and execution of prepared statements is supported using the following syntax.

```sql
PREPARE select_test FROM 'SELECT * FROM customer WHERE c_name = ?;';

EXECUTE select_test('Max Mustermann');
```
