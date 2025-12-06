# Load Tables
CREATE TABLE IF NOT EXISTS students FROM TBL FILE 'test/students.tbl';
CREATE TABLE IF NOT EXISTS test_big FROM TBL FILE 'test/lin_xxs.tbl';
CREATE TABLE IF NOT EXISTS companies FROM TBL FILE 'test/tables/companies.tbl';
CREATE TABLE IF NOT EXISTS employees FROM TBL FILE 'test/tables/employees.tbl';
CREATE TABLE IF NOT EXISTS test (v1 INTEGER, v2 INTEGER, v3 INTEGER);
# SELECT
SELECT * FROM students WHERE grade <= 2.0;
# SUB-SELECT
SELECT name, city, grade FROM (SELECT * FROM students WHERE city = 'Potsdam') t1 WHERE grade <= 1.5 OR grade >= 3.5;
# GROUP
SELECT city, AVG(grade) AS average, MIN(grade) AS best, MAX(grade) AS worst FROM students GROUP BY city;
# UNION
SELECT * FROM students WHERE grade = 1.3 UNION SELECT * FROM students WHERE grade = 3.7;
# JOIN
SELECT * FROM companies JOIN employees ON company_id = employee_company_id;
# INSERT
INSERT INTO students VALUES ('Max Mustermann', 10101, 'Musterhausen', 1.7);
# CREATE
CREATE TABLE IF NOT EXISTS test (v1 INTEGER, v2 INTEGER, v3 INTEGER);
# INSERT/SELECT
INSERT INTO test VALUES (1, 23, 45);
INSERT INTO test VALUES (1, 23, 45);
INSERT INTO test VALUES (1, 23, 45);
SELECT * FROM test;
# CREATE/INSERT/SELECT
CREATE TABLE IF NOT EXISTS test (v1 INTEGER, v2 INTEGER, v3 INTEGER);
INSERT INTO test VALUES (1, 12, 43);
SELECT * FROM test;
# PREPARE 1
PREPARE select_test: SELECT * FROM students WHERE grade = ?;
# EXECUTE 1
EXECUTE select_test(2.0);
# PREPARE 2
PREPARE insert_test {
	INSERT INTO test VALUES (?, 0, 0);
	INSERT INTO test VALUES (?, 0, 0);
	INSERT INTO test VALUES (?, 0, 0);
	INSERT INTO test VALUES (?, 0, 0);
	INSERT INTO test VALUES (?, 0, 0);
};
EXECUTE insert_test(1, 2, 3, 4 ,5);
SELECT * FROM test;
# EXECUTE 2
EXECUTE insert_test(1, 2, 3, 4, 5);
#! GROUP
SELECT AVG(grade) FROM (SELECT city, AVG(grade) FROM students GROUP BY city) t1
#! UNION (kills hyrise)
SELECT name FROM students WHERE grade > 2.0 UNION SELECT name FROM students