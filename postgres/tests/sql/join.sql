\i sql/utils.psql

-- Clean up existing objects
DROP TABLE IF EXISTS dl_employees CASCADE;
DROP TABLE IF EXISTS pg_departments CASCADE;
DROP TABLE IF EXISTS dl_projects CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;

-- Create extension and tables
CREATE EXTENSION pg_deeplake;

-- DeepLake table 1
CREATE TABLE dl_employees (
    emp_id int,
    name text,
    dept_id int,
    salary numeric
) USING deeplake;

-- Regular Postgres table
CREATE TABLE pg_departments (
    dept_id int,
    dept_name text
);

-- DeepLake table 2
CREATE TABLE dl_projects (
    project_id int,
    emp_id int,
    project_name text
) USING deeplake;

DO $$ BEGIN
    -- Insert test data
    INSERT INTO dl_employees VALUES 
        (1, 'John', 1, 50000),
        (2, 'Jane', 2, 60000),
        (3, 'Bob', 1, 55000),
        (4, 'Alice', 3, 65000);    -- Added employee with non-existing dept_id

    INSERT INTO pg_departments VALUES 
        (1, 'Engineering'),
        (2, 'Marketing'),
        (5, 'HR');                 -- Added department with no employees

    INSERT INTO dl_projects VALUES
        (101, 1, 'Project A'),
        (102, 1, 'Project B'),
        (103, 2, 'Project C');

    -- Test 1: JOIN between DeepLake and Postgres tables
    PERFORM assert_query_row_count(
        3,
        'SELECT e.name, d.dept_name 
         FROM dl_employees e 
         JOIN pg_departments d ON e.dept_id = d.dept_id'
    );

    -- Test 2: JOIN between two DeepLake tables
    PERFORM assert_query_row_count(
        3,
        'SELECT e.name, p.project_name 
         FROM dl_employees e 
         JOIN dl_projects p ON e.emp_id = p.emp_id'
    );

    -- Test 3: LEFT JOIN to include all employees
    PERFORM assert_query_row_count(
        4,
        'SELECT e.name, d.dept_name 
         FROM dl_employees e 
         LEFT JOIN pg_departments d ON e.dept_id = d.dept_id'
    );

    -- Test 4: RIGHT JOIN to include all departments
    PERFORM assert_query_row_count(
        4,
        'SELECT e.name, d.dept_name 
         FROM dl_employees e 
         RIGHT JOIN pg_departments d ON e.dept_id = d.dept_id'
    );

    -- Test 5: FULL OUTER JOIN to include all records
    PERFORM assert_query_row_count(
        5,
        'SELECT e.name, d.dept_name 
         FROM dl_employees e 
         FULL OUTER JOIN pg_departments d ON e.dept_id = d.dept_id'
    );

    -- Test 6: CROSS JOIN between DeepLake tables
    PERFORM assert_query_row_count(
        12,  -- 4 employees × 3 projects = 12 rows
        'SELECT e.name, p.project_name 
         FROM dl_employees e 
         CROSS JOIN dl_projects p'
    );

    -- Test 7: Complex LEFT JOIN with multiple conditions
    PERFORM assert_query_row_count(
        5,
        'SELECT e.name, d.dept_name, p.project_name 
         FROM dl_employees e 
         LEFT JOIN pg_departments d ON e.dept_id = d.dept_id
         LEFT JOIN dl_projects p ON e.emp_id = p.emp_id
         ORDER BY e.name'
    );

    -- Verify NULL results in outer joins
    PERFORM assert_query_row_count(
        1,  -- Alice has no matching department
        'SELECT e.name 
         FROM dl_employees e 
         LEFT JOIN pg_departments d ON e.dept_id = d.dept_id 
         WHERE d.dept_name IS NULL'
    );

    RAISE NOTICE 'All JOIN tests passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: JOIN tests failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS dl_employees CASCADE;
DROP TABLE IF EXISTS pg_departments CASCADE;
DROP TABLE IF EXISTS dl_projects CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;