Known Limitations & Missing Features
====================================

This page contains an overview of known missing limitations and missing features in our SQL parser project. In general, we would like to see all of these features being supported at some point. If you are particularly interested in a specific feature, feel free to contribute to this project through a pull request.

### Completely Missing Statement Types

  * EXPLAIN
  * EXPORT
  * RENAME
  * ALTER

Additionally, there are a lot of statement types that are specific to certain database systems. Supporting all of these is not on our roadmap, but if someone implements support for such a statement, we can also integrate it.

### Other SQL Limitations

 * Tables names ignore the schema name (see grammar rule `table_name`). This affects, for example, `INSERT, IMPORT, DROP, DELETE`.
 * Column data types only support `INT, DOUBLE, TEXT`.
