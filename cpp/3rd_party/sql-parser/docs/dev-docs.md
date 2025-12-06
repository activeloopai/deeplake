Developer Documentation
=======================

## Basic Requirements

**Requirements for development:**
 * gcc 4.8+ (or clang 3.4+)
 * [bison](https://www.gnu.org/software/bison/) (v3.0.2+)
 * [flex](http://flex.sourceforge.net/) (v2.5.5+)

First step to extending this parser is cloning the repository `git clone git@github.com:hyrise/sql-parser.git` and making sure everything works by running the following steps:

```bash
make parser   # builds the bison parser and flex lexer
make library  # builds the libsqlparser.so
make test     # runs the tests with the library
```

Rerun these steps whenever you change part of the parse. To execute the entire pipeline automatically you can run:

```bash
make cleanall  # cleans the parser build and library build
make test      # build parser, library and runs the tests
```


## Developing New Functionality

This section contains information about how to extend this parser with new functionalities.


### Implementing a new Statement

Create a new file and class in `src/sql/` or extend any of the existing Statements. Every statement needs to have the base class SQLStatement and needs to call its super constructor with its type. If you're defining a new statement type, you need to define a new StatementType in `SQLStatement.h`.

It is important that you create an appropriate constructor for your statement that zero-initializes all its pointer variables and that you create an appropriate destructor.

Finally you will need to include your new file in `src/sql/statements.h`.


### Extending the Grammar

Related files:
```
src/parser/bison_parser.y
src/parser/flex_lexer.l
src/parser/keywordlist_generator.py
src/parser/sql_keywords.txt
```

To extend the grammar the file you will mostly have to deal with is the bison grammar definition in `src/parser/bison_parser.y`.

If you're extending an existing statement, skip to the non-terminal definition for that statement. I.e. for an InsertStatement the non-terminal insert_statement.

If you're defining a new statement, you will need to define your type in the \%union directive `hsql::ExampleStatement example_stmt`. Next you need to associate this type with a non-terminal `\%type <example_stmt> example_statement`. Then you have to define the non-terminal `example_statement`. Look the other non-terminals for statements to figure out how.



## Implementing Tests

All test related files are in `test/`. Take a look to see how tests are implemented.


