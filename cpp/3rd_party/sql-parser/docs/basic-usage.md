Using the Library
=======================

Using the SQL parser library is very simple. First step will be to download and build the library. Either get the latest sources from the repository or download the latest release. The only requirement is a modern C++ compiler. Versions that are definitely working are gcc 4.8 and clang 3.4, but older versions might work also or only need small modifications. To build it simply go into the directory and run

```
make         # creates libsqlparser.so
make install # copies the library to /usr/local/lib/
```
To include it in your own code you only need to include one header file: SQLParser.h. The entire framework is wrapped in the namespace hsql. To parse a SQL string you have to call the static method `hsql::SQLParser::parseSQLString(std::string query)`.

The `parseSQLString` method will return an object of type `SQLParserResult*`. When the query was valid SQL the result will contain a list of `SQLStatement` objects that represent the statements in your query. To check whether the query was valid, you can check the `result->isValid` flag. The successfully parsed statements are stored at `result->statements` which is of type `std::vector<SQLStatement*>`.

This is a list of the currently available statement types, each being a subclass of `SQLStatement`:

```
CreateStatement
DeleteStatement
DropStatement
ExecuteStatement
ImportStatement
PrepareStatement
SelectStatement
UpdateStatement
```

To find out what type of statement a certain `SQLStatement` is, you can check the `stmt->type()`, which will return an enum value. This `enum StatementType` is defined in `SQLStatement.h`. There you can see all the available values. Some of these do not match to statement classes though, because they are not implemented yet.

Probably the best way to get familiar with the properties is to look at the class definitions itself in the repository here. The statement definitions are simply structs holding the data from the query. You could also take a look at the utility code in `sqlhelper.cpp` which contains code that prints information about statements to the console.

## Example Code

example.cpp

```
// include the sql parser
#include "SQLParser.h"

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        fprintf(stderr, "Usage: ./example \"SELECT * FROM test;\"\n");
        return -1;
    }
    std::string query = argv[1];

    // parse a given query
    hsql::SQLParserResult* result = hsql::SQLParser::parseSQLString(query);
 
    // check whether the parsing was successful
    if (result->isValid) {
        printf("Parsed successfully!\n");
        printf("Number of statements: %lu\n", result->size());
        // process the statements...
    } else {
        printf("The SQL string is invalid!\n");
        return -1;
    }

    return 0;
}
```

Makefile

```
CFLAGS = -std=c++11 -lstdc++ -Wall -I../src/ -L../

all:
    $(CXX) $(CFLAGS) example.cpp -o example -lsqlparser
```
