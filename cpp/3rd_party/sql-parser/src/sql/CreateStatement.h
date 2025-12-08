#ifndef SQLPARSER_CREATE_STATEMENT_H
#define SQLPARSER_CREATE_STATEMENT_H

#include "ColumnType.h"
#include "SQLStatement.h"

#include <ostream>

// Note: Implementations of constructors and destructors can be found in statements.cpp.
namespace hsql {
struct SelectStatement;

enum struct ConstraintType { None, NotNull, Null, PrimaryKey, Unique };

// Superclass for both TableConstraint and Column Definition
struct TableElement {
  virtual ~TableElement() {}
};

// Represents definition of a table constraint
struct TableConstraint : TableElement {
  TableConstraint(ConstraintType keyType, std::vector<char*>* columnNames);

  ~TableConstraint() override;

  ConstraintType type;
  std::vector<char*>* columnNames;
};

// Represents definition of a table column
struct ColumnDefinition : TableElement {
  ColumnDefinition(char* name, ColumnType type, std::vector<ConstraintType>* column_constraints);

  ~ColumnDefinition() override;

  void setNullableExplicit() {
    nullable = false;

    for (unsigned long constraint_index = 0; constraint_index < column_constraints->size(); constraint_index++) {
      if (column_constraints->at(constraint_index) == ConstraintType::Null) {
        nullable = true;
        column_constraints->erase(column_constraints->cbegin() + constraint_index);
      } else if (column_constraints->at(constraint_index) == ConstraintType::NotNull) {
        column_constraints->erase(column_constraints->cbegin() + constraint_index);
      }
    }
  }

  std::vector<ConstraintType>* column_constraints;
  char* name;
  ColumnType type;
  bool nullable;
};

enum CreateType {
  kCreateTable,
  kCreateTableFromTbl,  // Hyrise file format
  kCreateView,
  kCreateIndex
};

// Represents SQL Create statements.
// Example: "CREATE TABLE students (name TEXT, student_number INTEGER, city TEXT, grade DOUBLE)"
struct CreateStatement : SQLStatement {
  CreateStatement(CreateType type);
  ~CreateStatement() override;

  void setColumnDefsAndConstraints(std::vector<TableElement*>* tableElements) {
    columns = new std::vector<ColumnDefinition*>();
    tableConstraints = new std::vector<TableConstraint*>();

    for (auto tableElem : *tableElements) {
      if (auto* colDef = dynamic_cast<ColumnDefinition*>(tableElem)) {
        columns->emplace_back(colDef);
      } else if (auto* tableConstraint = dynamic_cast<TableConstraint*>(tableElem)) {
        tableConstraints->emplace_back(tableConstraint);
      }
    }
  }

  CreateType type;
  bool ifNotExists;                                 // default: false
  char* filePath;                                   // default: nullptr
  char* schema;                                     // default: nullptr
  char* tableName;                                  // default: nullptr
  char* indexName;                                  // default: nullptr
  std::vector<char*>* indexColumns;                 // default: nullptr
  std::vector<ColumnDefinition*>* columns;          // default: nullptr
  std::vector<TableConstraint*>* tableConstraints;  // default: nullptr
  std::vector<char*>* viewColumns;
  SelectStatement* select;
};

}  // namespace hsql

#endif
