#ifndef SQLPARSER_EXPORT_STATEMENT_H
#define SQLPARSER_EXPORT_STATEMENT_H

#include "ImportStatement.h"
#include "SQLStatement.h"

namespace hsql {
// Represents SQL Export statements.
struct ExportStatement : SQLStatement {
  ExportStatement(ImportType type);
  ~ExportStatement() override;

  // ImportType is used for compatibility reasons
  ImportType type;
  char* filePath;
  char* schema;
  char* tableName;
};

}  // namespace hsql

#endif
