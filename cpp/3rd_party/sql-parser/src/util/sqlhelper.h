#pragma once

#include "../sql/statements.h"

namespace hsql {

// Prints a summary of the given SQLStatement.
void printStatementInfo(std::ostream& s, const SQLStatement* stmt);

// Prints a summary of the given SelectStatement with the given indentation.
void printSelectStatementInfo(std::ostream& s, const SelectStatement* stmt, uintmax_t num_indent);

// Prints a summary of the given ImportStatement with the given indentation.
void printImportStatementInfo(std::ostream& s, const ImportStatement* stmt, uintmax_t num_indent);

// Prints a summary of the given CopyStatement with the given indentation.
void printExportStatementInfo(std::ostream& s, const ExportStatement* stmt, uintmax_t num_indent);

// Prints a summary of the given InsertStatement with the given indentation.
void printInsertStatementInfo(std::ostream& s, const InsertStatement* stmt, uintmax_t num_indent);

// Prints a summary of the given CreateStatement with the given indentation.
void printCreateStatementInfo(std::ostream& s, const CreateStatement* stmt, uintmax_t num_indent);

// Prints a summary of the given TransactionStatement with the given indentation.
void printTransactionStatementInfo(std::ostream& s, const TransactionStatement* stmt, uintmax_t num_indent);

// Prints a summary of the given Expression with the given indentation.
void printExpression(std::ostream& s, const Expr* expr, uintmax_t num_indent);

// Prints an ORDER BY clause
void printOrderBy(std::ostream& s, const std::vector<OrderDescription*>* expr, uintmax_t num_indent);

std::string expression_to_string(const Expr* expr);

}  // namespace hsql
