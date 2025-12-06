
#include "sqlhelper.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace hsql {

void printOperatorExpression(Expr* expr, uintmax_t numIndent);
void printAlias(std::ostream& s, Alias* alias, uintmax_t numIndent);

std::ostream& operator<<(std::ostream& os, const OperatorType& op);
std::ostream& operator<<(std::ostream& os, const DatetimeField& op);

std::string indent(uintmax_t numIndent)
{
    return std::string(numIndent, '\t');
}

void inprint(std::ostream& s, int64_t val, uintmax_t numIndent)
{
    s << indent(numIndent).c_str() << val << "  " << std::endl;
}

void inprint(std::ostream& s, double val, uintmax_t numIndent)
{
    s << indent(numIndent).c_str() << val << std::endl;
}

void inprint(std::ostream& s, const char* val, uintmax_t numIndent)
{
    s << indent(numIndent).c_str() << val << std::endl;
}

void inprint(std::ostream& s, const char* val, const char* val2, uintmax_t numIndent)
{
    s << indent(numIndent).c_str() << val << "->" << val2 << std::endl;
}

void inprintC(std::ostream& s, char val, uintmax_t numIndent)
{
    s << indent(numIndent).c_str() << val << std::endl;
}

void inprint(std::ostream& s, const OperatorType& op, uintmax_t numIndent)
{
    s << indent(numIndent) << op << std::endl;
}

void inprint(std::ostream& s, const ColumnType& colType, uintmax_t numIndent)
{
    s << indent(numIndent) << colType << std::endl;
}

void inprint(std::ostream& s, const DatetimeField& colType, uintmax_t numIndent)
{
    s << indent(numIndent) << colType << std::endl;
}

void printTableRefInfo(std::ostream& s, TableRef* table, uintmax_t numIndent)
{
    switch (table->type) {
    case kTableName:
        inprint(s, table->name, numIndent);
        if (table->schema) {
            inprint(s, "Schema", numIndent + 1);
            inprint(s, table->schema, numIndent + 2);
        }
        break;
    case kTableSelect:
        printSelectStatementInfo(s, table->select, numIndent);
        break;
    case kTableJoin:
        inprint(s, "Join Table", numIndent);
        inprint(s, "Left", numIndent + 1);
        printTableRefInfo(s, table->join->left, numIndent + 2);
        inprint(s, "Right", numIndent + 1);
        printTableRefInfo(s, table->join->right, numIndent + 2);
        inprint(s, "Join Condition", numIndent + 1);
        printExpression(s, table->join->condition, numIndent + 2);
        break;
    case kTableCrossProduct:
        for (TableRef* tbl : *table->list) {
            printTableRefInfo(s, tbl, numIndent);
        }
        break;
    }

    if (table->alias) {
        printAlias(s, table->alias, numIndent);
    }
}

void printAlias(std::ostream& s, Alias* alias, uintmax_t numIndent)
{
    inprint(s, "Alias", numIndent + 1);
    inprint(s, alias->name, numIndent + 2);

    if (alias->columns) {
        for (char* column : *(alias->columns)) {
            inprint(s, column, numIndent + 3);
        }
    }
}

void printOperatorExpression(std::ostream& s, const Expr* expr, uintmax_t numIndent)
{
    if (expr == nullptr) {
        inprint(s, "null", numIndent);
        return;
    }

    inprint(s, expr->opType, numIndent);

    printExpression(s, expr->expr, numIndent + 1);
    if (expr->expr2 != nullptr) {
        printExpression(s, expr->expr2, numIndent + 1);
    } else if (expr->exprList != nullptr) {
        for (Expr* e : *expr->exprList) {
            printExpression(s, e, numIndent + 1);
        }
    }
}

void printExpression(std::ostream& s, const Expr* expr, uintmax_t numIndent)
{
    if (!expr) {
        return;
    }
    switch (expr->type) {
    case kExprStar:
        inprint(s, "*", numIndent);
        break;
    case kExprColumnRef:
        inprint(s, expr->name, numIndent);
        if (expr->table) {
            inprint(s, "Table:", numIndent + 1);
            inprint(s, expr->table, numIndent + 2);
        }
        break;
    // case kExprTableColumnRef: inprint(expr->table, expr->name, numIndent); break;
    case kExprLiteralFloat:
        inprint(s, expr->fval, numIndent);
        break;
    case kExprLiteralInt:
        inprint(s, expr->ival, numIndent);
        break;
    case kExprLiteralString:
        inprint(s, expr->name, numIndent);
        break;
    case kExprLiteralDate:
        inprint(s, expr->name, numIndent);
        break;
    case kExprLiteralNull:
        inprint(s, "NULL", numIndent);
        break;
    case kExprLiteralInterval:
        inprint(s, "INTERVAL", numIndent);
        inprint(s, expr->ival, numIndent + 1);
        inprint(s, expr->datetimeField, numIndent + 1);
        break;
    case kExprFunctionRef:
        inprint(s, expr->name, numIndent);
        for (Expr* e : *expr->exprList)
            printExpression(s, e, numIndent + 1);
        break;
    case kExprExtract:
        inprint(s, "EXTRACT", numIndent);
        inprint(s, expr->datetimeField, numIndent + 1);
        printExpression(s, expr->expr, numIndent + 1);
        break;
    case kExprCast:
        inprint(s, "CAST", numIndent);
        inprint(s, expr->columnType, numIndent + 1);
        printExpression(s, expr->expr, numIndent + 1);
        break;
    case kExprOperator:
        printOperatorExpression(s, expr, numIndent);
        break;
    case kExprSelect:
        printSelectStatementInfo(s, expr->select, numIndent);
        break;
    case kExprParameter:
        inprint(s, expr->ival, numIndent);
        break;
    case kExprArray:
        for (Expr* e : *expr->exprList) {
            printExpression(s, e, numIndent + 1);
        }
        break;
    case kExprArrayIndex:
        printExpression(s, expr->expr, numIndent + 1);
        inprint(s, expr->ival, numIndent);
        break;
    case kExprArrayDynamicIndex:
        printExpression(s, expr->expr, numIndent + 1);
        printExpression(s, expr->expr2, numIndent + 1);
        break;
    default:
        std::cerr << "Unrecognized expression type " << expr->type << std::endl;
        return;
    }
    if (expr->alias != nullptr) {
        inprint(s, "Alias", numIndent + 1);
        inprint(s, expr->alias, numIndent + 2);
    }
}

void printOrderBy(std::ostream& s, const std::vector<OrderDescription*>* expr, uintmax_t numIndent)
{
    if (!expr) {
        return;
    }
    for (const auto& order_description : *expr) {
        printExpression(s, order_description->expr, numIndent);
        if (order_description->type == kOrderAsc) {
            inprint(s, "ascending", numIndent);
        } else {
            inprint(s, "descending", numIndent);
        }
    }
}

void printSelectStatementInfo(std::ostream& s, const SelectStatement* stmt, uintmax_t numIndent)
{
    inprint(s, "SelectStatement", numIndent);
    inprint(s, "Fields:", numIndent + 1);
    for (Expr* expr : *stmt->selectList) {
        printExpression(s, expr, numIndent + 2);
    }

    if (stmt->fromTable != nullptr) {
        inprint(s, "Sources:", numIndent + 1);
        printTableRefInfo(s, stmt->fromTable, numIndent + 2);
    }

    if (stmt->whereClause != nullptr) {
        inprint(s, "Search Conditions:", numIndent + 1);
        printExpression(s, stmt->whereClause->expr, numIndent + 2);
    }

    if (stmt->groupBy != nullptr) {
        inprint(s, "GroupBy:", numIndent + 1);
        for (Expr* expr : *stmt->groupBy->columns)
            printExpression(s, expr, numIndent + 2);
        if (stmt->groupBy->having != nullptr) {
            inprint(s, "Having:", numIndent + 1);
            printExpression(s, stmt->groupBy->having, numIndent + 2);
        }
    }
    if (stmt->lockings != nullptr) {
        inprint(s, "Lock Info:", numIndent + 1);
        for (LockingClause* lockingClause : *stmt->lockings) {
            inprint(s, "Type", numIndent + 2);
            if (lockingClause->rowLockMode == RowLockMode::ForUpdate) {
                inprint(s, "FOR UPDATE", numIndent + 3);
            } else if (lockingClause->rowLockMode == RowLockMode::ForNoKeyUpdate) {
                inprint(s, "FOR NO KEY UPDATE", numIndent + 3);
            } else if (lockingClause->rowLockMode == RowLockMode::ForShare) {
                inprint(s, "FOR SHARE", numIndent + 3);
            } else if (lockingClause->rowLockMode == RowLockMode::ForKeyShare) {
                inprint(s, "FOR KEY SHARE", numIndent + 3);
            }
            if (lockingClause->tables != nullptr) {
                inprint(s, "Target tables:", numIndent + 2);
                for (char* dtable : *lockingClause->tables) {
                    inprint(s, dtable, numIndent + 3);
                }
            }
            if (lockingClause->rowLockWaitPolicy != RowLockWaitPolicy::None) {
                inprint(s, "Waiting policy: ", numIndent + 2);
                if (lockingClause->rowLockWaitPolicy == RowLockWaitPolicy::NoWait)
                    inprint(s, "NOWAIT", numIndent + 3);
                else
                    inprint(s, "SKIP LOCKED", numIndent + 3);
            }
        }
    }

    if (stmt->setOperations != nullptr) {
        for (SetOperation* setOperation : *stmt->setOperations) {
            switch (setOperation->setType) {
            case SetType::kSetIntersect:
                inprint(s, "Intersect:", numIndent + 1);
                break;
            case SetType::kSetUnion:
                inprint(s, "Union:", numIndent + 1);
                break;
            case SetType::kSetExcept:
                inprint(s, "Except:", numIndent + 1);
                break;
            }

            printSelectStatementInfo(s, setOperation->nestedSelectStatement, numIndent + 2);

            if (setOperation->resultOrder != nullptr) {
                inprint(s, "SetResultOrderBy:", numIndent + 1);
                printOrderBy(s, setOperation->resultOrder, numIndent + 2);
            }

            if (setOperation->resultLimit != nullptr) {
                if (setOperation->resultLimit->limit != nullptr) {
                    inprint(s, "SetResultLimit:", numIndent + 1);
                    printExpression(s, setOperation->resultLimit->limit, numIndent + 2);
                }

                if (setOperation->resultLimit->offset != nullptr) {
                    inprint(s, "SetResultOffset:", numIndent + 1);
                    printExpression(s, setOperation->resultLimit->offset, numIndent + 2);
                }
            }
        }
    }

    if (stmt->order != nullptr) {
        inprint(s, "OrderBy:", numIndent + 1);
        printOrderBy(s, stmt->order, numIndent + 2);
    }

    if (stmt->limit != nullptr && stmt->limit->limit != nullptr) {
        inprint(s, "Limit:", numIndent + 1);
        printExpression(s, stmt->limit->limit, numIndent + 2);
    }

    if (stmt->limit != nullptr && stmt->limit->offset != nullptr) {
        inprint(s, "Offset:", numIndent + 1);
        printExpression(s, stmt->limit->offset, numIndent + 2);
    }
}

void printImportStatementInfo(std::ostream& s, const ImportStatement* stmt, uintmax_t numIndent)
{
    inprint(s, "ImportStatement", numIndent);
    inprint(s, stmt->filePath, numIndent + 1);
    switch (stmt->type) {
    case ImportType::kImportCSV:
        inprint(s, "CSV", numIndent + 1);
        break;
    case ImportType::kImportTbl:
        inprint(s, "TBL", numIndent + 1);
        break;
    case ImportType::kImportBinary:
        inprint(s, "BINARY", numIndent + 1);
        break;
    case ImportType::kImportAuto:
        inprint(s, "AUTO", numIndent + 1);
        break;
    }
    inprint(s, stmt->tableName, numIndent + 1);
}

void printExportStatementInfo(std::ostream& s, const ExportStatement* stmt, uintmax_t numIndent)
{
    inprint(s, "ExportStatement", numIndent);
    inprint(s, stmt->filePath, numIndent + 1);
    switch (stmt->type) {
    case ImportType::kImportCSV:
        inprint(s, "CSV", numIndent + 1);
        break;
    case ImportType::kImportTbl:
        inprint(s, "TBL", numIndent + 1);
        break;
    case ImportType::kImportBinary:
        inprint(s, "BINARY", numIndent + 1);
        break;
    case ImportType::kImportAuto:
        inprint(s, "AUTO", numIndent + 1);
        break;
    }
    inprint(s, stmt->tableName, numIndent + 1);
}

void printCreateStatementInfo(std::ostream& s, const CreateStatement* stmt, uintmax_t numIndent)
{
    inprint(s, "CreateStatement", numIndent);
    inprint(s, stmt->tableName, numIndent + 1);
    if (stmt->filePath)
        inprint(s, stmt->filePath, numIndent + 1);
}

void printInsertStatementInfo(std::ostream& s, const InsertStatement* stmt, uintmax_t numIndent)
{
    inprint(s, "InsertStatement", numIndent);
    inprint(s, stmt->tableName, numIndent + 1);
    if (stmt->columns != nullptr) {
        inprint(s, "Columns", numIndent + 1);
        for (char* col_name : *stmt->columns) {
            inprint(s, col_name, numIndent + 2);
        }
    }
    switch (stmt->type) {
    case kInsertValues:
        inprint(s, "Values", numIndent + 1);
        for (Expr* expr : *stmt->values) {
            printExpression(s, expr, numIndent + 2);
        }
        break;
    case kInsertSelect:
        printSelectStatementInfo(s, stmt->select, numIndent + 1);
        break;
    }
}

void printTransactionStatementInfo(std::ostream& s, const TransactionStatement* stmt, uintmax_t numIndent)
{
    inprint(s, "TransactionStatement", numIndent);
    switch (stmt->command) {
    case kBeginTransaction:
        inprint(s, "BEGIN", numIndent + 1);
        break;
    case kCommitTransaction:
        inprint(s, "COMMIT", numIndent + 1);
        break;
    case kRollbackTransaction:
        inprint(s, "ROLLBACK", numIndent + 1);
        break;
    }
}

void printStatementInfo(std::ostream& s, const SQLStatement* stmt)
{
    switch (stmt->type()) {
    case kStmtSelect:
        printSelectStatementInfo(s, (const SelectStatement*)stmt, 0);
        break;
    case kStmtInsert:
        printInsertStatementInfo(s, (const InsertStatement*)stmt, 0);
        break;
    case kStmtCreate:
        printCreateStatementInfo(s, (const CreateStatement*)stmt, 0);
        break;
    case kStmtImport:
        printImportStatementInfo(s, (const ImportStatement*)stmt, 0);
        break;
    case kStmtExport:
        printExportStatementInfo(s, (const ExportStatement*)stmt, 0);
        break;
    case kStmtTransaction:
        printTransactionStatementInfo(s, (const TransactionStatement*)stmt, 0);
        break;
    default:
        break;
    }
}

std::ostream& operator<<(std::ostream& os, const OperatorType& op)
{
    static const std::map<const OperatorType, const std::string> operatorToToken = {
        {kOpNone, "None"},     {kOpBetween, "BETWEEN"},
        {kOpCase, "CASE"},     {kOpCaseListElement, "CASE LIST ELEMENT"},
        {kOpPlus, "+"},        {kOpMinus, "-"},
        {kOpAsterisk, "*"},    {kOpSlash, "/"},
        {kOpPercentage, "%"},  {kOpCaret, "^"},
        {kOpEquals, "="},      {kOpNotEquals, "!="},
        {kOpLess, "<"},        {kOpLessEq, "<="},
        {kOpGreater, ">"},     {kOpGreaterEq, ">="},
        {kOpLike, "LIKE"},     {kOpNotLike, "NOT LIKE"},
        {kOpILike, "ILIKE"},   {kOpAnd, "AND"},
        {kOpOr, "OR"},         {kOpIn, "IN"},
        {kOpConcat, "CONCAT"}, {kOpNot, "NOT"},
        {kOpUnaryMinus, "-"},  {kOpIsNull, "IS NULL"},
        {kOpExists, "EXISTS"}};

    const auto found = operatorToToken.find(op);
    if (found == operatorToToken.cend()) {
        return os << static_cast<uint64_t>(op);
    } else {
        return os << (*found).second;
    }
}

std::ostream& operator<<(std::ostream& os, const DatetimeField& datetime)
{
    static const std::map<const DatetimeField, const std::string> operatorToToken = {{kDatetimeNone, "None"},
                                                                                     {kDatetimeSecond, "SECOND"},
                                                                                     {kDatetimeMinute, "MINUTE"},
                                                                                     {kDatetimeHour, "HOUR"},
                                                                                     {kDatetimeDay, "DAY"},
                                                                                     {kDatetimeMonth, "MONTH"},
                                                                                     {kDatetimeYear, "YEAR"}};

    const auto found = operatorToToken.find(datetime);
    if (found == operatorToToken.cend()) {
        return os << static_cast<uint64_t>(datetime);
    } else {
        return os << (*found).second;
    }
}

std::string expression_to_string(const Expr* expr)
{
    std::stringstream ss;
    printExpression(ss, expr, 4);
    return ss.str();
}

} // namespace hsql
