#ifndef SQLPARSER_SELECT_STATEMENT_H
#define SQLPARSER_SELECT_STATEMENT_H

#include "Expr.h"
#include "SQLStatement.h"
#include "Table.h"

#include <base/base.hpp>

namespace hsql {
enum OrderType { kOrderAsc, kOrderDesc };

enum SetType { kSetUnion, kSetIntersect, kSetExcept };

enum RowLockMode { ForUpdate, ForNoKeyUpdate, ForShare, ForKeyShare };
enum RowLockWaitPolicy { NoWait, SkipLocked, None };

enum AcrossType { Time, Space };

// Description of the order by clause within a select statement.
struct OrderDescription {
  OrderDescription(OrderType type, Expr* expr);
  ~OrderDescription();

  OrderType type;
  Expr* expr;
};

// Description of the limit clause within a select statement.
struct LimitDescription {
  LimitDescription(Expr* limit, Expr* offset);
  ~LimitDescription();

  Expr* limit;
  Expr* offset;
};

// Description of the limit clause within a select statement.
struct SampleLimitDescription {
  SampleLimitDescription(Expr* limit, bool percent);
  ~SampleLimitDescription();

  Expr* limit;
  bool percent;
};

struct SampleDescription {
  SampleDescription(Expr* expr, SampleLimitDescription* limit, bool repeats);
  ~SampleDescription();

  Expr* expr;
  SampleLimitDescription* limit;
  bool repeats;
};

// Description of the group-by clause within a select statement.
struct GroupByDescription {
  GroupByDescription();
  ~GroupByDescription();

  std::vector<Expr*>* columns;
  Expr* having;
  AcrossType across;
};

struct UnGroupByDescription {
  UnGroupByDescription();
  ~UnGroupByDescription();

  Expr* expr = nullptr;
  bool split = false;
};

struct WithDescription {
  ~WithDescription();

  char* alias;
  SelectStatement* select;
};

struct SetOperation {
  SetOperation();
  ~SetOperation();

  SetType setType;
  bool isAll;

  SelectStatement* nestedSelectStatement;
  std::vector<OrderDescription*>* resultOrder;
  LimitDescription* resultLimit;
};

struct LockingClause {
  RowLockMode rowLockMode;
  RowLockWaitPolicy rowLockWaitPolicy;
  std::vector<char*>* tables;
};

struct Expansion {
  Expansion(int back, int forw, Expr* n, Expr* o)
    : name(n ? n->name : "")
    , backward(back)
    , forward(forw)
    , allow_overlap(o->ival)
  {
      base::log_warning(base::log_channel::tql, "EXPAND BY is deprecated and will be removed in a future release.");
      delete n;
      delete o;
  }

  std::string name;
  int backward;
  int forward;
  bool allow_overlap = true;
};

struct WhereClause {
  WhereClause(Expr* a)
    : expr(a)
  {}

  ~WhereClause()
  {
      delete expr;
  }

  Expr* expr;
};

/**
 * @brief Description of DISTINCT clause
 * Represents both simple DISTINCT and DISTINCT ON (columns)
 */
struct DistinctDescription {
    std::vector<Expr*>* distinct_columns = nullptr;

    DistinctDescription() = default;

    ~DistinctDescription() {
        delete distinct_columns;
    }
};

// Representation of a full SQL select statement.
struct SelectStatement : SQLStatement {
  SelectStatement();
  ~SelectStatement() override;

  TableRef* fromTable;
  DistinctDescription* distinct;
  std::vector<Expr*>* selectList;
  WhereClause* whereClause;
  Expansion* expansion;
  GroupByDescription* groupBy;
  UnGroupByDescription* unGroupBy;
  SampleDescription* sampleBy;

  // Note that a SetOperation is always connected to a
  // different SelectStatement. This statement can itself
  // have SetOperation connections to other SelectStatements.
  // To evaluate the operations in the correct order:
  //    Iterate over the setOperations vector:
  //      1. Fully evaluate the nestedSelectStatement within the SetOperation
  //      2. Connect the original statement with the
  //         evaluated nestedSelectStatement
  //      3. Apply the resultOrder and the resultLimit
  //      4. The result now functions as the the original statement
  //         for the next iteration
  //
  // Example:
  //
  //   (SELECT * FROM students INTERSECT SELECT * FROM students_2) UNION SELECT * FROM students_3 ORDER BY grade ASC;
  //
  //   1. We evaluate `Select * FROM students`
  //   2. Then we iterate over the setOperations vector
  //   3. We evalute the nestedSelectStatement of the first entry, which is: `SELECT * FROM students_2`
  //   4. We connect the result of 1. with the results of 3. using the setType, which is INTERSECT
  //   5. We continue the iteration of the setOperations vector
  //   6. We evaluate the new nestedSelectStatement which is: `SELECT * FROM students_3`
  //   7. We apply a Union-Operation to connect the results of 4. and 6.
  //   8. Finally, we apply the resultOrder of the last SetOperation (ORDER BY grade ASC)
  std::vector<SetOperation*>* setOperations;

  std::vector<OrderDescription*>* order;
  std::vector<WithDescription*>* withDescriptions;
  LimitDescription* limit;
  std::vector<LockingClause*>* lockings;

  inline bool selectDistinct() const {
    return distinct != nullptr;
  }
};

}  // namespace hsql

#endif
