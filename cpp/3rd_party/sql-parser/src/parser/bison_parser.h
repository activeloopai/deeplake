/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_HSQL_BISON_PARSER_H_INCLUDED
# define YY_HSQL_BISON_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef HSQL_DEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define HSQL_DEBUG 1
#  else
#   define HSQL_DEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define HSQL_DEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined HSQL_DEBUG */
#if HSQL_DEBUG
extern int hsql_debug;
#endif
/* "%code requires" blocks.  */
#line 42 "bison_parser.y"

  // clang-format on
  // %code requires block

#include "../SQLParserResult.h"
#include "../sql/statements.h"
#include "parser_typedef.h"

// Platform-specific case-insensitive string comparison
#ifdef _MSC_VER
  #define strcasecmp _stricmp
#endif

// Auto update column and line number
#define YY_USER_ACTION                        \
  yylloc->first_line = yylloc->last_line;     \
  yylloc->first_column = yylloc->last_column; \
  for (int i = 0; yytext[i] != '\0'; i++) {   \
    yylloc->total_column++;                   \
    yylloc->string_length++;                  \
    if (yytext[i] == '\n') {                  \
      yylloc->last_line++;                    \
      yylloc->last_column = 0;                \
    } else {                                  \
      yylloc->last_column++;                  \
    }                                         \
  }

#line 81 "bison_parser.h"

/* Token kinds.  */
#ifndef HSQL_TOKENTYPE
# define HSQL_TOKENTYPE
  enum hsql_tokentype
  {
    SQL_HSQL_EMPTY = -2,
    SQL_YYEOF = 0,                 /* "end of file"  */
    SQL_HSQL_error = 256,          /* error  */
    SQL_HSQL_UNDEF = 257,          /* "invalid token"  */
    SQL_IDENTIFIER = 258,          /* IDENTIFIER  */
    SQL_STRING = 259,              /* STRING  */
    SQL_FLOATVAL = 260,            /* FLOATVAL  */
    SQL_INTVAL = 261,              /* INTVAL  */
    SQL_DEALLOCATE = 262,          /* DEALLOCATE  */
    SQL_PARAMETERS = 263,          /* PARAMETERS  */
    SQL_INTERSECT = 264,           /* INTERSECT  */
    SQL_TEMPORARY = 265,           /* TEMPORARY  */
    SQL_TIMESTAMP = 266,           /* TIMESTAMP  */
    SQL_DISTINCT = 267,            /* DISTINCT  */
    SQL_NVARCHAR = 268,            /* NVARCHAR  */
    SQL_RESTRICT = 269,            /* RESTRICT  */
    SQL_TRUNCATE = 270,            /* TRUNCATE  */
    SQL_ANALYZE = 271,             /* ANALYZE  */
    SQL_BETWEEN = 272,             /* BETWEEN  */
    SQL_CASCADE = 273,             /* CASCADE  */
    SQL_COLUMNS = 274,             /* COLUMNS  */
    SQL_CONTROL = 275,             /* CONTROL  */
    SQL_DEFAULT = 276,             /* DEFAULT  */
    SQL_EXECUTE = 277,             /* EXECUTE  */
    SQL_EXPLAIN = 278,             /* EXPLAIN  */
    SQL_INTEGER = 279,             /* INTEGER  */
    SQL_NATURAL = 280,             /* NATURAL  */
    SQL_PREPARE = 281,             /* PREPARE  */
    SQL_PRIMARY = 282,             /* PRIMARY  */
    SQL_SCHEMAS = 283,             /* SCHEMAS  */
    SQL_CHARACTER_VARYING = 284,   /* CHARACTER_VARYING  */
    SQL_REAL = 285,                /* REAL  */
    SQL_DECIMAL = 286,             /* DECIMAL  */
    SQL_SMALLINT = 287,            /* SMALLINT  */
    SQL_BIGINT = 288,              /* BIGINT  */
    SQL_SPATIAL = 289,             /* SPATIAL  */
    SQL_VARCHAR = 290,             /* VARCHAR  */
    SQL_VIRTUAL = 291,             /* VIRTUAL  */
    SQL_DESCRIBE = 292,            /* DESCRIBE  */
    SQL_BEFORE = 293,              /* BEFORE  */
    SQL_COLUMN = 294,              /* COLUMN  */
    SQL_CREATE = 295,              /* CREATE  */
    SQL_DELETE = 296,              /* DELETE  */
    SQL_DIRECT = 297,              /* DIRECT  */
    SQL_DOUBLE = 298,              /* DOUBLE  */
    SQL_ESCAPE = 299,              /* ESCAPE  */
    SQL_EXCEPT = 300,              /* EXCEPT  */
    SQL_EXISTS = 301,              /* EXISTS  */
    SQL_EXTRACT = 302,             /* EXTRACT  */
    SQL_CAST = 303,                /* CAST  */
    SQL_FORMAT = 304,              /* FORMAT  */
    SQL_GLOBAL = 305,              /* GLOBAL  */
    SQL_HAVING = 306,              /* HAVING  */
    SQL_IMPORT = 307,              /* IMPORT  */
    SQL_INSERT = 308,              /* INSERT  */
    SQL_ISNULL = 309,              /* ISNULL  */
    SQL_OFFSET = 310,              /* OFFSET  */
    SQL_RENAME = 311,              /* RENAME  */
    SQL_SCHEMA = 312,              /* SCHEMA  */
    SQL_SELECT = 313,              /* SELECT  */
    SQL_SORTED = 314,              /* SORTED  */
    SQL_TABLES = 315,              /* TABLES  */
    SQL_UNIQUE = 316,              /* UNIQUE  */
    SQL_UNLOAD = 317,              /* UNLOAD  */
    SQL_UPDATE = 318,              /* UPDATE  */
    SQL_VALUES = 319,              /* VALUES  */
    SQL_AFTER = 320,               /* AFTER  */
    SQL_ALTER = 321,               /* ALTER  */
    SQL_CROSS = 322,               /* CROSS  */
    SQL_DELTA = 323,               /* DELTA  */
    SQL_FLOAT = 324,               /* FLOAT  */
    SQL_SPLIT = 325,               /* SPLIT  */
    SQL_UNGROUP = 326,             /* UNGROUP  */
    SQL_GROUP = 327,               /* GROUP  */
    SQL_INDEX = 328,               /* INDEX  */
    SQL_INNER = 329,               /* INNER  */
    SQL_LIMIT = 330,               /* LIMIT  */
    SQL_LOCAL = 331,               /* LOCAL  */
    SQL_MERGE = 332,               /* MERGE  */
    SQL_MINUS = 333,               /* MINUS  */
    SQL_ORDER = 334,               /* ORDER  */
    SQL_SAMPLE = 335,              /* SAMPLE  */
    SQL_REPLACE = 336,             /* REPLACE  */
    SQL_PERCENT = 337,             /* PERCENT  */
    SQL_OUTER = 338,               /* OUTER  */
    SQL_RIGHT = 339,               /* RIGHT  */
    SQL_TABLE = 340,               /* TABLE  */
    SQL_UNION = 341,               /* UNION  */
    SQL_USING = 342,               /* USING  */
    SQL_WHERE = 343,               /* WHERE  */
    SQL_EXPAND = 344,              /* EXPAND  */
    SQL_OVERLAP = 345,             /* OVERLAP  */
    SQL_CALL = 346,                /* CALL  */
    SQL_CASE = 347,                /* CASE  */
    SQL_CHAR = 348,                /* CHAR  */
    SQL_COPY = 349,                /* COPY  */
    SQL_DATE = 350,                /* DATE  */
    SQL_DATETIME = 351,            /* DATETIME  */
    SQL_DESC = 352,                /* DESC  */
    SQL_DROP = 353,                /* DROP  */
    SQL_ELSE = 354,                /* ELSE  */
    SQL_FILE = 355,                /* FILE  */
    SQL_FROM = 356,                /* FROM  */
    SQL_FULL = 357,                /* FULL  */
    SQL_HASH = 358,                /* HASH  */
    SQL_HINT = 359,                /* HINT  */
    SQL_INTO = 360,                /* INTO  */
    SQL_JOIN = 361,                /* JOIN  */
    SQL_LEFT = 362,                /* LEFT  */
    SQL_LIKE = 363,                /* LIKE  */
    SQL_LOAD = 364,                /* LOAD  */
    SQL_LONG = 365,                /* LONG  */
    SQL_NULL = 366,                /* NULL  */
    SQL_PLAN = 367,                /* PLAN  */
    SQL_SHOW = 368,                /* SHOW  */
    SQL_TEXT_INTERNAL_TQL = 369,   /* TEXT_INTERNAL_TQL  */
    SQL_THEN = 370,                /* THEN  */
    SQL_TIME = 371,                /* TIME  */
    SQL_VIEW = 372,                /* VIEW  */
    SQL_WHEN = 373,                /* WHEN  */
    SQL_WITH = 374,                /* WITH  */
    SQL_ADD = 375,                 /* ADD  */
    SQL_ALL = 376,                 /* ALL  */
    SQL_AND = 377,                 /* AND  */
    SQL_ASC = 378,                 /* ASC  */
    SQL_END = 379,                 /* END  */
    SQL_FOR = 380,                 /* FOR  */
    SQL_INT = 381,                 /* INT  */
    SQL_KEY = 382,                 /* KEY  */
    SQL_NOT = 383,                 /* NOT  */
    SQL_OFF = 384,                 /* OFF  */
    SQL_SET = 385,                 /* SET  */
    SQL_TOP = 386,                 /* TOP  */
    SQL_AS = 387,                  /* AS  */
    SQL_BY = 388,                  /* BY  */
    SQL_IF = 389,                  /* IF  */
    SQL_IN = 390,                  /* IN  */
    SQL_IS = 391,                  /* IS  */
    SQL_OF = 392,                  /* OF  */
    SQL_ON = 393,                  /* ON  */
    SQL_OR = 394,                  /* OR  */
    SQL_TO = 395,                  /* TO  */
    SQL_NO = 396,                  /* NO  */
    SQL_ARRAY = 397,               /* ARRAY  */
    SQL_CONCAT = 398,              /* CONCAT  */
    SQL_ILIKE = 399,               /* ILIKE  */
    SQL_SECONDS = 400,             /* SECONDS  */
    SQL_MINUTES = 401,             /* MINUTES  */
    SQL_HOURS = 402,               /* HOURS  */
    SQL_DAYS = 403,                /* DAYS  */
    SQL_MONTHS = 404,              /* MONTHS  */
    SQL_YEARS = 405,               /* YEARS  */
    SQL_INTERVAL = 406,            /* INTERVAL  */
    SQL_TRUE = 407,                /* TRUE  */
    SQL_FALSE = 408,               /* FALSE  */
    SQL_TRANSACTION = 409,         /* TRANSACTION  */
    SQL_BEGIN = 410,               /* BEGIN  */
    SQL_COMMIT = 411,              /* COMMIT  */
    SQL_ROLLBACK = 412,            /* ROLLBACK  */
    SQL_NOWAIT = 413,              /* NOWAIT  */
    SQL_SKIP = 414,                /* SKIP  */
    SQL_LOCKED = 415,              /* LOCKED  */
    SQL_SHARE = 416,               /* SHARE  */
    SQL_ACROSS = 417,              /* ACROSS  */
    SQL_SPACE = 418,               /* SPACE  */
    SQL_EQUALS = 419,              /* EQUALS  */
    SQL_NOTEQUALS = 420,           /* NOTEQUALS  */
    SQL_LESS = 421,                /* LESS  */
    SQL_GREATER = 422,             /* GREATER  */
    SQL_LESSEQ = 423,              /* LESSEQ  */
    SQL_GREATEREQ = 424,           /* GREATEREQ  */
    SQL_NOTNULL = 425,             /* NOTNULL  */
    SQL_UMINUS = 426               /* UMINUS  */
  };
  typedef enum hsql_tokentype hsql_token_kind_t;
#endif

/* Value type.  */
#if ! defined HSQL_STYPE && ! defined HSQL_STYPE_IS_DECLARED
union HSQL_STYPE
{
#line 102 "bison_parser.y"

  // clang-format on
  bool bval;
  char* sval;
  double fval;
  int64_t ival;
  uintmax_t uval;

  // statements
  hsql::AlterStatement* alter_stmt;
  hsql::CreateStatement* create_stmt;
  hsql::DeleteStatement* delete_stmt;
  hsql::DropStatement* drop_stmt;
  hsql::ExecuteStatement* exec_stmt;
  hsql::ExportStatement* export_stmt;
  hsql::ImportStatement* import_stmt;
  hsql::InsertStatement* insert_stmt;
  hsql::PrepareStatement* prep_stmt;
  hsql::SelectStatement* select_stmt;
  hsql::ShowStatement* show_stmt;
  hsql::SQLStatement* statement;
  hsql::TransactionStatement* transaction_stmt;
  hsql::UpdateStatement* update_stmt;

  hsql::Alias* alias_t;
  hsql::AlterAction* alter_action_t;
  hsql::ColumnDefinition* column_t;
  hsql::ColumnType column_type_t;
  hsql::ConstraintType column_constraint_t;
  hsql::DatetimeField datetime_field;
  hsql::DropColumnAction* drop_action_t;
  hsql::Expr* expr;
  hsql::Expansion* expansion;
  hsql::WhereClause* whereClause;
  hsql::GroupByDescription* group_t;
  hsql::UnGroupByDescription* ungroup_t;
  hsql::ImportType import_type_t;
  hsql::JoinType join_type;
  hsql::LimitDescription* limit;
  hsql::SampleLimitDescription* sample_limit;
  hsql::OrderDescription* order;
  hsql::SampleDescription* sample;
  hsql::OrderType order_type;
  hsql::AcrossType across_type;
  hsql::SetOperation* set_operator_t;
  hsql::TableConstraint* table_constraint_t;
  hsql::TableElement* table_element_t;
  hsql::TableName table_name;
  hsql::TableRef* table;
  hsql::UpdateClause* update_t;
  hsql::WithDescription* with_description_t;
  hsql::LockingClause* locking_t;
  hsql::DistinctDescription* distinct_description_t;

  std::vector<char*>* str_vec;
  std::vector<hsql::ConstraintType>* column_constraint_vec;
  std::vector<hsql::Expr*>* expr_vec;
  std::map<hsql::Expr*, hsql::Expr*>* expr_map;
  std::vector<hsql::OrderDescription*>* order_vec;
  std::vector<hsql::SQLStatement*>* stmt_vec;
  std::vector<hsql::TableElement*>* table_element_vec;
  std::vector<hsql::TableRef*>* table_vec;
  std::vector<hsql::UpdateClause*>* update_vec;
  std::vector<hsql::WithDescription*>* with_description_vec;
  std::vector<hsql::LockingClause*>* locking_clause_vec;

  std::pair<int64_t, int64_t>* ival_pair;

  hsql::RowLockMode lock_mode_t;
  hsql::RowLockWaitPolicy lock_wait_policy_t;

#line 341 "bison_parser.h"

};
typedef union HSQL_STYPE HSQL_STYPE;
# define HSQL_STYPE_IS_TRIVIAL 1
# define HSQL_STYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined HSQL_LTYPE && ! defined HSQL_LTYPE_IS_DECLARED
typedef struct HSQL_LTYPE HSQL_LTYPE;
struct HSQL_LTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define HSQL_LTYPE_IS_DECLARED 1
# define HSQL_LTYPE_IS_TRIVIAL 1
#endif




int hsql_parse (hsql::SQLParserResult* result, yyscan_t scanner);


#endif /* !YY_HSQL_BISON_PARSER_H_INCLUDED  */
