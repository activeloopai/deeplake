/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 2

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE         HSQL_STYPE
#define YYLTYPE         HSQL_LTYPE
/* Substitute the variable and function names.  */
#define yyparse         hsql_parse
#define yylex           hsql_lex
#define yyerror         hsql_error
#define yydebug         hsql_debug
#define yynerrs         hsql_nerrs

/* First part of user prologue.  */
#line 2 "bison_parser.y"

  // clang-format on
  /**
 * bison_parser.y
 * defines bison_parser.h
 * outputs bison_parser.c
 *
 * Grammar File Spec: http://dinosaur.compilertools.net/bison/bison_6.html
 *
 */
  /*********************************
 ** Section 1: C Declarations
 *********************************/

#include "bison_parser.h"
#include "flex_lexer.h"

#include <stdio.h>
#include <string.h>

#ifndef YYINITDEPTH
#define YYINITDEPTH 1000
#endif

  using namespace hsql;

  int yyerror(YYLTYPE * llocp, SQLParserResult * result, yyscan_t scanner, const char* msg) {
    result->setIsValid(false);
    result->setErrorDetails(strdup(msg), llocp->first_line, llocp->first_column);
    return 0;
  }
  // clang-format off

#line 112 "bison_parser.cpp"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "bison_parser.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_IDENTIFIER = 3,                 /* IDENTIFIER  */
  YYSYMBOL_STRING = 4,                     /* STRING  */
  YYSYMBOL_FLOATVAL = 5,                   /* FLOATVAL  */
  YYSYMBOL_INTVAL = 6,                     /* INTVAL  */
  YYSYMBOL_DEALLOCATE = 7,                 /* DEALLOCATE  */
  YYSYMBOL_PARAMETERS = 8,                 /* PARAMETERS  */
  YYSYMBOL_INTERSECT = 9,                  /* INTERSECT  */
  YYSYMBOL_TEMPORARY = 10,                 /* TEMPORARY  */
  YYSYMBOL_TIMESTAMP = 11,                 /* TIMESTAMP  */
  YYSYMBOL_DISTINCT = 12,                  /* DISTINCT  */
  YYSYMBOL_NVARCHAR = 13,                  /* NVARCHAR  */
  YYSYMBOL_RESTRICT = 14,                  /* RESTRICT  */
  YYSYMBOL_TRUNCATE = 15,                  /* TRUNCATE  */
  YYSYMBOL_ANALYZE = 16,                   /* ANALYZE  */
  YYSYMBOL_BETWEEN = 17,                   /* BETWEEN  */
  YYSYMBOL_CASCADE = 18,                   /* CASCADE  */
  YYSYMBOL_COLUMNS = 19,                   /* COLUMNS  */
  YYSYMBOL_CONTROL = 20,                   /* CONTROL  */
  YYSYMBOL_DEFAULT = 21,                   /* DEFAULT  */
  YYSYMBOL_EXECUTE = 22,                   /* EXECUTE  */
  YYSYMBOL_EXPLAIN = 23,                   /* EXPLAIN  */
  YYSYMBOL_INTEGER = 24,                   /* INTEGER  */
  YYSYMBOL_NATURAL = 25,                   /* NATURAL  */
  YYSYMBOL_PREPARE = 26,                   /* PREPARE  */
  YYSYMBOL_PRIMARY = 27,                   /* PRIMARY  */
  YYSYMBOL_SCHEMAS = 28,                   /* SCHEMAS  */
  YYSYMBOL_CHARACTER_VARYING = 29,         /* CHARACTER_VARYING  */
  YYSYMBOL_REAL = 30,                      /* REAL  */
  YYSYMBOL_DECIMAL = 31,                   /* DECIMAL  */
  YYSYMBOL_SMALLINT = 32,                  /* SMALLINT  */
  YYSYMBOL_BIGINT = 33,                    /* BIGINT  */
  YYSYMBOL_SPATIAL = 34,                   /* SPATIAL  */
  YYSYMBOL_VARCHAR = 35,                   /* VARCHAR  */
  YYSYMBOL_VIRTUAL = 36,                   /* VIRTUAL  */
  YYSYMBOL_DESCRIBE = 37,                  /* DESCRIBE  */
  YYSYMBOL_BEFORE = 38,                    /* BEFORE  */
  YYSYMBOL_COLUMN = 39,                    /* COLUMN  */
  YYSYMBOL_CREATE = 40,                    /* CREATE  */
  YYSYMBOL_DELETE = 41,                    /* DELETE  */
  YYSYMBOL_DIRECT = 42,                    /* DIRECT  */
  YYSYMBOL_DOUBLE = 43,                    /* DOUBLE  */
  YYSYMBOL_ESCAPE = 44,                    /* ESCAPE  */
  YYSYMBOL_EXCEPT = 45,                    /* EXCEPT  */
  YYSYMBOL_EXISTS = 46,                    /* EXISTS  */
  YYSYMBOL_EXTRACT = 47,                   /* EXTRACT  */
  YYSYMBOL_CAST = 48,                      /* CAST  */
  YYSYMBOL_FORMAT = 49,                    /* FORMAT  */
  YYSYMBOL_GLOBAL = 50,                    /* GLOBAL  */
  YYSYMBOL_HAVING = 51,                    /* HAVING  */
  YYSYMBOL_IMPORT = 52,                    /* IMPORT  */
  YYSYMBOL_INSERT = 53,                    /* INSERT  */
  YYSYMBOL_ISNULL = 54,                    /* ISNULL  */
  YYSYMBOL_OFFSET = 55,                    /* OFFSET  */
  YYSYMBOL_RENAME = 56,                    /* RENAME  */
  YYSYMBOL_SCHEMA = 57,                    /* SCHEMA  */
  YYSYMBOL_SELECT = 58,                    /* SELECT  */
  YYSYMBOL_SORTED = 59,                    /* SORTED  */
  YYSYMBOL_TABLES = 60,                    /* TABLES  */
  YYSYMBOL_UNIQUE = 61,                    /* UNIQUE  */
  YYSYMBOL_UNLOAD = 62,                    /* UNLOAD  */
  YYSYMBOL_UPDATE = 63,                    /* UPDATE  */
  YYSYMBOL_VALUES = 64,                    /* VALUES  */
  YYSYMBOL_AFTER = 65,                     /* AFTER  */
  YYSYMBOL_ALTER = 66,                     /* ALTER  */
  YYSYMBOL_CROSS = 67,                     /* CROSS  */
  YYSYMBOL_DELTA = 68,                     /* DELTA  */
  YYSYMBOL_FLOAT = 69,                     /* FLOAT  */
  YYSYMBOL_SPLIT = 70,                     /* SPLIT  */
  YYSYMBOL_UNGROUP = 71,                   /* UNGROUP  */
  YYSYMBOL_GROUP = 72,                     /* GROUP  */
  YYSYMBOL_INDEX = 73,                     /* INDEX  */
  YYSYMBOL_INNER = 74,                     /* INNER  */
  YYSYMBOL_LIMIT = 75,                     /* LIMIT  */
  YYSYMBOL_LOCAL = 76,                     /* LOCAL  */
  YYSYMBOL_MERGE = 77,                     /* MERGE  */
  YYSYMBOL_MINUS = 78,                     /* MINUS  */
  YYSYMBOL_ORDER = 79,                     /* ORDER  */
  YYSYMBOL_SAMPLE = 80,                    /* SAMPLE  */
  YYSYMBOL_REPLACE = 81,                   /* REPLACE  */
  YYSYMBOL_PERCENT = 82,                   /* PERCENT  */
  YYSYMBOL_OUTER = 83,                     /* OUTER  */
  YYSYMBOL_RIGHT = 84,                     /* RIGHT  */
  YYSYMBOL_TABLE = 85,                     /* TABLE  */
  YYSYMBOL_UNION = 86,                     /* UNION  */
  YYSYMBOL_USING = 87,                     /* USING  */
  YYSYMBOL_WHERE = 88,                     /* WHERE  */
  YYSYMBOL_EXPAND = 89,                    /* EXPAND  */
  YYSYMBOL_OVERLAP = 90,                   /* OVERLAP  */
  YYSYMBOL_CALL = 91,                      /* CALL  */
  YYSYMBOL_CASE = 92,                      /* CASE  */
  YYSYMBOL_CHAR = 93,                      /* CHAR  */
  YYSYMBOL_COPY = 94,                      /* COPY  */
  YYSYMBOL_DATE = 95,                      /* DATE  */
  YYSYMBOL_DATETIME = 96,                  /* DATETIME  */
  YYSYMBOL_DESC = 97,                      /* DESC  */
  YYSYMBOL_DROP = 98,                      /* DROP  */
  YYSYMBOL_ELSE = 99,                      /* ELSE  */
  YYSYMBOL_FILE = 100,                     /* FILE  */
  YYSYMBOL_FROM = 101,                     /* FROM  */
  YYSYMBOL_FULL = 102,                     /* FULL  */
  YYSYMBOL_HASH = 103,                     /* HASH  */
  YYSYMBOL_HINT = 104,                     /* HINT  */
  YYSYMBOL_INTO = 105,                     /* INTO  */
  YYSYMBOL_JOIN = 106,                     /* JOIN  */
  YYSYMBOL_LEFT = 107,                     /* LEFT  */
  YYSYMBOL_LIKE = 108,                     /* LIKE  */
  YYSYMBOL_LOAD = 109,                     /* LOAD  */
  YYSYMBOL_LONG = 110,                     /* LONG  */
  YYSYMBOL_NULL = 111,                     /* NULL  */
  YYSYMBOL_PLAN = 112,                     /* PLAN  */
  YYSYMBOL_SHOW = 113,                     /* SHOW  */
  YYSYMBOL_TEXT_INTERNAL_TQL = 114,        /* TEXT_INTERNAL_TQL  */
  YYSYMBOL_THEN = 115,                     /* THEN  */
  YYSYMBOL_TIME = 116,                     /* TIME  */
  YYSYMBOL_VIEW = 117,                     /* VIEW  */
  YYSYMBOL_WHEN = 118,                     /* WHEN  */
  YYSYMBOL_WITH = 119,                     /* WITH  */
  YYSYMBOL_ADD = 120,                      /* ADD  */
  YYSYMBOL_ALL = 121,                      /* ALL  */
  YYSYMBOL_AND = 122,                      /* AND  */
  YYSYMBOL_ASC = 123,                      /* ASC  */
  YYSYMBOL_END = 124,                      /* END  */
  YYSYMBOL_FOR = 125,                      /* FOR  */
  YYSYMBOL_INT = 126,                      /* INT  */
  YYSYMBOL_KEY = 127,                      /* KEY  */
  YYSYMBOL_NOT = 128,                      /* NOT  */
  YYSYMBOL_OFF = 129,                      /* OFF  */
  YYSYMBOL_SET = 130,                      /* SET  */
  YYSYMBOL_TOP = 131,                      /* TOP  */
  YYSYMBOL_AS = 132,                       /* AS  */
  YYSYMBOL_BY = 133,                       /* BY  */
  YYSYMBOL_IF = 134,                       /* IF  */
  YYSYMBOL_IN = 135,                       /* IN  */
  YYSYMBOL_IS = 136,                       /* IS  */
  YYSYMBOL_OF = 137,                       /* OF  */
  YYSYMBOL_ON = 138,                       /* ON  */
  YYSYMBOL_OR = 139,                       /* OR  */
  YYSYMBOL_TO = 140,                       /* TO  */
  YYSYMBOL_NO = 141,                       /* NO  */
  YYSYMBOL_ARRAY = 142,                    /* ARRAY  */
  YYSYMBOL_CONCAT = 143,                   /* CONCAT  */
  YYSYMBOL_ILIKE = 144,                    /* ILIKE  */
  YYSYMBOL_SECONDS = 145,                  /* SECONDS  */
  YYSYMBOL_MINUTES = 146,                  /* MINUTES  */
  YYSYMBOL_HOURS = 147,                    /* HOURS  */
  YYSYMBOL_DAYS = 148,                     /* DAYS  */
  YYSYMBOL_MONTHS = 149,                   /* MONTHS  */
  YYSYMBOL_YEARS = 150,                    /* YEARS  */
  YYSYMBOL_INTERVAL = 151,                 /* INTERVAL  */
  YYSYMBOL_TRUE = 152,                     /* TRUE  */
  YYSYMBOL_FALSE = 153,                    /* FALSE  */
  YYSYMBOL_TRANSACTION = 154,              /* TRANSACTION  */
  YYSYMBOL_BEGIN = 155,                    /* BEGIN  */
  YYSYMBOL_COMMIT = 156,                   /* COMMIT  */
  YYSYMBOL_ROLLBACK = 157,                 /* ROLLBACK  */
  YYSYMBOL_NOWAIT = 158,                   /* NOWAIT  */
  YYSYMBOL_SKIP = 159,                     /* SKIP  */
  YYSYMBOL_LOCKED = 160,                   /* LOCKED  */
  YYSYMBOL_SHARE = 161,                    /* SHARE  */
  YYSYMBOL_ACROSS = 162,                   /* ACROSS  */
  YYSYMBOL_SPACE = 163,                    /* SPACE  */
  YYSYMBOL_164_ = 164,                     /* '='  */
  YYSYMBOL_EQUALS = 165,                   /* EQUALS  */
  YYSYMBOL_NOTEQUALS = 166,                /* NOTEQUALS  */
  YYSYMBOL_167_ = 167,                     /* '<'  */
  YYSYMBOL_168_ = 168,                     /* '>'  */
  YYSYMBOL_LESS = 169,                     /* LESS  */
  YYSYMBOL_GREATER = 170,                  /* GREATER  */
  YYSYMBOL_LESSEQ = 171,                   /* LESSEQ  */
  YYSYMBOL_GREATEREQ = 172,                /* GREATEREQ  */
  YYSYMBOL_NOTNULL = 173,                  /* NOTNULL  */
  YYSYMBOL_174_ = 174,                     /* '+'  */
  YYSYMBOL_175_ = 175,                     /* '-'  */
  YYSYMBOL_176_ = 176,                     /* '*'  */
  YYSYMBOL_177_ = 177,                     /* '/'  */
  YYSYMBOL_178_ = 178,                     /* '%'  */
  YYSYMBOL_179_ = 179,                     /* '^'  */
  YYSYMBOL_UMINUS = 180,                   /* UMINUS  */
  YYSYMBOL_181_ = 181,                     /* '['  */
  YYSYMBOL_182_ = 182,                     /* ']'  */
  YYSYMBOL_183_ = 183,                     /* '('  */
  YYSYMBOL_184_ = 184,                     /* ')'  */
  YYSYMBOL_185_ = 185,                     /* '.'  */
  YYSYMBOL_186_ = 186,                     /* ';'  */
  YYSYMBOL_187_ = 187,                     /* ','  */
  YYSYMBOL_188_ = 188,                     /* ':'  */
  YYSYMBOL_189_ = 189,                     /* '?'  */
  YYSYMBOL_YYACCEPT = 190,                 /* $accept  */
  YYSYMBOL_input = 191,                    /* input  */
  YYSYMBOL_statement_list = 192,           /* statement_list  */
  YYSYMBOL_statement = 193,                /* statement  */
  YYSYMBOL_preparable_statement = 194,     /* preparable_statement  */
  YYSYMBOL_opt_hints = 195,                /* opt_hints  */
  YYSYMBOL_hint_list = 196,                /* hint_list  */
  YYSYMBOL_hint = 197,                     /* hint  */
  YYSYMBOL_transaction_statement = 198,    /* transaction_statement  */
  YYSYMBOL_opt_transaction_keyword = 199,  /* opt_transaction_keyword  */
  YYSYMBOL_prepare_statement = 200,        /* prepare_statement  */
  YYSYMBOL_prepare_target_query = 201,     /* prepare_target_query  */
  YYSYMBOL_execute_statement = 202,        /* execute_statement  */
  YYSYMBOL_import_statement = 203,         /* import_statement  */
  YYSYMBOL_file_type = 204,                /* file_type  */
  YYSYMBOL_file_path = 205,                /* file_path  */
  YYSYMBOL_opt_file_type = 206,            /* opt_file_type  */
  YYSYMBOL_export_statement = 207,         /* export_statement  */
  YYSYMBOL_show_statement = 208,           /* show_statement  */
  YYSYMBOL_create_statement = 209,         /* create_statement  */
  YYSYMBOL_opt_not_exists = 210,           /* opt_not_exists  */
  YYSYMBOL_table_elem_commalist = 211,     /* table_elem_commalist  */
  YYSYMBOL_table_elem = 212,               /* table_elem  */
  YYSYMBOL_column_def = 213,               /* column_def  */
  YYSYMBOL_column_type = 214,              /* column_type  */
  YYSYMBOL_opt_time_precision = 215,       /* opt_time_precision  */
  YYSYMBOL_opt_decimal_specification = 216, /* opt_decimal_specification  */
  YYSYMBOL_opt_column_constraints = 217,   /* opt_column_constraints  */
  YYSYMBOL_column_constraint_list = 218,   /* column_constraint_list  */
  YYSYMBOL_column_constraint = 219,        /* column_constraint  */
  YYSYMBOL_table_constraint = 220,         /* table_constraint  */
  YYSYMBOL_drop_statement = 221,           /* drop_statement  */
  YYSYMBOL_opt_exists = 222,               /* opt_exists  */
  YYSYMBOL_alter_statement = 223,          /* alter_statement  */
  YYSYMBOL_alter_action = 224,             /* alter_action  */
  YYSYMBOL_drop_action = 225,              /* drop_action  */
  YYSYMBOL_delete_statement = 226,         /* delete_statement  */
  YYSYMBOL_truncate_statement = 227,       /* truncate_statement  */
  YYSYMBOL_insert_statement = 228,         /* insert_statement  */
  YYSYMBOL_opt_column_list = 229,          /* opt_column_list  */
  YYSYMBOL_update_statement = 230,         /* update_statement  */
  YYSYMBOL_update_clause_commalist = 231,  /* update_clause_commalist  */
  YYSYMBOL_update_clause = 232,            /* update_clause  */
  YYSYMBOL_select_statement = 233,         /* select_statement  */
  YYSYMBOL_select_within_set_operation = 234, /* select_within_set_operation  */
  YYSYMBOL_select_within_set_operation_no_parentheses = 235, /* select_within_set_operation_no_parentheses  */
  YYSYMBOL_select_with_paren = 236,        /* select_with_paren  */
  YYSYMBOL_select_no_paren = 237,          /* select_no_paren  */
  YYSYMBOL_set_operator = 238,             /* set_operator  */
  YYSYMBOL_set_type = 239,                 /* set_type  */
  YYSYMBOL_opt_all = 240,                  /* opt_all  */
  YYSYMBOL_select_clause = 241,            /* select_clause  */
  YYSYMBOL_opt_distinct = 242,             /* opt_distinct  */
  YYSYMBOL_select_list = 243,              /* select_list  */
  YYSYMBOL_opt_from_clause = 244,          /* opt_from_clause  */
  YYSYMBOL_from_clause = 245,              /* from_clause  */
  YYSYMBOL_opt_where = 246,                /* opt_where  */
  YYSYMBOL_opt_expand = 247,               /* opt_expand  */
  YYSYMBOL_opt_expand_name = 248,          /* opt_expand_name  */
  YYSYMBOL_opt_expand_overlap = 249,       /* opt_expand_overlap  */
  YYSYMBOL_opt_across = 250,               /* opt_across  */
  YYSYMBOL_opt_group = 251,                /* opt_group  */
  YYSYMBOL_opt_ungroup = 252,              /* opt_ungroup  */
  YYSYMBOL_opt_having = 253,               /* opt_having  */
  YYSYMBOL_opt_sample = 254,               /* opt_sample  */
  YYSYMBOL_sample_desc = 255,              /* sample_desc  */
  YYSYMBOL_opt_order = 256,                /* opt_order  */
  YYSYMBOL_order_list = 257,               /* order_list  */
  YYSYMBOL_order_desc = 258,               /* order_desc  */
  YYSYMBOL_opt_order_type = 259,           /* opt_order_type  */
  YYSYMBOL_opt_top = 260,                  /* opt_top  */
  YYSYMBOL_opt_limit = 261,                /* opt_limit  */
  YYSYMBOL_opt_sample_limit = 262,         /* opt_sample_limit  */
  YYSYMBOL_expr_list = 263,                /* expr_list  */
  YYSYMBOL_expr_pair_list = 264,           /* expr_pair_list  */
  YYSYMBOL_opt_literal_list = 265,         /* opt_literal_list  */
  YYSYMBOL_literal_list = 266,             /* literal_list  */
  YYSYMBOL_expr_alias = 267,               /* expr_alias  */
  YYSYMBOL_expr = 268,                     /* expr  */
  YYSYMBOL_operand = 269,                  /* operand  */
  YYSYMBOL_scalar_expr = 270,              /* scalar_expr  */
  YYSYMBOL_unary_expr = 271,               /* unary_expr  */
  YYSYMBOL_binary_expr = 272,              /* binary_expr  */
  YYSYMBOL_logic_expr = 273,               /* logic_expr  */
  YYSYMBOL_in_expr = 274,                  /* in_expr  */
  YYSYMBOL_case_expr = 275,                /* case_expr  */
  YYSYMBOL_case_list = 276,                /* case_list  */
  YYSYMBOL_exists_expr = 277,              /* exists_expr  */
  YYSYMBOL_comp_expr = 278,                /* comp_expr  */
  YYSYMBOL_function_expr = 279,            /* function_expr  */
  YYSYMBOL_extract_expr = 280,             /* extract_expr  */
  YYSYMBOL_cast_expr = 281,                /* cast_expr  */
  YYSYMBOL_datetime_field = 282,           /* datetime_field  */
  YYSYMBOL_duration_field = 283,           /* duration_field  */
  YYSYMBOL_array_expr = 284,               /* array_expr  */
  YYSYMBOL_array_index = 285,              /* array_index  */
  YYSYMBOL_string_array_index = 286,       /* string_array_index  */
  YYSYMBOL_fancy_array_index = 287,        /* fancy_array_index  */
  YYSYMBOL_dynamic_array_index_operand = 288, /* dynamic_array_index_operand  */
  YYSYMBOL_dynamic_array_index = 289,      /* dynamic_array_index  */
  YYSYMBOL_fancy_array_index_list = 290,   /* fancy_array_index_list  */
  YYSYMBOL_slice_literal = 291,            /* slice_literal  */
  YYSYMBOL_slice_literal_0_0_0 = 292,      /* slice_literal_0_0_0  */
  YYSYMBOL_slice_literal_0_0_1 = 293,      /* slice_literal_0_0_1  */
  YYSYMBOL_slice_literal_0_1_0 = 294,      /* slice_literal_0_1_0  */
  YYSYMBOL_slice_literal_0_1_1 = 295,      /* slice_literal_0_1_1  */
  YYSYMBOL_slice_literal_1_0_0 = 296,      /* slice_literal_1_0_0  */
  YYSYMBOL_slice_literal_1_0_1 = 297,      /* slice_literal_1_0_1  */
  YYSYMBOL_slice_literal_1_1_0 = 298,      /* slice_literal_1_1_0  */
  YYSYMBOL_slice_literal_1_1_1 = 299,      /* slice_literal_1_1_1  */
  YYSYMBOL_between_expr = 300,             /* between_expr  */
  YYSYMBOL_column_name = 301,              /* column_name  */
  YYSYMBOL_literal = 302,                  /* literal  */
  YYSYMBOL_string_literal = 303,           /* string_literal  */
  YYSYMBOL_bool_literal = 304,             /* bool_literal  */
  YYSYMBOL_num_literal = 305,              /* num_literal  */
  YYSYMBOL_int_literal = 306,              /* int_literal  */
  YYSYMBOL_null_literal = 307,             /* null_literal  */
  YYSYMBOL_date_literal = 308,             /* date_literal  */
  YYSYMBOL_interval_literal = 309,         /* interval_literal  */
  YYSYMBOL_param_expr = 310,               /* param_expr  */
  YYSYMBOL_table_ref = 311,                /* table_ref  */
  YYSYMBOL_table_ref_atomic = 312,         /* table_ref_atomic  */
  YYSYMBOL_nonjoin_table_ref_atomic = 313, /* nonjoin_table_ref_atomic  */
  YYSYMBOL_table_ref_commalist = 314,      /* table_ref_commalist  */
  YYSYMBOL_table_ref_name = 315,           /* table_ref_name  */
  YYSYMBOL_table_ref_name_no_alias = 316,  /* table_ref_name_no_alias  */
  YYSYMBOL_table_name = 317,               /* table_name  */
  YYSYMBOL_opt_index_name = 318,           /* opt_index_name  */
  YYSYMBOL_table_alias = 319,              /* table_alias  */
  YYSYMBOL_opt_table_alias = 320,          /* opt_table_alias  */
  YYSYMBOL_alias = 321,                    /* alias  */
  YYSYMBOL_opt_alias = 322,                /* opt_alias  */
  YYSYMBOL_opt_locking_clause = 323,       /* opt_locking_clause  */
  YYSYMBOL_opt_locking_clause_list = 324,  /* opt_locking_clause_list  */
  YYSYMBOL_locking_clause = 325,           /* locking_clause  */
  YYSYMBOL_row_lock_mode = 326,            /* row_lock_mode  */
  YYSYMBOL_opt_row_lock_policy = 327,      /* opt_row_lock_policy  */
  YYSYMBOL_opt_with_clause = 328,          /* opt_with_clause  */
  YYSYMBOL_with_clause = 329,              /* with_clause  */
  YYSYMBOL_with_description_list = 330,    /* with_description_list  */
  YYSYMBOL_with_description = 331,         /* with_description  */
  YYSYMBOL_join_clause = 332,              /* join_clause  */
  YYSYMBOL_opt_join_type = 333,            /* opt_join_type  */
  YYSYMBOL_join_condition = 334,           /* join_condition  */
  YYSYMBOL_opt_semicolon = 335,            /* opt_semicolon  */
  YYSYMBOL_ident_commalist = 336           /* ident_commalist  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if 1

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined HSQL_LTYPE_IS_TRIVIAL && HSQL_LTYPE_IS_TRIVIAL \
             && defined HSQL_STYPE_IS_TRIVIAL && HSQL_STYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  67
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1130

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  190
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  147
/* YYNRULES -- Number of rules.  */
#define YYNRULES  375
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  665

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   426


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,   178,     2,     2,
     183,   184,   176,   174,   187,   175,   185,   177,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,   188,   186,
     167,   164,   168,   189,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   181,     2,   182,   179,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   165,
     166,   169,   170,   171,   172,   173,   180
};

#if HSQL_DEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   328,   328,   347,   353,   360,   364,   368,   369,   370,
     372,   373,   374,   375,   376,   377,   378,   379,   380,   381,
     387,   388,   390,   394,   399,   403,   413,   414,   415,   417,
     417,   423,   429,   431,   435,   446,   452,   459,   474,   479,
     480,   486,   498,   499,   504,   515,   528,   536,   543,   550,
     559,   560,   562,   566,   571,   572,   574,   579,   580,   581,
     582,   583,   584,   588,   589,   590,   591,   592,   593,   594,
     595,   596,   597,   599,   600,   602,   603,   604,   606,   607,
     609,   613,   618,   619,   620,   621,   623,   624,   632,   638,
     644,   650,   656,   657,   664,   670,   672,   682,   689,   700,
     707,   715,   716,   723,   730,   734,   739,   749,   753,   757,
     770,   770,   772,   773,   782,   783,   785,   803,   816,   821,
     825,   829,   834,   835,   837,   850,   853,   857,   861,   863,
     864,   866,   868,   869,   871,   872,   873,   875,   876,   878,
     879,   880,   882,   883,   884,   887,   893,   895,   898,   902,
     904,   905,   907,   908,   910,   911,   912,   913,   915,   916,
     918,   922,   927,   929,   930,   931,   935,   936,   938,   939,
     940,   941,   942,   943,   945,   946,   947,   952,   956,   961,
     965,   970,   971,   973,   977,   982,   990,   990,   990,   990,
     990,   992,   993,   993,   993,   993,   993,   993,   993,   993,
     994,   994,   998,   998,  1000,  1001,  1002,  1003,  1004,  1006,
    1006,  1007,  1008,  1009,  1010,  1011,  1012,  1013,  1014,  1015,
    1017,  1018,  1020,  1021,  1022,  1023,  1027,  1028,  1029,  1030,
    1032,  1033,  1035,  1036,  1038,  1039,  1040,  1041,  1042,  1043,
    1044,  1046,  1047,  1048,  1049,  1051,  1053,  1055,  1056,  1057,
    1058,  1059,  1060,  1062,  1064,  1066,  1066,  1066,  1068,  1069,
    1071,  1071,  1071,  1071,  1071,  1071,  1071,  1071,  1071,  1072,
    1074,  1077,  1082,  1082,  1082,  1082,  1082,  1082,  1082,  1082,
    1084,  1084,  1085,  1086,  1086,  1087,  1088,  1088,  1088,  1089,
    1090,  1090,  1091,  1093,  1095,  1096,  1097,  1098,  1100,  1100,
    1100,  1100,  1100,  1100,  1100,  1102,  1104,  1105,  1107,  1108,
    1110,  1112,  1114,  1125,  1129,  1140,  1172,  1181,  1181,  1188,
    1188,  1190,  1190,  1197,  1201,  1206,  1214,  1220,  1224,  1229,
    1230,  1232,  1232,  1234,  1234,  1236,  1237,  1239,  1239,  1245,
    1246,  1248,  1252,  1257,  1263,  1270,  1271,  1272,  1273,  1275,
    1276,  1277,  1283,  1283,  1285,  1287,  1291,  1296,  1306,  1313,
    1321,  1337,  1338,  1339,  1340,  1341,  1342,  1343,  1344,  1345,
    1346,  1348,  1354,  1354,  1357,  1361
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "IDENTIFIER", "STRING",
  "FLOATVAL", "INTVAL", "DEALLOCATE", "PARAMETERS", "INTERSECT",
  "TEMPORARY", "TIMESTAMP", "DISTINCT", "NVARCHAR", "RESTRICT", "TRUNCATE",
  "ANALYZE", "BETWEEN", "CASCADE", "COLUMNS", "CONTROL", "DEFAULT",
  "EXECUTE", "EXPLAIN", "INTEGER", "NATURAL", "PREPARE", "PRIMARY",
  "SCHEMAS", "CHARACTER_VARYING", "REAL", "DECIMAL", "SMALLINT", "BIGINT",
  "SPATIAL", "VARCHAR", "VIRTUAL", "DESCRIBE", "BEFORE", "COLUMN",
  "CREATE", "DELETE", "DIRECT", "DOUBLE", "ESCAPE", "EXCEPT", "EXISTS",
  "EXTRACT", "CAST", "FORMAT", "GLOBAL", "HAVING", "IMPORT", "INSERT",
  "ISNULL", "OFFSET", "RENAME", "SCHEMA", "SELECT", "SORTED", "TABLES",
  "UNIQUE", "UNLOAD", "UPDATE", "VALUES", "AFTER", "ALTER", "CROSS",
  "DELTA", "FLOAT", "SPLIT", "UNGROUP", "GROUP", "INDEX", "INNER", "LIMIT",
  "LOCAL", "MERGE", "MINUS", "ORDER", "SAMPLE", "REPLACE", "PERCENT",
  "OUTER", "RIGHT", "TABLE", "UNION", "USING", "WHERE", "EXPAND",
  "OVERLAP", "CALL", "CASE", "CHAR", "COPY", "DATE", "DATETIME", "DESC",
  "DROP", "ELSE", "FILE", "FROM", "FULL", "HASH", "HINT", "INTO", "JOIN",
  "LEFT", "LIKE", "LOAD", "LONG", "NULL", "PLAN", "SHOW",
  "TEXT_INTERNAL_TQL", "THEN", "TIME", "VIEW", "WHEN", "WITH", "ADD",
  "ALL", "AND", "ASC", "END", "FOR", "INT", "KEY", "NOT", "OFF", "SET",
  "TOP", "AS", "BY", "IF", "IN", "IS", "OF", "ON", "OR", "TO", "NO",
  "ARRAY", "CONCAT", "ILIKE", "SECONDS", "MINUTES", "HOURS", "DAYS",
  "MONTHS", "YEARS", "INTERVAL", "TRUE", "FALSE", "TRANSACTION", "BEGIN",
  "COMMIT", "ROLLBACK", "NOWAIT", "SKIP", "LOCKED", "SHARE", "ACROSS",
  "SPACE", "'='", "EQUALS", "NOTEQUALS", "'<'", "'>'", "LESS", "GREATER",
  "LESSEQ", "GREATEREQ", "NOTNULL", "'+'", "'-'", "'*'", "'/'", "'%'",
  "'^'", "UMINUS", "'['", "']'", "'('", "')'", "'.'", "';'", "','", "':'",
  "'?'", "$accept", "input", "statement_list", "statement",
  "preparable_statement", "opt_hints", "hint_list", "hint",
  "transaction_statement", "opt_transaction_keyword", "prepare_statement",
  "prepare_target_query", "execute_statement", "import_statement",
  "file_type", "file_path", "opt_file_type", "export_statement",
  "show_statement", "create_statement", "opt_not_exists",
  "table_elem_commalist", "table_elem", "column_def", "column_type",
  "opt_time_precision", "opt_decimal_specification",
  "opt_column_constraints", "column_constraint_list", "column_constraint",
  "table_constraint", "drop_statement", "opt_exists", "alter_statement",
  "alter_action", "drop_action", "delete_statement", "truncate_statement",
  "insert_statement", "opt_column_list", "update_statement",
  "update_clause_commalist", "update_clause", "select_statement",
  "select_within_set_operation",
  "select_within_set_operation_no_parentheses", "select_with_paren",
  "select_no_paren", "set_operator", "set_type", "opt_all",
  "select_clause", "opt_distinct", "select_list", "opt_from_clause",
  "from_clause", "opt_where", "opt_expand", "opt_expand_name",
  "opt_expand_overlap", "opt_across", "opt_group", "opt_ungroup",
  "opt_having", "opt_sample", "sample_desc", "opt_order", "order_list",
  "order_desc", "opt_order_type", "opt_top", "opt_limit",
  "opt_sample_limit", "expr_list", "expr_pair_list", "opt_literal_list",
  "literal_list", "expr_alias", "expr", "operand", "scalar_expr",
  "unary_expr", "binary_expr", "logic_expr", "in_expr", "case_expr",
  "case_list", "exists_expr", "comp_expr", "function_expr", "extract_expr",
  "cast_expr", "datetime_field", "duration_field", "array_expr",
  "array_index", "string_array_index", "fancy_array_index",
  "dynamic_array_index_operand", "dynamic_array_index",
  "fancy_array_index_list", "slice_literal", "slice_literal_0_0_0",
  "slice_literal_0_0_1", "slice_literal_0_1_0", "slice_literal_0_1_1",
  "slice_literal_1_0_0", "slice_literal_1_0_1", "slice_literal_1_1_0",
  "slice_literal_1_1_1", "between_expr", "column_name", "literal",
  "string_literal", "bool_literal", "num_literal", "int_literal",
  "null_literal", "date_literal", "interval_literal", "param_expr",
  "table_ref", "table_ref_atomic", "nonjoin_table_ref_atomic",
  "table_ref_commalist", "table_ref_name", "table_ref_name_no_alias",
  "table_name", "opt_index_name", "table_alias", "opt_table_alias",
  "alias", "opt_alias", "opt_locking_clause", "opt_locking_clause_list",
  "locking_clause", "row_lock_mode", "opt_row_lock_policy",
  "opt_with_clause", "with_clause", "with_description_list",
  "with_description", "join_clause", "opt_join_type", "join_condition",
  "opt_semicolon", "ident_commalist", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-480)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-373)

#define yytable_value_is_error(Yyn) \
  ((Yyn) == YYTABLE_NINF)

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     813,    21,    77,    79,   215,    77,   -20,    85,   127,   149,
      77,   154,    77,    96,    49,   258,   129,   129,   129,   275,
     120,  -480,   200,  -480,   200,  -480,  -480,  -480,  -480,  -480,
    -480,  -480,  -480,  -480,  -480,  -480,  -480,   -22,  -480,   320,
     143,  -480,   162,   249,  -480,   231,   231,   231,    77,   352,
      77,   256,  -480,   265,    -6,   265,   265,   265,    77,  -480,
     266,   217,  -480,  -480,  -480,  -480,  -480,  -480,   774,  -480,
     302,  -480,  -480,   279,   -22,    98,  -480,   151,  -480,   411,
      37,   415,   305,   419,    77,    77,   351,  -480,   334,   257,
     449,   407,    77,   450,   450,   452,    77,    77,  -480,   273,
     258,  -480,   274,   456,   446,   282,   285,  -480,  -480,  -480,
     344,   347,   337,   -22,    15,  -480,  -480,  -480,  -480,   468,
    -480,   472,  -480,  -480,  -480,   295,   291,  -480,  -480,  -480,
    -480,   235,  -480,  -480,  -480,  -480,  -480,  -480,   435,  -480,
     345,   -28,   257,   369,  -480,   450,   481,    13,   324,   -43,
    -480,  -480,   391,   372,  -480,   372,  -480,  -480,  -480,  -480,
    -480,   490,  -480,   356,   369,  -480,  -480,   416,  -480,  -480,
     369,   416,  -480,  -480,    98,   369,   439,   420,  -480,   235,
    -480,    37,  -480,  -480,  -480,  -480,  -480,  -480,  -480,  -480,
    -480,    77,   496,   344,    45,   370,   144,   318,   321,   325,
     173,   327,   550,   322,   601,  -480,   284,   111,   627,  -480,
    -480,  -480,  -480,  -480,  -480,  -480,  -480,  -480,  -480,  -480,
    -480,  -480,  -480,  -480,  -480,  -480,  -480,  -480,   408,  -480,
     134,   329,  -480,   369,   449,  -480,   476,  -480,  -480,   467,
    -480,  -480,   335,   153,  -480,   336,   422,   330,  -480,    32,
      15,   338,  -480,   125,    15,   -22,   111,   -18,    19,   394,
     399,  -480,  -480,  -480,   349,   428,  -480,  1004,   406,   353,
     159,  -480,  -480,  -480,   344,     8,    22,   477,   235,   369,
     369,   -55,   118,   446,   359,   601,   830,   369,   157,   362,
     -73,   369,   369,   601,  -480,   601,   -47,   364,   -32,   601,
     601,   601,   601,   601,   601,   601,   601,   601,   601,   601,
     601,   601,   601,   601,   535,    77,  -480,   545,    37,   111,
    -480,   265,   352,    37,  -480,   490,   369,    24,   351,  -480,
     369,  -480,   546,  -480,  -480,   420,   369,  -480,  -480,  -480,
     420,  -480,   369,   369,   369,     6,  -480,   399,  -480,   481,
     450,  -480,   368,  -480,   374,  -480,  -480,   376,  -480,  -480,
     378,  -480,  -480,  -480,  -480,   379,  -480,    28,   380,   481,
    -480,    45,  -480,  -480,   369,  -480,  -480,   381,   463,   171,
     138,   139,   369,   369,  -480,   369,   477,   458,  -106,  -480,
    -480,  -480,   447,   769,   848,   601,   387,   284,  -480,   460,
     392,   848,   848,   848,   848,   570,   570,   570,   570,   157,
     157,    91,    91,    91,    29,   284,    16,   830,   390,   395,
     396,   397,   398,   404,   405,   412,   413,  -104,  -480,  -480,
    -480,  -480,  -480,  -480,  -480,  -480,  -480,   417,   221,  -480,
    -480,   160,   571,  -480,   177,  -480,   203,   344,  -480,    68,
    -480,   389,  -480,    27,  -480,   500,  -480,  -480,  -480,  -480,
     399,   111,   111,   110,  -480,   432,   473,  -480,   121,  -480,
     204,  -480,   595,   596,  -480,   603,   604,   605,  -480,   485,
    -480,  -480,   502,  -480,    28,  -480,   481,   210,  -480,   234,
     236,   429,  -480,   369,  1004,   369,   369,  -480,   175,   156,
     243,   434,  -480,   601,   848,   284,   436,   244,  -480,    41,
     456,   431,  -480,  -480,    17,  -480,    18,  -480,  -480,  -480,
    -480,   437,   510,  -480,  -480,  -480,   540,   542,   543,   523,
      24,   628,  -480,  -480,  -480,   499,   561,  -480,   369,    30,
    -480,  -480,   572,   481,  -480,   474,  -480,  -480,   453,   245,
     454,   457,   459,  -480,  -480,  -480,   254,  -480,  -480,  -480,
     369,   369,    75,   466,   111,   183,  -480,   369,  -480,  -480,
     830,   469,   263,  -480,  -480,   465,  -480,   456,  -480,   448,
     456,   464,    27,    24,  -480,  -480,  -480,    24,   202,   471,
     634,   518,   584,   113,   582,   582,  -480,  -480,    57,  -480,
    -480,  -480,   652,  -480,  -480,  -480,  -480,   478,  -480,  -480,
    -480,  -480,   111,  -480,  -480,  -480,  -480,   456,  -480,  -480,
     501,   481,    50,   369,   526,  -480,  -480,  -480,  -480,  -480,
     480,   369,  -480,   479,   369,   264,   575,   225,   528,   -19,
     354,  -480,  -480,    23,   111,  -480,  -480,   528,  -480,   664,
    -480,   369,   506,  -480,   111,   484,   486,  -480,  -480,   111,
     -62,  -480,  -480,  -480,  -480
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int16 yydefact[] =
{
     353,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    30,    30,    30,     0,
     373,     3,    21,    19,    21,    18,     8,     9,     7,    11,
      16,    17,    13,    14,    12,    15,    10,     0,   352,     0,
     327,    98,    33,     0,    44,    51,    51,    51,     0,     0,
       0,     0,   326,    93,     0,    93,    93,    93,     0,    42,
       0,   354,   355,    29,    26,    28,    27,     1,   353,     2,
       0,     6,     5,   167,     0,   107,   108,   159,    90,     0,
     182,     0,     0,   330,     0,     0,   133,    37,     0,   102,
       0,     0,     0,     0,     0,     0,     0,     0,    43,     0,
       0,     4,     0,     0,   127,     0,     0,   120,   121,   119,
     353,   123,     0,     0,   173,   328,   305,   308,   310,     0,
     311,     0,   306,   307,   316,     0,   181,   183,   298,   299,
     300,   309,   301,   302,   303,   304,    32,    31,     0,   329,
       0,     0,   102,     0,    97,     0,     0,     0,     0,   133,
     104,    92,     0,    40,    38,    40,    91,    88,    89,   357,
     356,     0,   166,   125,     0,   115,   114,   159,   122,   118,
       0,   159,   111,   110,   112,     0,     0,   153,   312,   315,
      34,     0,   247,   248,   249,   250,   251,   252,   253,   313,
      50,     0,     0,   353,     0,     0,   294,     0,     0,     0,
       0,     0,     0,     0,     0,   296,     0,   132,   186,   193,
     194,   195,   188,   190,   196,   189,   209,   197,   198,   199,
     200,   192,   256,   257,   255,   187,   202,   203,     0,   374,
       0,     0,   100,     0,     0,   103,     0,    94,    95,     0,
      36,    41,    24,     0,    22,     0,   130,   128,   177,   338,
     173,   158,   160,   165,   173,     0,   169,   171,   168,     0,
     340,   152,   314,   184,     0,     0,    47,     0,     0,     0,
       0,    52,    54,    55,   353,   127,     0,     0,     0,     0,
       0,     0,     0,   127,     0,     0,   205,     0,   204,     0,
       0,     0,     0,     0,   206,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   101,     0,     0,   106,
     105,    93,     0,     0,    20,     0,     0,     0,   133,   129,
       0,   336,     0,   337,   185,   153,     0,   164,   163,   162,
     153,   113,     0,     0,     0,     0,   116,   339,   341,     0,
       0,    65,     0,    67,    77,    68,    69,     0,    63,    64,
       0,    60,    61,    66,    70,    74,    57,    79,     0,     0,
      46,     0,    49,   241,     0,   295,   297,     0,     0,     0,
       0,     0,     0,     0,   228,     0,     0,     0,     0,   201,
     191,   220,   221,     0,   216,     0,     0,     0,   207,     0,
     219,   218,   234,   235,   236,   237,   238,   239,   240,   211,
     210,   213,   212,   214,   215,     0,   280,     0,   193,   194,
     195,   196,   197,   198,   199,   200,     0,     0,   270,   272,
     273,   274,   275,   276,   277,   278,   279,   298,   309,    35,
     375,     0,     0,    39,     0,    23,     0,   353,   131,   317,
     319,     0,   321,   334,   320,   136,   178,   335,   109,   161,
     340,   172,   170,   176,   345,     0,     0,   347,   351,   342,
       0,    45,     0,     0,    62,     0,     0,     0,    71,     0,
      83,    84,     0,    56,    78,    80,     0,     0,    53,     0,
       0,   177,   232,     0,     0,     0,     0,   226,     0,     0,
       0,     0,   254,     0,   217,     0,     0,     0,   208,     0,
     281,   283,   269,   259,     0,   258,   287,    99,    96,    25,
     126,     0,     0,   369,   361,   367,   365,   368,   363,     0,
       0,     0,   333,   325,   331,     0,   146,   117,     0,   176,
     155,   348,     0,     0,   350,     0,   343,    48,     0,     0,
       0,     0,     0,    82,    85,    81,     0,    87,   242,   243,
       0,     0,     0,     0,   230,     0,   229,     0,   244,   233,
     293,     0,     0,   224,   222,   191,   282,   284,   271,   286,
     288,   290,   334,     0,   364,   366,   362,     0,   318,   335,
       0,     0,   149,   174,   176,   176,   154,   346,   351,   349,
      59,    76,     0,    72,    58,    73,    86,     0,   179,   245,
     246,   227,   231,   225,   223,   285,   289,   291,   322,   358,
     370,     0,   141,     0,     0,   124,   175,   157,   156,   344,
       0,     0,   292,     0,     0,     0,   141,   140,   138,   151,
       0,    75,   180,     0,   371,   359,   332,   138,   139,     0,
     135,     0,   144,   148,   147,   294,     0,   134,   137,   150,
       0,   145,   360,   142,   143
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -480,  -480,  -480,   606,  -480,   648,  -480,   348,  -480,   294,
    -480,  -480,  -480,  -480,   357,   -88,   520,  -480,  -480,  -480,
     346,  -480,   309,  -480,   182,  -480,  -480,  -480,  -480,   198,
    -480,  -480,   -39,  -480,  -480,  -480,  -480,  -480,  -480,   541,
    -480,  -480,   451,  -108,   440,  -480,     1,   -34,   -38,  -480,
    -480,   -82,    -9,  -480,  -480,  -480,  -141,  -480,    43,    48,
    -480,  -480,  -480,  -480,  -178,  -480,   131,  -480,   355,  -480,
    -480,    42,  -386,  -282,  -480,  -480,   -94,  -315,  -142,  -183,
     375,   383,   384,  -480,  -480,   385,   423,  -480,  -480,   386,
     393,   400,  -133,  -480,   401,  -480,  -480,  -480,  -480,  -480,
    -480,   180,  -480,  -480,  -480,  -480,  -480,  -480,  -480,  -480,
    -480,    62,   -67,   -83,    71,  -480,  -103,  -480,  -480,  -480,
    -480,  -480,  -479,   126,  -480,  -480,  -480,     2,  -480,  -480,
     135,   470,  -480,   260,  -480,   403,  -480,   123,  -480,  -480,
    -480,   616,  -480,  -480,  -480,  -480,  -340
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
       0,    19,    20,    21,    22,    71,   243,   244,    23,    64,
      24,   137,    25,    26,    88,   153,   240,    27,    28,    29,
      83,   270,   271,   272,   367,   478,   474,   483,   484,   485,
     273,    30,    92,    31,   237,   238,    32,    33,    34,   147,
      35,   149,   150,    36,   171,   172,   173,   289,   110,   111,
     169,    77,   164,   246,   328,   329,   144,   536,   650,   638,
     661,   592,   625,   652,   260,   261,   114,   251,   252,   339,
     104,   177,   540,   247,   490,   125,   126,   248,   249,   208,
     209,   210,   211,   212,   213,   214,   282,   215,   216,   217,
     218,   219,   188,   189,   220,   221,   222,   223,   426,   224,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     225,   226,   227,   128,   129,   130,   131,   132,   133,   134,
     135,   448,   449,   450,   451,   452,    51,   453,   140,   532,
     533,   534,   334,   346,   347,   348,   468,   546,    37,    38,
      61,    62,   454,   529,   645,    69,   230
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
     162,   207,   167,    76,    41,   388,   155,    44,   235,   470,
     154,   154,    52,   127,    54,   456,    95,    96,    97,   286,
     163,   288,   118,   118,   118,   375,   655,    40,   253,   487,
     331,   174,   651,   256,   258,   331,    73,   342,    75,   113,
     106,   116,   117,   118,   446,   143,   262,    39,   267,   291,
      86,   588,    89,    45,   663,   479,   636,   228,   281,   491,
      98,   395,   154,   280,   290,    46,   292,   291,    58,   464,
     175,    73,   268,   192,   343,   105,   502,   231,   513,   398,
      40,   330,    42,   514,   292,   266,   141,   142,   396,   480,
     176,   319,   489,   522,   152,    93,   399,    47,   157,   158,
     159,   664,   286,   500,   193,   538,   269,   107,   620,    59,
     393,   390,   394,   232,   263,   507,   400,   401,   402,   403,
     404,   405,   406,   407,   408,   409,   410,   411,   412,   413,
     414,   417,   119,   465,    94,   523,   255,   379,   380,   481,
     637,   291,   524,   108,   234,   378,   556,   466,   120,   391,
     392,   525,   526,   596,   291,   194,   482,   458,   292,   531,
     107,    74,   460,   291,   332,   283,   372,   467,   330,    55,
     527,   292,   299,   174,  -370,   528,   196,   116,   117,   118,
     292,    56,   594,   595,   109,   538,    48,   455,   121,   122,
     123,   539,   373,   264,   253,   626,   108,   291,   376,   205,
     461,   462,   463,   598,   510,   416,   580,   447,   627,   628,
     314,   438,   504,    57,   292,   544,   545,   382,    43,   197,
     198,   199,   337,   572,   441,   575,   124,   522,    49,   444,
     112,   437,   291,   291,   299,   291,   383,   109,   496,    53,
     498,   499,   384,   377,   317,   607,   608,   291,   338,   292,
     292,   127,   292,   495,    50,  -323,   127,   383,   543,   609,
     291,    60,   471,   497,   292,   200,   374,   154,   119,   523,
     313,   567,   314,   509,   385,    67,   524,   292,   291,   544,
     545,   635,   442,    63,   120,   525,   526,   196,   116,   117,
     118,   280,   335,   291,   201,   292,   340,   291,   250,   566,
     299,   202,   254,   494,   527,   291,    68,   611,  -370,   528,
     292,    65,    66,   511,   292,   203,   642,   439,   316,    70,
     570,   317,   292,    78,   121,   122,   123,   275,    79,   276,
     197,   198,   199,   310,   311,   312,   313,   324,   314,   521,
     325,   639,    73,   370,   517,    80,   371,   181,   204,   205,
      81,   562,   501,   564,   565,    87,   206,   196,   116,   117,
     118,   519,   124,   506,   181,    82,   182,   183,   184,   185,
     186,   187,   196,   116,   117,   118,   200,   122,   123,   119,
     182,   183,   184,   185,   186,   187,    90,   520,   547,  -324,
     330,   317,    84,    85,   557,   120,   593,   317,    99,    91,
     197,   198,   199,  -286,   100,   201,   102,   576,  -286,   516,
     103,   579,   202,   581,   115,   197,   198,   199,   558,   136,
     559,   330,   139,   560,   653,   612,   203,   568,   574,   601,
     330,   330,   602,   138,   145,   121,   122,   123,   606,   143,
     146,   317,   196,   116,   117,   118,   200,   614,   646,   119,
     330,   317,   148,   151,   116,   156,    74,   161,   163,   204,
     205,   200,   118,    15,   119,   120,   165,   206,   168,   166,
     170,   571,   178,   124,   615,   201,   179,   616,   181,   180,
     120,   190,   202,   191,   229,   197,   198,   199,   233,   236,
     201,   239,   644,   242,   245,   112,   203,   202,   654,   265,
     259,   277,   274,   287,   278,   121,   122,   123,   279,   659,
     283,   203,   318,   315,   632,   321,   322,   330,   323,   326,
     121,   122,   123,   327,   345,   336,   522,   344,   350,   204,
     205,   200,   349,   368,   119,    73,   369,   206,   196,   116,
     117,   118,   386,   124,   204,   205,   389,   397,   440,   457,
     120,   472,   206,   196,   116,   117,   118,   473,   124,   475,
     257,   476,   477,   486,   493,   492,   395,   202,   523,   291,
     505,   508,  -261,   314,   518,   524,   530,  -262,  -263,  -264,
    -265,   203,   198,   199,   525,   526,  -266,  -267,   633,   535,
     121,   122,   123,   541,  -268,   512,   284,   198,   199,   515,
     542,   548,   549,   527,   196,   116,   117,   118,   528,   550,
     551,   552,   553,   554,   204,   205,   583,   561,   569,   577,
     573,   582,   206,   584,   294,   585,   586,   200,   124,   587,
     119,   589,   590,   591,   599,   597,   516,   600,   603,   634,
     622,   604,   200,   605,   293,   119,   120,  -260,   198,   199,
     610,   623,   617,   613,   621,   624,   201,   538,   630,   640,
     649,   120,   643,   285,   641,   637,   631,   658,   660,   276,
     662,   201,    72,   445,   101,   241,   563,   203,   285,   443,
     488,   294,   555,   195,   647,   320,   121,   122,   123,   418,
     657,   459,   203,   200,   578,   341,   119,   419,   420,   421,
     422,   121,   122,   123,   381,   656,   298,   423,   648,   619,
     204,   205,   120,   299,   424,   425,   160,   618,   415,   333,
     537,   629,   201,   416,   124,   204,   205,     0,     0,   285,
       0,     0,     0,   206,     0,   295,     0,  -373,  -373,   124,
       0,  -373,  -373,   203,   308,   309,   310,   311,   312,   313,
     469,   314,   121,   122,   123,   296,     0,     0,     0,     0,
       0,     0,   297,   298,     0,     0,     0,     0,     0,     0,
     299,   300,     0,     0,  -372,     0,   204,   205,     0,     0,
       0,     1,     0,     0,   206,     0,     0,     0,     0,     2,
     124,   301,   302,   303,   304,   305,     3,     0,   306,   307,
       4,   308,   309,   310,   311,   312,   313,     0,   314,     0,
       0,     5,     0,     0,     6,     7,     0,     0,     0,     0,
       1,     0,     0,   294,     0,     0,     8,     9,     2,     0,
       0,     0,     0,     0,     0,     3,     0,    10,     0,     4,
      11,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       5,     0,     0,     6,     7,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     8,     9,     0,    12,     0,
       0,     0,    13,     0,     0,     0,    10,   295,     0,    11,
       0,     0,     0,     0,   294,     0,     0,    14,     0,     0,
       0,   503,     0,    15,     0,     0,     0,   387,     0,     0,
       0,     0,   294,     0,     0,   298,     0,    12,     0,     0,
       0,    13,   299,   300,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    14,     0,     0,    16,
      17,    18,    15,   301,   302,   303,   304,   305,   295,     0,
     306,   307,     0,   308,   309,   310,   311,   312,   313,     0,
     314,     0,     0,     0,     0,     0,  -373,     0,   387,     0,
       0,     0,     0,     0,     0,     0,   298,     0,    16,    17,
      18,     0,     0,   299,   300,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   298,     0,     0,     0,     0,     0,
       0,   299,  -373,     0,   301,   302,   303,   304,   305,     0,
       0,   306,   307,     0,   308,   309,   310,   311,   312,   313,
       0,   314,  -373,  -373,  -373,   304,   305,     0,     0,   306,
     307,     0,   308,   309,   310,   311,   312,   313,   351,   314,
       0,     0,     0,   352,   353,   354,   355,   356,     0,   357,
       0,     0,     0,     0,     0,     0,     0,   358,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   359,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   360,     0,   361,
     362,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   363,     0,     0,     0,   364,     0,
     365,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     366
};

static const yytype_int16 yycheck[] =
{
     103,   143,   110,    37,     2,   287,    94,     5,   149,   349,
      93,    94,    10,    80,    12,   330,    55,    56,    57,   202,
      12,   204,     6,     6,     6,     3,     3,     3,   170,   369,
       3,   113,    51,   175,   176,     3,    58,    55,    37,    77,
      74,     4,     5,     6,   326,    88,   179,    26,     3,   122,
      48,   530,    50,    73,   116,    27,     6,   145,   200,   374,
      58,   108,   145,   118,   206,    85,   139,   122,    19,    63,
      55,    58,    27,   101,    55,    74,   182,    64,   182,   111,
       3,   187,     3,   187,   139,   193,    84,    85,   135,    61,
      75,   233,   374,    25,    92,   101,   128,   117,    96,    97,
      99,   163,   285,   385,   132,    75,    61,     9,   587,    60,
     293,   184,   295,   147,   181,   397,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,    95,   127,   140,    67,   174,   279,   280,   111,
      90,   122,    74,    45,   187,   278,   486,   141,   111,   291,
     292,    83,    84,   539,   122,   183,   128,   335,   139,   132,
       9,   183,   340,   122,   132,   183,   274,   161,   187,    73,
     102,   139,   143,   255,   106,   107,     3,     4,     5,     6,
     139,    85,   152,   153,    86,    75,   101,   328,   151,   152,
     153,    81,   184,   191,   336,    82,    45,   122,   176,   176,
     342,   343,   344,   543,   188,   188,   188,   183,   594,   595,
     181,   314,   395,   117,   139,   158,   159,    99,     3,    46,
      47,    48,    97,   505,   318,   184,   189,    25,   101,   323,
      79,   314,   122,   122,   143,   122,   118,    86,    99,    85,
     382,   383,   124,   277,   187,   560,   561,   122,   123,   139,
     139,   318,   139,   115,   105,   187,   323,   118,   137,   184,
     122,     3,   350,   124,   139,    92,   275,   350,    95,    67,
     179,   115,   181,   415,   283,     0,    74,   139,   122,   158,
     159,   621,   321,   154,   111,    83,    84,     3,     4,     5,
       6,   118,   250,   122,   121,   139,   254,   122,   167,   124,
     143,   128,   171,   132,   102,   122,   186,   124,   106,   107,
     139,    17,    18,   416,   139,   142,   631,   315,   184,   119,
     503,   187,   139,     3,   151,   152,   153,   183,   185,   185,
      46,    47,    48,   176,   177,   178,   179,   184,   181,   447,
     187,   623,    58,   184,   184,   183,   187,   187,   175,   176,
     101,   493,   386,   495,   496,     3,   183,     3,     4,     5,
       6,   184,   189,   397,   187,   134,   145,   146,   147,   148,
     149,   150,     3,     4,     5,     6,    92,   152,   153,    95,
     145,   146,   147,   148,   149,   150,   130,   184,   184,   187,
     187,   187,    46,    47,   184,   111,   538,   187,   132,   134,
      46,    47,    48,   182,   187,   121,   104,   510,   187,   188,
     131,   514,   128,   516,     3,    46,    47,    48,   184,     4,
     184,   187,     3,   187,    70,   567,   142,   184,   184,   184,
     187,   187,   187,   128,   100,   151,   152,   153,   184,    88,
     183,   187,     3,     4,     5,     6,    92,   184,   184,    95,
     187,   187,     3,    46,     4,     3,   183,   183,    12,   175,
     176,    92,     6,   119,    95,   111,   184,   183,   121,   184,
     133,   505,     4,   189,   577,   121,     4,   580,   187,   184,
     111,    46,   128,   138,     3,    46,    47,    48,   164,    98,
     121,   119,   634,     3,   138,    79,   142,   128,   640,     3,
      80,   183,   132,   181,   183,   151,   152,   153,   183,   651,
     183,   142,   183,   105,   617,    39,    49,   187,   183,   183,
     151,   152,   153,   101,   125,   187,    25,   133,   100,   175,
     176,    92,   183,   127,    95,    58,   183,   183,     3,     4,
       5,     6,   183,   189,   175,   176,   184,   183,     3,     3,
     111,   183,   183,     3,     4,     5,     6,   183,   189,   183,
     121,   183,   183,   183,   101,   184,   108,   128,    67,   122,
     183,   111,   182,   181,     3,    74,   187,   182,   182,   182,
     182,   142,    47,    48,    83,    84,   182,   182,    87,    89,
     151,   152,   153,   161,   182,   182,    46,    47,    48,   182,
     127,     6,     6,   102,     3,     4,     5,     6,   107,     6,
       6,     6,   127,   111,   175,   176,   106,   188,   184,   188,
     184,   184,   183,    83,    54,    83,    83,    92,   189,   106,
      95,     3,   133,    72,   160,    63,   188,   184,   184,   138,
       6,   184,    92,   184,    17,    95,   111,   182,    47,    48,
     184,   133,   188,   184,   183,    71,   121,    75,     6,   133,
     132,   111,   183,   128,   184,    90,   188,     3,   162,   185,
     184,   121,    24,   325,    68,   155,   494,   142,   128,   322,
     371,    54,   484,   142,   636,   234,   151,   152,   153,   314,
     647,   336,   142,    92,   514,   255,    95,   314,   314,   314,
     314,   151,   152,   153,   281,   643,   136,   314,   637,   583,
     175,   176,   111,   143,   314,   314,   100,   582,   183,   249,
     460,   598,   121,   188,   189,   175,   176,    -1,    -1,   128,
      -1,    -1,    -1,   183,    -1,   108,    -1,   167,   168,   189,
      -1,   171,   172,   142,   174,   175,   176,   177,   178,   179,
     347,   181,   151,   152,   153,   128,    -1,    -1,    -1,    -1,
      -1,    -1,   135,   136,    -1,    -1,    -1,    -1,    -1,    -1,
     143,   144,    -1,    -1,     0,    -1,   175,   176,    -1,    -1,
      -1,     7,    -1,    -1,   183,    -1,    -1,    -1,    -1,    15,
     189,   164,   165,   166,   167,   168,    22,    -1,   171,   172,
      26,   174,   175,   176,   177,   178,   179,    -1,   181,    -1,
      -1,    37,    -1,    -1,    40,    41,    -1,    -1,    -1,    -1,
       7,    -1,    -1,    54,    -1,    -1,    52,    53,    15,    -1,
      -1,    -1,    -1,    -1,    -1,    22,    -1,    63,    -1,    26,
      66,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    40,    41,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    52,    53,    -1,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    63,   108,    -1,    66,
      -1,    -1,    -1,    -1,    54,    -1,    -1,   113,    -1,    -1,
      -1,   122,    -1,   119,    -1,    -1,    -1,   128,    -1,    -1,
      -1,    -1,    54,    -1,    -1,   136,    -1,    94,    -1,    -1,
      -1,    98,   143,   144,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   113,    -1,    -1,   155,
     156,   157,   119,   164,   165,   166,   167,   168,   108,    -1,
     171,   172,    -1,   174,   175,   176,   177,   178,   179,    -1,
     181,    -1,    -1,    -1,    -1,    -1,   108,    -1,   128,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   136,    -1,   155,   156,
     157,    -1,    -1,   143,   144,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   136,    -1,    -1,    -1,    -1,    -1,
      -1,   143,   144,    -1,   164,   165,   166,   167,   168,    -1,
      -1,   171,   172,    -1,   174,   175,   176,   177,   178,   179,
      -1,   181,   164,   165,   166,   167,   168,    -1,    -1,   171,
     172,    -1,   174,   175,   176,   177,   178,   179,    24,   181,
      -1,    -1,    -1,    29,    30,    31,    32,    33,    -1,    35,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    69,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    -1,    95,
      96,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   110,    -1,    -1,    -1,   114,    -1,
     116,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     126
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int16 yystos[] =
{
       0,     7,    15,    22,    26,    37,    40,    41,    52,    53,
      63,    66,    94,    98,   113,   119,   155,   156,   157,   191,
     192,   193,   194,   198,   200,   202,   203,   207,   208,   209,
     221,   223,   226,   227,   228,   230,   233,   328,   329,    26,
       3,   317,     3,     3,   317,    73,    85,   117,   101,   101,
     105,   316,   317,    85,   317,    73,    85,   117,    19,    60,
       3,   330,   331,   154,   199,   199,   199,     0,   186,   335,
     119,   195,   195,    58,   183,   236,   237,   241,     3,   185,
     183,   101,   134,   210,   210,   210,   317,     3,   204,   317,
     130,   134,   222,   101,   140,   222,   222,   222,   317,   132,
     187,   193,   104,   131,   260,   236,   237,     9,    45,    86,
     238,   239,    79,   238,   256,     3,     4,     5,     6,    95,
     111,   151,   152,   153,   189,   265,   266,   302,   303,   304,
     305,   306,   307,   308,   309,   310,     4,   201,   128,     3,
     318,   317,   317,    88,   246,   100,   183,   229,     3,   231,
     232,    46,   317,   205,   303,   205,     3,   317,   317,   236,
     331,   183,   306,    12,   242,   184,   184,   233,   121,   240,
     133,   234,   235,   236,   241,    55,    75,   261,     4,     4,
     184,   187,   145,   146,   147,   148,   149,   150,   282,   283,
      46,   138,   101,   132,   183,   229,     3,    46,    47,    48,
      92,   121,   128,   142,   175,   176,   183,   268,   269,   270,
     271,   272,   273,   274,   275,   277,   278,   279,   280,   281,
     284,   285,   286,   287,   289,   300,   301,   302,   205,     3,
     336,    64,   237,   164,   187,   246,    98,   224,   225,   119,
     206,   206,     3,   196,   197,   138,   243,   263,   267,   268,
     256,   257,   258,   268,   256,   238,   268,   121,   268,    80,
     254,   255,   282,   302,   317,     3,   233,     3,    27,    61,
     211,   212,   213,   220,   132,   183,   185,   183,   183,   183,
     118,   268,   276,   183,    46,   128,   269,   181,   269,   237,
     268,   122,   139,    17,    54,   108,   128,   135,   136,   143,
     144,   164,   165,   166,   167,   168,   171,   172,   174,   175,
     176,   177,   178,   179,   181,   105,   184,   187,   183,   268,
     232,    39,    49,   183,   184,   187,   183,   101,   244,   245,
     187,     3,   132,   321,   322,   261,   187,    97,   123,   259,
     261,   234,    55,    55,   133,   125,   323,   324,   325,   183,
     100,    24,    29,    30,    31,    32,    33,    35,    43,    69,
      93,    95,    96,   110,   114,   116,   126,   214,   127,   183,
     184,   187,   233,   184,   242,     3,   176,   237,   282,   268,
     268,   276,    99,   118,   124,   242,   183,   128,   263,   184,
     184,   268,   268,   269,   269,   108,   135,   183,   111,   128,
     269,   269,   269,   269,   269,   269,   269,   269,   269,   269,
     269,   269,   269,   269,   269,   183,   188,   269,   270,   271,
     272,   275,   279,   280,   281,   284,   288,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   303,   306,   317,
       3,   266,   222,   204,   266,   197,   263,   183,   311,   312,
     313,   314,   315,   317,   332,   246,   267,     3,   254,   258,
     254,   268,   268,   268,    63,   127,   141,   161,   326,   325,
     336,   205,   183,   183,   216,   183,   183,   183,   215,    27,
      61,   111,   128,   217,   218,   219,   183,   336,   212,   263,
     264,   267,   184,   101,   132,   115,    99,   124,   268,   268,
     263,   237,   182,   122,   269,   183,   237,   263,   111,   268,
     188,   306,   182,   182,   187,   182,   188,   184,     3,   184,
     184,   233,    25,    67,    74,    83,    84,   102,   107,   333,
     187,   132,   319,   320,   321,    89,   247,   323,    75,    81,
     262,   161,   127,   137,   158,   159,   327,   184,     6,     6,
       6,     6,     6,   127,   111,   219,   336,   184,   184,   184,
     187,   188,   268,   214,   268,   268,   124,   115,   184,   184,
     269,   237,   263,   184,   184,   184,   306,   188,   291,   306,
     188,   306,   184,   106,    83,    83,    83,   106,   312,     3,
     133,    72,   251,   268,   152,   153,   262,    63,   336,   160,
     184,   184,   187,   184,   184,   184,   184,   267,   267,   184,
     184,   124,   268,   184,   184,   306,   306,   188,   320,   313,
     312,   183,     6,   133,    71,   252,    82,   262,   262,   327,
       6,   188,   306,    87,   138,   336,     6,    90,   249,   263,
     133,   184,   267,   183,   268,   334,   184,   249,   304,   132,
     248,    51,   253,    70,   268,     3,   301,   248,     3,   268,
     162,   250,   184,   116,   163
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int16 yyr1[] =
{
       0,   190,   191,   192,   192,   193,   193,   193,   193,   193,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     195,   195,   196,   196,   197,   197,   198,   198,   198,   199,
     199,   200,   201,   202,   202,   203,   203,   204,   205,   206,
     206,   207,   208,   208,   208,   209,   209,   209,   209,   209,
     210,   210,   211,   211,   212,   212,   213,   214,   214,   214,
     214,   214,   214,   214,   214,   214,   214,   214,   214,   214,
     214,   214,   214,   215,   215,   216,   216,   216,   217,   217,
     218,   218,   219,   219,   219,   219,   220,   220,   221,   221,
     221,   221,   222,   222,   223,   224,   225,   226,   227,   228,
     228,   229,   229,   230,   231,   231,   232,   233,   233,   233,
     234,   234,   235,   235,   236,   236,   237,   237,   238,   239,
     239,   239,   240,   240,   241,   242,   242,   242,   243,   244,
     244,   245,   246,   246,   247,   247,   247,   248,   248,   249,
     249,   249,   250,   250,   250,   251,   251,   252,   252,   252,
     253,   253,   254,   254,   255,   255,   255,   255,   256,   256,
     257,   257,   258,   259,   259,   259,   260,   260,   261,   261,
     261,   261,   261,   261,   262,   262,   262,   263,   263,   264,
     264,   265,   265,   266,   266,   267,   268,   268,   268,   268,
     268,   269,   269,   269,   269,   269,   269,   269,   269,   269,
     269,   269,   270,   270,   271,   271,   271,   271,   271,   272,
     272,   272,   272,   272,   272,   272,   272,   272,   272,   272,
     273,   273,   274,   274,   274,   274,   275,   275,   275,   275,
     276,   276,   277,   277,   278,   278,   278,   278,   278,   278,
     278,   279,   279,   279,   279,   280,   281,   282,   282,   282,
     282,   282,   282,   283,   284,   285,   285,   285,   286,   287,
     288,   288,   288,   288,   288,   288,   288,   288,   288,   289,
     290,   290,   291,   291,   291,   291,   291,   291,   291,   291,
     292,   292,   293,   294,   294,   295,   296,   296,   296,   297,
     298,   298,   299,   300,   301,   301,   301,   301,   302,   302,
     302,   302,   302,   302,   302,   303,   304,   304,   305,   305,
     306,   307,   308,   309,   309,   309,   310,   311,   311,   312,
     312,   313,   313,   314,   314,   315,   316,   317,   317,   318,
     318,   319,   319,   320,   320,   321,   321,   322,   322,   323,
     323,   324,   324,   325,   325,   326,   326,   326,   326,   327,
     327,   327,   328,   328,   329,   330,   330,   331,   332,   332,
     332,   333,   333,   333,   333,   333,   333,   333,   333,   333,
     333,   334,   335,   335,   336,   336
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     2,     1,     3,     2,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       5,     0,     1,     3,     1,     4,     2,     2,     2,     1,
       0,     4,     1,     2,     5,     7,     5,     1,     1,     3,
       0,     5,     2,     3,     2,     8,     7,     6,     9,     7,
       3,     0,     1,     3,     1,     1,     3,     1,     4,     4,
       1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     4,     3,     0,     5,     3,     0,     1,     0,
       1,     2,     2,     1,     1,     2,     5,     4,     4,     4,
       3,     4,     2,     0,     5,     1,     4,     4,     2,     8,
       5,     3,     0,     5,     1,     3,     3,     2,     2,     7,
       1,     1,     1,     3,     3,     3,     5,     7,     2,     1,
       1,     1,     1,     0,     9,     1,     5,     0,     1,     1,
       0,     2,     2,     0,     6,     5,     0,     2,     0,     2,
       1,     0,     2,     2,     0,     5,     0,     3,     3,     0,
       2,     0,     1,     0,     5,     4,     6,     6,     3,     0,
       1,     3,     2,     1,     1,     0,     2,     0,     2,     2,
       4,     2,     4,     0,     2,     3,     0,     1,     3,     3,
       5,     1,     0,     1,     3,     2,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     1,     2,     2,     2,     3,     4,     1,
       3,     3,     3,     3,     3,     3,     3,     4,     3,     3,
       3,     3,     5,     6,     5,     6,     4,     6,     3,     5,
       4,     5,     4,     5,     3,     3,     3,     3,     3,     3,
       3,     3,     5,     5,     5,     6,     6,     1,     1,     1,
       1,     1,     1,     1,     4,     1,     1,     1,     4,     4,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     4,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     3,     2,     3,     4,     1,     2,     3,     4,
       3,     4,     5,     5,     1,     3,     1,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     2,     3,     2,     1,     1,     3,     1,
       1,     1,     4,     1,     3,     2,     1,     1,     3,     1,
       0,     1,     5,     1,     0,     2,     1,     1,     0,     1,
       0,     1,     2,     3,     5,     1,     3,     1,     2,     2,
       1,     0,     1,     0,     2,     1,     3,     3,     4,     6,
       8,     1,     2,     1,     2,     1,     2,     1,     1,     1,
       0,     1,     1,     0,     1,     3
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = SQL_HSQL_EMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == SQL_HSQL_EMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (&yylloc, result, scanner, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use SQL_HSQL_error or SQL_HSQL_UNDEF. */
#define YYERRCODE SQL_HSQL_UNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if HSQL_DEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YYLOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YYLOCATION_PRINT

#  if defined YY_LOCATION_PRINT

   /* Temporary convenience wrapper in case some people defined the
      undocumented and private YY_LOCATION_PRINT macros.  */
#   define YYLOCATION_PRINT(File, Loc)  YY_LOCATION_PRINT(File, *(Loc))

#  elif defined HSQL_LTYPE_IS_TRIVIAL && HSQL_LTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
}

#   define YYLOCATION_PRINT  yy_location_print_

    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT(File, Loc)  YYLOCATION_PRINT(File, &(Loc))

#  else

#   define YYLOCATION_PRINT(File, Loc) ((void) 0)
    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT  YYLOCATION_PRINT

#  endif
# endif /* !defined YYLOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location, result, scanner); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, hsql::SQLParserResult* result, yyscan_t scanner)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  YY_USE (yylocationp);
  YY_USE (result);
  YY_USE (scanner);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, hsql::SQLParserResult* result, yyscan_t scanner)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  YYLOCATION_PRINT (yyo, yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yykind, yyvaluep, yylocationp, result, scanner);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                 int yyrule, hsql::SQLParserResult* result, yyscan_t scanner)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)],
                       &(yylsp[(yyi + 1) - (yynrhs)]), result, scanner);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule, result, scanner); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !HSQL_DEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !HSQL_DEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yysymbol_kind_t yytoken;
  YYLTYPE *yylloc;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  int yyn = yypact[+*yyctx->yyssp];
  if (!yypact_value_is_default (yyn))
    {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;
      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYSYMBOL_YYerror
            && !yytable_value_is_error (yytable[yyx + yyn]))
          {
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = YY_CAST (yysymbol_kind_t, yyx);
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = YYSYMBOL_YYEMPTY;
  return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
# endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
      char const *yyp = yystr;
      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
#endif


static int
yy_syntax_error_arguments (const yypcontext_t *yyctx,
                           yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yyctx->yytoken != YYSYMBOL_YYEMPTY)
    {
      int yyn;
      if (yyarg)
        yyarg[yycount] = yyctx->yytoken;
      ++yycount;
      yyn = yypcontext_expected_tokens (yyctx,
                                        yyarg ? yyarg + 1 : yyarg, yyargn - 1);
      if (yyn == YYENOMEM)
        return YYENOMEM;
      else
        yycount += yyn;
    }
  return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                const yypcontext_t *yyctx)
{
  enum { YYARGS_MAX = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  yysymbol_kind_t yyarg[YYARGS_MAX];
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* Actual size of YYARG. */
  int yycount = yy_syntax_error_arguments (yyctx, yyarg, YYARGS_MAX);
  if (yycount == YYENOMEM)
    return YYENOMEM;

  switch (yycount)
    {
#define YYCASE_(N, S)                       \
      case N:                               \
        yyformat = S;                       \
        break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  /* Compute error message size.  Don't count the "%s"s, but reserve
     room for the terminator.  */
  yysize = yystrlen (yyformat) - 2 * yycount + 1;
  {
    int yyi;
    for (yyi = 0; yyi < yycount; ++yyi)
      {
        YYPTRDIFF_T yysize1
          = yysize + yytnamerr (YY_NULLPTR, yytname[yyarg[yyi]]);
        if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
          yysize = yysize1;
        else
          return YYENOMEM;
      }
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return -1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yytname[yyarg[yyi++]]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp, hsql::SQLParserResult* result, yyscan_t scanner)
{
  YY_USE (yyvaluep);
  YY_USE (yylocationp);
  YY_USE (result);
  YY_USE (scanner);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yykind)
    {
    case YYSYMBOL_IDENTIFIER: /* IDENTIFIER  */
#line 180 "bison_parser.y"
                { free( (((*yyvaluep).sval)) ); }
#line 2170 "bison_parser.cpp"
        break;

    case YYSYMBOL_STRING: /* STRING  */
#line 180 "bison_parser.y"
                { free( (((*yyvaluep).sval)) ); }
#line 2176 "bison_parser.cpp"
        break;

    case YYSYMBOL_FLOATVAL: /* FLOATVAL  */
#line 178 "bison_parser.y"
                { }
#line 2182 "bison_parser.cpp"
        break;

    case YYSYMBOL_INTVAL: /* INTVAL  */
#line 178 "bison_parser.y"
                { }
#line 2188 "bison_parser.cpp"
        break;

    case YYSYMBOL_statement_list: /* statement_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).stmt_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).stmt_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).stmt_vec));
    }
#line 2201 "bison_parser.cpp"
        break;

    case YYSYMBOL_statement: /* statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).statement)); }
#line 2207 "bison_parser.cpp"
        break;

    case YYSYMBOL_preparable_statement: /* preparable_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).statement)); }
#line 2213 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_hints: /* opt_hints  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2226 "bison_parser.cpp"
        break;

    case YYSYMBOL_hint_list: /* hint_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2239 "bison_parser.cpp"
        break;

    case YYSYMBOL_hint: /* hint  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2245 "bison_parser.cpp"
        break;

    case YYSYMBOL_transaction_statement: /* transaction_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).transaction_stmt)); }
#line 2251 "bison_parser.cpp"
        break;

    case YYSYMBOL_prepare_statement: /* prepare_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).prep_stmt)); }
#line 2257 "bison_parser.cpp"
        break;

    case YYSYMBOL_prepare_target_query: /* prepare_target_query  */
#line 180 "bison_parser.y"
                { free( (((*yyvaluep).sval)) ); }
#line 2263 "bison_parser.cpp"
        break;

    case YYSYMBOL_execute_statement: /* execute_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).exec_stmt)); }
#line 2269 "bison_parser.cpp"
        break;

    case YYSYMBOL_import_statement: /* import_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).import_stmt)); }
#line 2275 "bison_parser.cpp"
        break;

    case YYSYMBOL_file_type: /* file_type  */
#line 178 "bison_parser.y"
                { }
#line 2281 "bison_parser.cpp"
        break;

    case YYSYMBOL_file_path: /* file_path  */
#line 180 "bison_parser.y"
                { free( (((*yyvaluep).sval)) ); }
#line 2287 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_file_type: /* opt_file_type  */
#line 178 "bison_parser.y"
                { }
#line 2293 "bison_parser.cpp"
        break;

    case YYSYMBOL_export_statement: /* export_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).export_stmt)); }
#line 2299 "bison_parser.cpp"
        break;

    case YYSYMBOL_show_statement: /* show_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).show_stmt)); }
#line 2305 "bison_parser.cpp"
        break;

    case YYSYMBOL_create_statement: /* create_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).create_stmt)); }
#line 2311 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_not_exists: /* opt_not_exists  */
#line 178 "bison_parser.y"
                { }
#line 2317 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_elem_commalist: /* table_elem_commalist  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).table_element_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).table_element_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).table_element_vec));
    }
#line 2330 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_elem: /* table_elem  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table_element_t)); }
#line 2336 "bison_parser.cpp"
        break;

    case YYSYMBOL_column_def: /* column_def  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).column_t)); }
#line 2342 "bison_parser.cpp"
        break;

    case YYSYMBOL_column_type: /* column_type  */
#line 178 "bison_parser.y"
                { }
#line 2348 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_time_precision: /* opt_time_precision  */
#line 178 "bison_parser.y"
                { }
#line 2354 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_decimal_specification: /* opt_decimal_specification  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).ival_pair)); }
#line 2360 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_column_constraints: /* opt_column_constraints  */
#line 178 "bison_parser.y"
                { }
#line 2366 "bison_parser.cpp"
        break;

    case YYSYMBOL_column_constraint_list: /* column_constraint_list  */
#line 178 "bison_parser.y"
                { }
#line 2372 "bison_parser.cpp"
        break;

    case YYSYMBOL_column_constraint: /* column_constraint  */
#line 178 "bison_parser.y"
                { }
#line 2378 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_constraint: /* table_constraint  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table_constraint_t)); }
#line 2384 "bison_parser.cpp"
        break;

    case YYSYMBOL_drop_statement: /* drop_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).drop_stmt)); }
#line 2390 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_exists: /* opt_exists  */
#line 178 "bison_parser.y"
                { }
#line 2396 "bison_parser.cpp"
        break;

    case YYSYMBOL_alter_statement: /* alter_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).alter_stmt)); }
#line 2402 "bison_parser.cpp"
        break;

    case YYSYMBOL_alter_action: /* alter_action  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).alter_action_t)); }
#line 2408 "bison_parser.cpp"
        break;

    case YYSYMBOL_drop_action: /* drop_action  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).drop_action_t)); }
#line 2414 "bison_parser.cpp"
        break;

    case YYSYMBOL_delete_statement: /* delete_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).delete_stmt)); }
#line 2420 "bison_parser.cpp"
        break;

    case YYSYMBOL_truncate_statement: /* truncate_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).delete_stmt)); }
#line 2426 "bison_parser.cpp"
        break;

    case YYSYMBOL_insert_statement: /* insert_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).insert_stmt)); }
#line 2432 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_column_list: /* opt_column_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).str_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).str_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).str_vec));
    }
#line 2445 "bison_parser.cpp"
        break;

    case YYSYMBOL_update_statement: /* update_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).update_stmt)); }
#line 2451 "bison_parser.cpp"
        break;

    case YYSYMBOL_update_clause_commalist: /* update_clause_commalist  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).update_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).update_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).update_vec));
    }
#line 2464 "bison_parser.cpp"
        break;

    case YYSYMBOL_update_clause: /* update_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).update_t)); }
#line 2470 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_statement: /* select_statement  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).select_stmt)); }
#line 2476 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_within_set_operation: /* select_within_set_operation  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).select_stmt)); }
#line 2482 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_within_set_operation_no_parentheses: /* select_within_set_operation_no_parentheses  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).select_stmt)); }
#line 2488 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_with_paren: /* select_with_paren  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).select_stmt)); }
#line 2494 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_no_paren: /* select_no_paren  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).select_stmt)); }
#line 2500 "bison_parser.cpp"
        break;

    case YYSYMBOL_set_operator: /* set_operator  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).set_operator_t)); }
#line 2506 "bison_parser.cpp"
        break;

    case YYSYMBOL_set_type: /* set_type  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).set_operator_t)); }
#line 2512 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_all: /* opt_all  */
#line 178 "bison_parser.y"
                { }
#line 2518 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_clause: /* select_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).select_stmt)); }
#line 2524 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_distinct: /* opt_distinct  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).distinct_description_t)); }
#line 2530 "bison_parser.cpp"
        break;

    case YYSYMBOL_select_list: /* select_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2543 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_from_clause: /* opt_from_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 2549 "bison_parser.cpp"
        break;

    case YYSYMBOL_from_clause: /* from_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 2555 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_where: /* opt_where  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).whereClause)); }
#line 2561 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_expand: /* opt_expand  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expansion)); }
#line 2567 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_expand_name: /* opt_expand_name  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2573 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_expand_overlap: /* opt_expand_overlap  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2579 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_across: /* opt_across  */
#line 178 "bison_parser.y"
                { }
#line 2585 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_group: /* opt_group  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).group_t)); }
#line 2591 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_ungroup: /* opt_ungroup  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).ungroup_t)); }
#line 2597 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_having: /* opt_having  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2603 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_sample: /* opt_sample  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).sample)); }
#line 2609 "bison_parser.cpp"
        break;

    case YYSYMBOL_sample_desc: /* sample_desc  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).sample)); }
#line 2615 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_order: /* opt_order  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).order_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).order_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).order_vec));
    }
#line 2628 "bison_parser.cpp"
        break;

    case YYSYMBOL_order_list: /* order_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).order_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).order_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).order_vec));
    }
#line 2641 "bison_parser.cpp"
        break;

    case YYSYMBOL_order_desc: /* order_desc  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).order)); }
#line 2647 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_order_type: /* opt_order_type  */
#line 178 "bison_parser.y"
                { }
#line 2653 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_top: /* opt_top  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).limit)); }
#line 2659 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_limit: /* opt_limit  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).limit)); }
#line 2665 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_sample_limit: /* opt_sample_limit  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).sample_limit)); }
#line 2671 "bison_parser.cpp"
        break;

    case YYSYMBOL_expr_list: /* expr_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2684 "bison_parser.cpp"
        break;

    case YYSYMBOL_expr_pair_list: /* expr_pair_list  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr_map)); }
#line 2690 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_literal_list: /* opt_literal_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2703 "bison_parser.cpp"
        break;

    case YYSYMBOL_literal_list: /* literal_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2716 "bison_parser.cpp"
        break;

    case YYSYMBOL_expr_alias: /* expr_alias  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2722 "bison_parser.cpp"
        break;

    case YYSYMBOL_expr: /* expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2728 "bison_parser.cpp"
        break;

    case YYSYMBOL_operand: /* operand  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2734 "bison_parser.cpp"
        break;

    case YYSYMBOL_scalar_expr: /* scalar_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2740 "bison_parser.cpp"
        break;

    case YYSYMBOL_unary_expr: /* unary_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2746 "bison_parser.cpp"
        break;

    case YYSYMBOL_binary_expr: /* binary_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2752 "bison_parser.cpp"
        break;

    case YYSYMBOL_logic_expr: /* logic_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2758 "bison_parser.cpp"
        break;

    case YYSYMBOL_in_expr: /* in_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2764 "bison_parser.cpp"
        break;

    case YYSYMBOL_case_expr: /* case_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2770 "bison_parser.cpp"
        break;

    case YYSYMBOL_case_list: /* case_list  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2776 "bison_parser.cpp"
        break;

    case YYSYMBOL_exists_expr: /* exists_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2782 "bison_parser.cpp"
        break;

    case YYSYMBOL_comp_expr: /* comp_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2788 "bison_parser.cpp"
        break;

    case YYSYMBOL_function_expr: /* function_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2794 "bison_parser.cpp"
        break;

    case YYSYMBOL_extract_expr: /* extract_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2800 "bison_parser.cpp"
        break;

    case YYSYMBOL_cast_expr: /* cast_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2806 "bison_parser.cpp"
        break;

    case YYSYMBOL_datetime_field: /* datetime_field  */
#line 178 "bison_parser.y"
                { }
#line 2812 "bison_parser.cpp"
        break;

    case YYSYMBOL_duration_field: /* duration_field  */
#line 178 "bison_parser.y"
                { }
#line 2818 "bison_parser.cpp"
        break;

    case YYSYMBOL_array_expr: /* array_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2824 "bison_parser.cpp"
        break;

    case YYSYMBOL_array_index: /* array_index  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2830 "bison_parser.cpp"
        break;

    case YYSYMBOL_string_array_index: /* string_array_index  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2836 "bison_parser.cpp"
        break;

    case YYSYMBOL_fancy_array_index: /* fancy_array_index  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2842 "bison_parser.cpp"
        break;

    case YYSYMBOL_dynamic_array_index_operand: /* dynamic_array_index_operand  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2848 "bison_parser.cpp"
        break;

    case YYSYMBOL_dynamic_array_index: /* dynamic_array_index  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2854 "bison_parser.cpp"
        break;

    case YYSYMBOL_fancy_array_index_list: /* fancy_array_index_list  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).expr_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).expr_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).expr_vec));
    }
#line 2867 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal: /* slice_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2873 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_0_0_0: /* slice_literal_0_0_0  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2879 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_0_0_1: /* slice_literal_0_0_1  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2885 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_0_1_0: /* slice_literal_0_1_0  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2891 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_0_1_1: /* slice_literal_0_1_1  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2897 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_1_0_0: /* slice_literal_1_0_0  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2903 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_1_0_1: /* slice_literal_1_0_1  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2909 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_1_1_0: /* slice_literal_1_1_0  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2915 "bison_parser.cpp"
        break;

    case YYSYMBOL_slice_literal_1_1_1: /* slice_literal_1_1_1  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2921 "bison_parser.cpp"
        break;

    case YYSYMBOL_between_expr: /* between_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2927 "bison_parser.cpp"
        break;

    case YYSYMBOL_column_name: /* column_name  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2933 "bison_parser.cpp"
        break;

    case YYSYMBOL_literal: /* literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2939 "bison_parser.cpp"
        break;

    case YYSYMBOL_string_literal: /* string_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2945 "bison_parser.cpp"
        break;

    case YYSYMBOL_bool_literal: /* bool_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2951 "bison_parser.cpp"
        break;

    case YYSYMBOL_num_literal: /* num_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2957 "bison_parser.cpp"
        break;

    case YYSYMBOL_int_literal: /* int_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2963 "bison_parser.cpp"
        break;

    case YYSYMBOL_null_literal: /* null_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2969 "bison_parser.cpp"
        break;

    case YYSYMBOL_date_literal: /* date_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2975 "bison_parser.cpp"
        break;

    case YYSYMBOL_interval_literal: /* interval_literal  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2981 "bison_parser.cpp"
        break;

    case YYSYMBOL_param_expr: /* param_expr  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 2987 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_ref: /* table_ref  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 2993 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_ref_atomic: /* table_ref_atomic  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 2999 "bison_parser.cpp"
        break;

    case YYSYMBOL_nonjoin_table_ref_atomic: /* nonjoin_table_ref_atomic  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 3005 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_ref_commalist: /* table_ref_commalist  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).table_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).table_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).table_vec));
    }
#line 3018 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_ref_name: /* table_ref_name  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 3024 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_ref_name_no_alias: /* table_ref_name_no_alias  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 3030 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_name: /* table_name  */
#line 179 "bison_parser.y"
                { free( (((*yyvaluep).table_name).name) ); free( (((*yyvaluep).table_name).schema) ); }
#line 3036 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_index_name: /* opt_index_name  */
#line 180 "bison_parser.y"
                { free( (((*yyvaluep).sval)) ); }
#line 3042 "bison_parser.cpp"
        break;

    case YYSYMBOL_table_alias: /* table_alias  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).alias_t)); }
#line 3048 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_table_alias: /* opt_table_alias  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).alias_t)); }
#line 3054 "bison_parser.cpp"
        break;

    case YYSYMBOL_alias: /* alias  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).alias_t)); }
#line 3060 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_alias: /* opt_alias  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).alias_t)); }
#line 3066 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_locking_clause: /* opt_locking_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).locking_clause_vec)); }
#line 3072 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_locking_clause_list: /* opt_locking_clause_list  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).locking_clause_vec)); }
#line 3078 "bison_parser.cpp"
        break;

    case YYSYMBOL_locking_clause: /* locking_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).locking_t)); }
#line 3084 "bison_parser.cpp"
        break;

    case YYSYMBOL_row_lock_mode: /* row_lock_mode  */
#line 178 "bison_parser.y"
                { }
#line 3090 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_row_lock_policy: /* opt_row_lock_policy  */
#line 178 "bison_parser.y"
                { }
#line 3096 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_with_clause: /* opt_with_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).with_description_vec)); }
#line 3102 "bison_parser.cpp"
        break;

    case YYSYMBOL_with_clause: /* with_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).with_description_vec)); }
#line 3108 "bison_parser.cpp"
        break;

    case YYSYMBOL_with_description_list: /* with_description_list  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).with_description_vec)); }
#line 3114 "bison_parser.cpp"
        break;

    case YYSYMBOL_with_description: /* with_description  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).with_description_t)); }
#line 3120 "bison_parser.cpp"
        break;

    case YYSYMBOL_join_clause: /* join_clause  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).table)); }
#line 3126 "bison_parser.cpp"
        break;

    case YYSYMBOL_opt_join_type: /* opt_join_type  */
#line 178 "bison_parser.y"
                { }
#line 3132 "bison_parser.cpp"
        break;

    case YYSYMBOL_join_condition: /* join_condition  */
#line 189 "bison_parser.y"
                { delete (((*yyvaluep).expr)); }
#line 3138 "bison_parser.cpp"
        break;

    case YYSYMBOL_ident_commalist: /* ident_commalist  */
#line 181 "bison_parser.y"
                {
      if ((((*yyvaluep).str_vec)) != nullptr) {
        for (auto ptr : *(((*yyvaluep).str_vec))) {
          delete ptr;
        }
      }
      delete (((*yyvaluep).str_vec));
    }
#line 3151 "bison_parser.cpp"
        break;

      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}






/*----------.
| yyparse.  |
`----------*/

int
yyparse (hsql::SQLParserResult* result, yyscan_t scanner)
{
/* Lookahead token kind.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

/* Location data for the lookahead symbol.  */
static YYLTYPE yyloc_default
# if defined HSQL_LTYPE_IS_TRIVIAL && HSQL_LTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
YYLTYPE yylloc = yyloc_default;

    /* Number of syntax errors so far.  */
    int yynerrs = 0;

    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    /* The location stack: array, bottom, top.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls = yylsa;
    YYLTYPE *yylsp = yyls;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];

  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = SQL_HSQL_EMPTY; /* Cause a token to be read.  */


/* User initialization code.  */
#line 81 "bison_parser.y"
{
  // Initialize
  yylloc.first_column = 0;
  yylloc.last_column = 0;
  yylloc.first_line = 0;
  yylloc.last_line = 0;
  yylloc.total_column = 0;
  yylloc.string_length = 0;
}

#line 3259 "bison_parser.cpp"

  yylsp[0] = yylloc;
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yyls1, yysize * YYSIZEOF (*yylsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
        yyls = yyls1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == SQL_HSQL_EMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex (&yylval, &yylloc, scanner);
    }

  if (yychar <= SQL_YYEOF)
    {
      yychar = SQL_YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == SQL_HSQL_error)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = SQL_HSQL_UNDEF;
      yytoken = YYSYMBOL_YYerror;
      yyerror_range[1] = yylloc;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = SQL_HSQL_EMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* input: statement_list opt_semicolon  */
#line 328 "bison_parser.y"
                                     {
  for (SQLStatement* stmt : *(yyvsp[-1].stmt_vec)) {
    // Transfers ownership of the statement.
    result->addStatement(stmt);
  }

  unsigned param_id = 0;
  for (void* param : yyloc.param_list) {
    if (param != nullptr) {
      Expr* expr = (Expr*)param;
      expr->ival = param_id;
      result->addParameter(expr);
      ++param_id;
    }
  }
    delete (yyvsp[-1].stmt_vec);
  }
#line 3488 "bison_parser.cpp"
    break;

  case 3: /* statement_list: statement  */
#line 347 "bison_parser.y"
                           {
  (yyvsp[0].statement)->stringLength = yylloc.string_length;
  yylloc.string_length = 0;
  (yyval.stmt_vec) = new std::vector<SQLStatement*>();
  (yyval.stmt_vec)->push_back((yyvsp[0].statement));
}
#line 3499 "bison_parser.cpp"
    break;

  case 4: /* statement_list: statement_list ';' statement  */
#line 353 "bison_parser.y"
                               {
  (yyvsp[0].statement)->stringLength = yylloc.string_length;
  yylloc.string_length = 0;
  (yyvsp[-2].stmt_vec)->push_back((yyvsp[0].statement));
  (yyval.stmt_vec) = (yyvsp[-2].stmt_vec);
}
#line 3510 "bison_parser.cpp"
    break;

  case 5: /* statement: prepare_statement opt_hints  */
#line 360 "bison_parser.y"
                                        {
  (yyval.statement) = (yyvsp[-1].prep_stmt);
  (yyval.statement)->hints = (yyvsp[0].expr_vec);
}
#line 3519 "bison_parser.cpp"
    break;

  case 6: /* statement: preparable_statement opt_hints  */
#line 364 "bison_parser.y"
                                 {
  (yyval.statement) = (yyvsp[-1].statement);
  (yyval.statement)->hints = (yyvsp[0].expr_vec);
}
#line 3528 "bison_parser.cpp"
    break;

  case 7: /* statement: show_statement  */
#line 368 "bison_parser.y"
                 { (yyval.statement) = (yyvsp[0].show_stmt); }
#line 3534 "bison_parser.cpp"
    break;

  case 8: /* statement: import_statement  */
#line 369 "bison_parser.y"
                   { (yyval.statement) = (yyvsp[0].import_stmt); }
#line 3540 "bison_parser.cpp"
    break;

  case 9: /* statement: export_statement  */
#line 370 "bison_parser.y"
                   { (yyval.statement) = (yyvsp[0].export_stmt); }
#line 3546 "bison_parser.cpp"
    break;

  case 10: /* preparable_statement: select_statement  */
#line 372 "bison_parser.y"
                                        { (yyval.statement) = (yyvsp[0].select_stmt); }
#line 3552 "bison_parser.cpp"
    break;

  case 11: /* preparable_statement: create_statement  */
#line 373 "bison_parser.y"
                   { (yyval.statement) = (yyvsp[0].create_stmt); }
#line 3558 "bison_parser.cpp"
    break;

  case 12: /* preparable_statement: insert_statement  */
#line 374 "bison_parser.y"
                   { (yyval.statement) = (yyvsp[0].insert_stmt); }
#line 3564 "bison_parser.cpp"
    break;

  case 13: /* preparable_statement: delete_statement  */
#line 375 "bison_parser.y"
                   { (yyval.statement) = (yyvsp[0].delete_stmt); }
#line 3570 "bison_parser.cpp"
    break;

  case 14: /* preparable_statement: truncate_statement  */
#line 376 "bison_parser.y"
                     { (yyval.statement) = (yyvsp[0].delete_stmt); }
#line 3576 "bison_parser.cpp"
    break;

  case 15: /* preparable_statement: update_statement  */
#line 377 "bison_parser.y"
                   { (yyval.statement) = (yyvsp[0].update_stmt); }
#line 3582 "bison_parser.cpp"
    break;

  case 16: /* preparable_statement: drop_statement  */
#line 378 "bison_parser.y"
                 { (yyval.statement) = (yyvsp[0].drop_stmt); }
#line 3588 "bison_parser.cpp"
    break;

  case 17: /* preparable_statement: alter_statement  */
#line 379 "bison_parser.y"
                  { (yyval.statement) = (yyvsp[0].alter_stmt); }
#line 3594 "bison_parser.cpp"
    break;

  case 18: /* preparable_statement: execute_statement  */
#line 380 "bison_parser.y"
                    { (yyval.statement) = (yyvsp[0].exec_stmt); }
#line 3600 "bison_parser.cpp"
    break;

  case 19: /* preparable_statement: transaction_statement  */
#line 381 "bison_parser.y"
                        { (yyval.statement) = (yyvsp[0].transaction_stmt); }
#line 3606 "bison_parser.cpp"
    break;

  case 20: /* opt_hints: WITH HINT '(' hint_list ')'  */
#line 387 "bison_parser.y"
                                        { (yyval.expr_vec) = (yyvsp[-1].expr_vec); }
#line 3612 "bison_parser.cpp"
    break;

  case 21: /* opt_hints: %empty  */
#line 388 "bison_parser.y"
              { (yyval.expr_vec) = nullptr; }
#line 3618 "bison_parser.cpp"
    break;

  case 22: /* hint_list: hint  */
#line 390 "bison_parser.y"
                 {
  (yyval.expr_vec) = new std::vector<Expr*>();
  (yyval.expr_vec)->push_back((yyvsp[0].expr));
}
#line 3627 "bison_parser.cpp"
    break;

  case 23: /* hint_list: hint_list ',' hint  */
#line 394 "bison_parser.y"
                     {
  (yyvsp[-2].expr_vec)->push_back((yyvsp[0].expr));
  (yyval.expr_vec) = (yyvsp[-2].expr_vec);
}
#line 3636 "bison_parser.cpp"
    break;

  case 24: /* hint: IDENTIFIER  */
#line 399 "bison_parser.y"
                  {
  (yyval.expr) = Expr::make(kExprHint);
  (yyval.expr)->name = (yyvsp[0].sval);
}
#line 3645 "bison_parser.cpp"
    break;

  case 25: /* hint: IDENTIFIER '(' literal_list ')'  */
#line 403 "bison_parser.y"
                                  {
  (yyval.expr) = Expr::make(kExprHint);
  (yyval.expr)->name = (yyvsp[-3].sval);
  (yyval.expr)->exprList = (yyvsp[-1].expr_vec);
}
#line 3655 "bison_parser.cpp"
    break;

  case 26: /* transaction_statement: BEGIN opt_transaction_keyword  */
#line 413 "bison_parser.y"
                                                      { (yyval.transaction_stmt) = new TransactionStatement(kBeginTransaction); }
#line 3661 "bison_parser.cpp"
    break;

  case 27: /* transaction_statement: ROLLBACK opt_transaction_keyword  */
#line 414 "bison_parser.y"
                                   { (yyval.transaction_stmt) = new TransactionStatement(kRollbackTransaction); }
#line 3667 "bison_parser.cpp"
    break;

  case 28: /* transaction_statement: COMMIT opt_transaction_keyword  */
#line 415 "bison_parser.y"
                                 { (yyval.transaction_stmt) = new TransactionStatement(kCommitTransaction); }
#line 3673 "bison_parser.cpp"
    break;

  case 31: /* prepare_statement: PREPARE IDENTIFIER FROM prepare_target_query  */
#line 423 "bison_parser.y"
                                                                 {
  (yyval.prep_stmt) = new PrepareStatement();
  (yyval.prep_stmt)->name = (yyvsp[-2].sval);
  (yyval.prep_stmt)->query = (yyvsp[0].sval);
}
#line 3683 "bison_parser.cpp"
    break;

  case 33: /* execute_statement: EXECUTE IDENTIFIER  */
#line 431 "bison_parser.y"
                                                                  {
  (yyval.exec_stmt) = new ExecuteStatement();
  (yyval.exec_stmt)->name = (yyvsp[0].sval);
}
#line 3692 "bison_parser.cpp"
    break;

  case 34: /* execute_statement: EXECUTE IDENTIFIER '(' opt_literal_list ')'  */
#line 435 "bison_parser.y"
                                              {
  (yyval.exec_stmt) = new ExecuteStatement();
  (yyval.exec_stmt)->name = (yyvsp[-3].sval);
  (yyval.exec_stmt)->parameters = (yyvsp[-1].expr_vec);
}
#line 3702 "bison_parser.cpp"
    break;

  case 35: /* import_statement: IMPORT FROM file_type FILE file_path INTO table_name  */
#line 446 "bison_parser.y"
                                                                        {
  (yyval.import_stmt) = new ImportStatement((yyvsp[-4].import_type_t));
  (yyval.import_stmt)->filePath = (yyvsp[-2].sval);
  (yyval.import_stmt)->schema = (yyvsp[0].table_name).schema;
  (yyval.import_stmt)->tableName = (yyvsp[0].table_name).name;
}
#line 3713 "bison_parser.cpp"
    break;

  case 36: /* import_statement: COPY table_name FROM file_path opt_file_type  */
#line 452 "bison_parser.y"
                                               {
  (yyval.import_stmt) = new ImportStatement((yyvsp[0].import_type_t));
  (yyval.import_stmt)->filePath = (yyvsp[-1].sval);
  (yyval.import_stmt)->schema = (yyvsp[-3].table_name).schema;
  (yyval.import_stmt)->tableName = (yyvsp[-3].table_name).name;
}
#line 3724 "bison_parser.cpp"
    break;

  case 37: /* file_type: IDENTIFIER  */
#line 459 "bison_parser.y"
                       {
  if (strcasecmp((yyvsp[0].sval), "csv") == 0) {
    (yyval.import_type_t) = kImportCSV;
  } else if (strcasecmp((yyvsp[0].sval), "tbl") == 0) {
    (yyval.import_type_t) = kImportTbl;
  } else if (strcasecmp((yyvsp[0].sval), "binary") == 0 || strcasecmp((yyvsp[0].sval), "bin") == 0) {
    (yyval.import_type_t) = kImportBinary;
  } else {
    free((yyvsp[0].sval));
    yyerror(&yyloc, result, scanner, "File type is unknown.");
    YYERROR;
  }
  free((yyvsp[0].sval));
}
#line 3743 "bison_parser.cpp"
    break;

  case 38: /* file_path: string_literal  */
#line 474 "bison_parser.y"
                           {
  (yyval.sval) = strdup((yyvsp[0].expr)->name);
  delete (yyvsp[0].expr);
}
#line 3752 "bison_parser.cpp"
    break;

  case 39: /* opt_file_type: WITH FORMAT file_type  */
#line 479 "bison_parser.y"
                                      { (yyval.import_type_t) = (yyvsp[0].import_type_t); }
#line 3758 "bison_parser.cpp"
    break;

  case 40: /* opt_file_type: %empty  */
#line 480 "bison_parser.y"
              { (yyval.import_type_t) = kImportAuto; }
#line 3764 "bison_parser.cpp"
    break;

  case 41: /* export_statement: COPY table_name TO file_path opt_file_type  */
#line 486 "bison_parser.y"
                                                              {
  (yyval.export_stmt) = new ExportStatement((yyvsp[0].import_type_t));
  (yyval.export_stmt)->filePath = (yyvsp[-1].sval);
  (yyval.export_stmt)->schema = (yyvsp[-3].table_name).schema;
  (yyval.export_stmt)->tableName = (yyvsp[-3].table_name).name;
}
#line 3775 "bison_parser.cpp"
    break;

  case 42: /* show_statement: SHOW TABLES  */
#line 498 "bison_parser.y"
                             { (yyval.show_stmt) = new ShowStatement(kShowTables); }
#line 3781 "bison_parser.cpp"
    break;

  case 43: /* show_statement: SHOW COLUMNS table_name  */
#line 499 "bison_parser.y"
                          {
  (yyval.show_stmt) = new ShowStatement(kShowColumns);
  (yyval.show_stmt)->schema = (yyvsp[0].table_name).schema;
  (yyval.show_stmt)->name = (yyvsp[0].table_name).name;
}
#line 3791 "bison_parser.cpp"
    break;

  case 44: /* show_statement: DESCRIBE table_name  */
#line 504 "bison_parser.y"
                      {
  (yyval.show_stmt) = new ShowStatement(kShowColumns);
  (yyval.show_stmt)->schema = (yyvsp[0].table_name).schema;
  (yyval.show_stmt)->name = (yyvsp[0].table_name).name;
}
#line 3801 "bison_parser.cpp"
    break;

  case 45: /* create_statement: CREATE TABLE opt_not_exists table_name FROM IDENTIFIER FILE file_path  */
#line 515 "bison_parser.y"
                                                                                         {
  (yyval.create_stmt) = new CreateStatement(kCreateTableFromTbl);
  (yyval.create_stmt)->ifNotExists = (yyvsp[-5].bval);
  (yyval.create_stmt)->schema = (yyvsp[-4].table_name).schema;
  (yyval.create_stmt)->tableName = (yyvsp[-4].table_name).name;
  if (strcasecmp((yyvsp[-2].sval), "tbl") != 0) {
    free((yyvsp[-2].sval));
    yyerror(&yyloc, result, scanner, "File type is unknown.");
    YYERROR;
  }
  free((yyvsp[-2].sval));
  (yyval.create_stmt)->filePath = (yyvsp[0].sval);
}
#line 3819 "bison_parser.cpp"
    break;

  case 46: /* create_statement: CREATE TABLE opt_not_exists table_name '(' table_elem_commalist ')'  */
#line 528 "bison_parser.y"
                                                                      {
  (yyval.create_stmt) = new CreateStatement(kCreateTable);
  (yyval.create_stmt)->ifNotExists = (yyvsp[-4].bval);
  (yyval.create_stmt)->schema = (yyvsp[-3].table_name).schema;
  (yyval.create_stmt)->tableName = (yyvsp[-3].table_name).name;
  (yyval.create_stmt)->setColumnDefsAndConstraints((yyvsp[-1].table_element_vec));
  delete (yyvsp[-1].table_element_vec);
}
#line 3832 "bison_parser.cpp"
    break;

  case 47: /* create_statement: CREATE TABLE opt_not_exists table_name AS select_statement  */
#line 536 "bison_parser.y"
                                                             {
  (yyval.create_stmt) = new CreateStatement(kCreateTable);
  (yyval.create_stmt)->ifNotExists = (yyvsp[-3].bval);
  (yyval.create_stmt)->schema = (yyvsp[-2].table_name).schema;
  (yyval.create_stmt)->tableName = (yyvsp[-2].table_name).name;
  (yyval.create_stmt)->select = (yyvsp[0].select_stmt);
}
#line 3844 "bison_parser.cpp"
    break;

  case 48: /* create_statement: CREATE INDEX opt_not_exists opt_index_name ON table_name '(' ident_commalist ')'  */
#line 543 "bison_parser.y"
                                                                                   {
  (yyval.create_stmt) = new CreateStatement(kCreateIndex);
  (yyval.create_stmt)->indexName = (yyvsp[-5].sval);
  (yyval.create_stmt)->ifNotExists = (yyvsp[-6].bval);
  (yyval.create_stmt)->tableName = (yyvsp[-3].table_name).name;
  (yyval.create_stmt)->indexColumns = (yyvsp[-1].str_vec);
}
#line 3856 "bison_parser.cpp"
    break;

  case 49: /* create_statement: CREATE VIEW opt_not_exists table_name opt_column_list AS select_statement  */
#line 550 "bison_parser.y"
                                                                            {
  (yyval.create_stmt) = new CreateStatement(kCreateView);
  (yyval.create_stmt)->ifNotExists = (yyvsp[-4].bval);
  (yyval.create_stmt)->schema = (yyvsp[-3].table_name).schema;
  (yyval.create_stmt)->tableName = (yyvsp[-3].table_name).name;
  (yyval.create_stmt)->viewColumns = (yyvsp[-2].str_vec);
  (yyval.create_stmt)->select = (yyvsp[0].select_stmt);
}
#line 3869 "bison_parser.cpp"
    break;

  case 50: /* opt_not_exists: IF NOT EXISTS  */
#line 559 "bison_parser.y"
                               { (yyval.bval) = true; }
#line 3875 "bison_parser.cpp"
    break;

  case 51: /* opt_not_exists: %empty  */
#line 560 "bison_parser.y"
              { (yyval.bval) = false; }
#line 3881 "bison_parser.cpp"
    break;

  case 52: /* table_elem_commalist: table_elem  */
#line 562 "bison_parser.y"
                                  {
  (yyval.table_element_vec) = new std::vector<TableElement*>();
  (yyval.table_element_vec)->push_back((yyvsp[0].table_element_t));
}
#line 3890 "bison_parser.cpp"
    break;

  case 53: /* table_elem_commalist: table_elem_commalist ',' table_elem  */
#line 566 "bison_parser.y"
                                      {
  (yyvsp[-2].table_element_vec)->push_back((yyvsp[0].table_element_t));
  (yyval.table_element_vec) = (yyvsp[-2].table_element_vec);
}
#line 3899 "bison_parser.cpp"
    break;

  case 54: /* table_elem: column_def  */
#line 571 "bison_parser.y"
                        { (yyval.table_element_t) = (yyvsp[0].column_t); }
#line 3905 "bison_parser.cpp"
    break;

  case 55: /* table_elem: table_constraint  */
#line 572 "bison_parser.y"
                   { (yyval.table_element_t) = (yyvsp[0].table_constraint_t); }
#line 3911 "bison_parser.cpp"
    break;

  case 56: /* column_def: IDENTIFIER column_type opt_column_constraints  */
#line 574 "bison_parser.y"
                                                           {
  (yyval.column_t) = new ColumnDefinition((yyvsp[-2].sval), (yyvsp[-1].column_type_t), (yyvsp[0].column_constraint_vec));
  (yyval.column_t)->setNullableExplicit();
}
#line 3920 "bison_parser.cpp"
    break;

  case 57: /* column_type: INT  */
#line 579 "bison_parser.y"
                  { (yyval.column_type_t) = ColumnType{DataType::INT}; }
#line 3926 "bison_parser.cpp"
    break;

  case 58: /* column_type: CHAR '(' INTVAL ')'  */
#line 580 "bison_parser.y"
                      { (yyval.column_type_t) = ColumnType{DataType::CHAR, (yyvsp[-1].ival)}; }
#line 3932 "bison_parser.cpp"
    break;

  case 59: /* column_type: CHARACTER_VARYING '(' INTVAL ')'  */
#line 581 "bison_parser.y"
                                   { (yyval.column_type_t) = ColumnType{DataType::VARCHAR, (yyvsp[-1].ival)}; }
#line 3938 "bison_parser.cpp"
    break;

  case 60: /* column_type: DATE  */
#line 582 "bison_parser.y"
       { (yyval.column_type_t) = ColumnType{DataType::DATE}; }
#line 3944 "bison_parser.cpp"
    break;

  case 61: /* column_type: DATETIME  */
#line 583 "bison_parser.y"
           { (yyval.column_type_t) = ColumnType{DataType::DATETIME}; }
#line 3950 "bison_parser.cpp"
    break;

  case 62: /* column_type: DECIMAL opt_decimal_specification  */
#line 584 "bison_parser.y"
                                    {
  (yyval.column_type_t) = ColumnType{DataType::DECIMAL, 0, (yyvsp[0].ival_pair)->first, (yyvsp[0].ival_pair)->second};
  delete (yyvsp[0].ival_pair);
}
#line 3959 "bison_parser.cpp"
    break;

  case 63: /* column_type: DOUBLE  */
#line 588 "bison_parser.y"
         { (yyval.column_type_t) = ColumnType{DataType::DOUBLE}; }
#line 3965 "bison_parser.cpp"
    break;

  case 64: /* column_type: FLOAT  */
#line 589 "bison_parser.y"
        { (yyval.column_type_t) = ColumnType{DataType::FLOAT}; }
#line 3971 "bison_parser.cpp"
    break;

  case 65: /* column_type: INTEGER  */
#line 590 "bison_parser.y"
          { (yyval.column_type_t) = ColumnType{DataType::INT}; }
#line 3977 "bison_parser.cpp"
    break;

  case 66: /* column_type: LONG  */
#line 591 "bison_parser.y"
       { (yyval.column_type_t) = ColumnType{DataType::LONG}; }
#line 3983 "bison_parser.cpp"
    break;

  case 67: /* column_type: REAL  */
#line 592 "bison_parser.y"
       { (yyval.column_type_t) = ColumnType{DataType::REAL}; }
#line 3989 "bison_parser.cpp"
    break;

  case 68: /* column_type: SMALLINT  */
#line 593 "bison_parser.y"
           { (yyval.column_type_t) = ColumnType{DataType::SMALLINT}; }
#line 3995 "bison_parser.cpp"
    break;

  case 69: /* column_type: BIGINT  */
#line 594 "bison_parser.y"
         { (yyval.column_type_t) = ColumnType{DataType::BIGINT}; }
#line 4001 "bison_parser.cpp"
    break;

  case 70: /* column_type: TEXT_INTERNAL_TQL  */
#line 595 "bison_parser.y"
                    { (yyval.column_type_t) = ColumnType{DataType::TEXT}; }
#line 4007 "bison_parser.cpp"
    break;

  case 71: /* column_type: TIME opt_time_precision  */
#line 596 "bison_parser.y"
                          { (yyval.column_type_t) = ColumnType{DataType::TIME, 0, (yyvsp[0].ival)}; }
#line 4013 "bison_parser.cpp"
    break;

  case 72: /* column_type: VARCHAR '(' INTVAL ')'  */
#line 597 "bison_parser.y"
                         { (yyval.column_type_t) = ColumnType{DataType::VARCHAR, (yyvsp[-1].ival)}; }
#line 4019 "bison_parser.cpp"
    break;

  case 73: /* opt_time_precision: '(' INTVAL ')'  */
#line 599 "bison_parser.y"
                                    { (yyval.ival) = (yyvsp[-1].ival); }
#line 4025 "bison_parser.cpp"
    break;

  case 74: /* opt_time_precision: %empty  */
#line 600 "bison_parser.y"
              { (yyval.ival) = 0; }
#line 4031 "bison_parser.cpp"
    break;

  case 75: /* opt_decimal_specification: '(' INTVAL ',' INTVAL ')'  */
#line 602 "bison_parser.y"
                                                      { (yyval.ival_pair) = new std::pair<int64_t, int64_t>{(yyvsp[-3].ival), (yyvsp[-1].ival)}; }
#line 4037 "bison_parser.cpp"
    break;

  case 76: /* opt_decimal_specification: '(' INTVAL ')'  */
#line 603 "bison_parser.y"
                 { (yyval.ival_pair) = new std::pair<int64_t, int64_t>{(yyvsp[-1].ival), 0}; }
#line 4043 "bison_parser.cpp"
    break;

  case 77: /* opt_decimal_specification: %empty  */
#line 604 "bison_parser.y"
              { (yyval.ival_pair) = new std::pair<int64_t, int64_t>{0, 0}; }
#line 4049 "bison_parser.cpp"
    break;

  case 78: /* opt_column_constraints: column_constraint_list  */
#line 606 "bison_parser.y"
                                                { (yyval.column_constraint_vec) = (yyvsp[0].column_constraint_vec); }
#line 4055 "bison_parser.cpp"
    break;

  case 79: /* opt_column_constraints: %empty  */
#line 607 "bison_parser.y"
              { (yyval.column_constraint_vec) = new std::vector<ConstraintType>(); }
#line 4061 "bison_parser.cpp"
    break;

  case 80: /* column_constraint_list: column_constraint  */
#line 609 "bison_parser.y"
                                           {
  (yyval.column_constraint_vec) = new std::vector<ConstraintType>();
  (yyval.column_constraint_vec)->push_back((yyvsp[0].column_constraint_t));
}
#line 4070 "bison_parser.cpp"
    break;

  case 81: /* column_constraint_list: column_constraint_list column_constraint  */
#line 613 "bison_parser.y"
                                           {
  (yyvsp[-1].column_constraint_vec)->push_back((yyvsp[0].column_constraint_t));
  (yyval.column_constraint_vec) = (yyvsp[-1].column_constraint_vec);
}
#line 4079 "bison_parser.cpp"
    break;

  case 82: /* column_constraint: PRIMARY KEY  */
#line 618 "bison_parser.y"
                                { (yyval.column_constraint_t) = ConstraintType::PrimaryKey; }
#line 4085 "bison_parser.cpp"
    break;

  case 83: /* column_constraint: UNIQUE  */
#line 619 "bison_parser.y"
         { (yyval.column_constraint_t) = ConstraintType::Unique; }
#line 4091 "bison_parser.cpp"
    break;

  case 84: /* column_constraint: NULL  */
#line 620 "bison_parser.y"
       { (yyval.column_constraint_t) = ConstraintType::Null; }
#line 4097 "bison_parser.cpp"
    break;

  case 85: /* column_constraint: NOT NULL  */
#line 621 "bison_parser.y"
           { (yyval.column_constraint_t) = ConstraintType::NotNull; }
#line 4103 "bison_parser.cpp"
    break;

  case 86: /* table_constraint: PRIMARY KEY '(' ident_commalist ')'  */
#line 623 "bison_parser.y"
                                                       { (yyval.table_constraint_t) = new TableConstraint(ConstraintType::PrimaryKey, (yyvsp[-1].str_vec)); }
#line 4109 "bison_parser.cpp"
    break;

  case 87: /* table_constraint: UNIQUE '(' ident_commalist ')'  */
#line 624 "bison_parser.y"
                                 { (yyval.table_constraint_t) = new TableConstraint(ConstraintType::Unique, (yyvsp[-1].str_vec)); }
#line 4115 "bison_parser.cpp"
    break;

  case 88: /* drop_statement: DROP TABLE opt_exists table_name  */
#line 632 "bison_parser.y"
                                                  {
  (yyval.drop_stmt) = new DropStatement(kDropTable);
  (yyval.drop_stmt)->ifExists = (yyvsp[-1].bval);
  (yyval.drop_stmt)->schema = (yyvsp[0].table_name).schema;
  (yyval.drop_stmt)->name = (yyvsp[0].table_name).name;
}
#line 4126 "bison_parser.cpp"
    break;

  case 89: /* drop_statement: DROP VIEW opt_exists table_name  */
#line 638 "bison_parser.y"
                                  {
  (yyval.drop_stmt) = new DropStatement(kDropView);
  (yyval.drop_stmt)->ifExists = (yyvsp[-1].bval);
  (yyval.drop_stmt)->schema = (yyvsp[0].table_name).schema;
  (yyval.drop_stmt)->name = (yyvsp[0].table_name).name;
}
#line 4137 "bison_parser.cpp"
    break;

  case 90: /* drop_statement: DEALLOCATE PREPARE IDENTIFIER  */
#line 644 "bison_parser.y"
                                {
  (yyval.drop_stmt) = new DropStatement(kDropPreparedStatement);
  (yyval.drop_stmt)->ifExists = false;
  (yyval.drop_stmt)->name = (yyvsp[0].sval);
}
#line 4147 "bison_parser.cpp"
    break;

  case 91: /* drop_statement: DROP INDEX opt_exists IDENTIFIER  */
#line 650 "bison_parser.y"
                                   {
  (yyval.drop_stmt) = new DropStatement(kDropIndex);
  (yyval.drop_stmt)->ifExists = (yyvsp[-1].bval);
  (yyval.drop_stmt)->indexName = (yyvsp[0].sval);
}
#line 4157 "bison_parser.cpp"
    break;

  case 92: /* opt_exists: IF EXISTS  */
#line 656 "bison_parser.y"
                       { (yyval.bval) = true; }
#line 4163 "bison_parser.cpp"
    break;

  case 93: /* opt_exists: %empty  */
#line 657 "bison_parser.y"
              { (yyval.bval) = false; }
#line 4169 "bison_parser.cpp"
    break;

  case 94: /* alter_statement: ALTER TABLE opt_exists table_name alter_action  */
#line 664 "bison_parser.y"
                                                                 {
  (yyval.alter_stmt) = new AlterStatement((yyvsp[-1].table_name).name, (yyvsp[0].alter_action_t));
  (yyval.alter_stmt)->ifTableExists = (yyvsp[-2].bval);
  (yyval.alter_stmt)->schema = (yyvsp[-1].table_name).schema;
}
#line 4179 "bison_parser.cpp"
    break;

  case 95: /* alter_action: drop_action  */
#line 670 "bison_parser.y"
                           { (yyval.alter_action_t) = (yyvsp[0].drop_action_t); }
#line 4185 "bison_parser.cpp"
    break;

  case 96: /* drop_action: DROP COLUMN opt_exists IDENTIFIER  */
#line 672 "bison_parser.y"
                                                {
  (yyval.drop_action_t) = new DropColumnAction((yyvsp[0].sval));
  (yyval.drop_action_t)->ifExists = (yyvsp[-1].bval);
}
#line 4194 "bison_parser.cpp"
    break;

  case 97: /* delete_statement: DELETE FROM table_name opt_where  */
#line 682 "bison_parser.y"
                                                    {
  (yyval.delete_stmt) = new DeleteStatement();
  (yyval.delete_stmt)->schema = (yyvsp[-1].table_name).schema;
  (yyval.delete_stmt)->tableName = (yyvsp[-1].table_name).name;
  (yyval.delete_stmt)->expr = (yyvsp[0].whereClause)->expr;
}
#line 4205 "bison_parser.cpp"
    break;

  case 98: /* truncate_statement: TRUNCATE table_name  */
#line 689 "bison_parser.y"
                                         {
  (yyval.delete_stmt) = new DeleteStatement();
  (yyval.delete_stmt)->schema = (yyvsp[0].table_name).schema;
  (yyval.delete_stmt)->tableName = (yyvsp[0].table_name).name;
}
#line 4215 "bison_parser.cpp"
    break;

  case 99: /* insert_statement: INSERT INTO table_name opt_column_list VALUES '(' literal_list ')'  */
#line 700 "bison_parser.y"
                                                                                      {
  (yyval.insert_stmt) = new InsertStatement(kInsertValues);
  (yyval.insert_stmt)->schema = (yyvsp[-5].table_name).schema;
  (yyval.insert_stmt)->tableName = (yyvsp[-5].table_name).name;
  (yyval.insert_stmt)->columns = (yyvsp[-4].str_vec);
  (yyval.insert_stmt)->values = (yyvsp[-1].expr_vec);
}
#line 4227 "bison_parser.cpp"
    break;

  case 100: /* insert_statement: INSERT INTO table_name opt_column_list select_no_paren  */
#line 707 "bison_parser.y"
                                                         {
  (yyval.insert_stmt) = new InsertStatement(kInsertSelect);
  (yyval.insert_stmt)->schema = (yyvsp[-2].table_name).schema;
  (yyval.insert_stmt)->tableName = (yyvsp[-2].table_name).name;
  (yyval.insert_stmt)->columns = (yyvsp[-1].str_vec);
  (yyval.insert_stmt)->select = (yyvsp[0].select_stmt);
}
#line 4239 "bison_parser.cpp"
    break;

  case 101: /* opt_column_list: '(' ident_commalist ')'  */
#line 715 "bison_parser.y"
                                          { (yyval.str_vec) = (yyvsp[-1].str_vec); }
#line 4245 "bison_parser.cpp"
    break;

  case 102: /* opt_column_list: %empty  */
#line 716 "bison_parser.y"
              { (yyval.str_vec) = nullptr; }
#line 4251 "bison_parser.cpp"
    break;

  case 103: /* update_statement: UPDATE table_ref_name_no_alias SET update_clause_commalist opt_where  */
#line 723 "bison_parser.y"
                                                                                        {
  (yyval.update_stmt) = new UpdateStatement();
  (yyval.update_stmt)->table = (yyvsp[-3].table);
  (yyval.update_stmt)->updates = (yyvsp[-1].update_vec);
  (yyval.update_stmt)->where = (yyvsp[0].whereClause)->expr;
}
#line 4262 "bison_parser.cpp"
    break;

  case 104: /* update_clause_commalist: update_clause  */
#line 730 "bison_parser.y"
                                        {
  (yyval.update_vec) = new std::vector<UpdateClause*>();
  (yyval.update_vec)->push_back((yyvsp[0].update_t));
}
#line 4271 "bison_parser.cpp"
    break;

  case 105: /* update_clause_commalist: update_clause_commalist ',' update_clause  */
#line 734 "bison_parser.y"
                                            {
  (yyvsp[-2].update_vec)->push_back((yyvsp[0].update_t));
  (yyval.update_vec) = (yyvsp[-2].update_vec);
}
#line 4280 "bison_parser.cpp"
    break;

  case 106: /* update_clause: IDENTIFIER '=' expr  */
#line 739 "bison_parser.y"
                                    {
  (yyval.update_t) = new UpdateClause();
  (yyval.update_t)->column = (yyvsp[-2].sval);
  (yyval.update_t)->value = (yyvsp[0].expr);
}
#line 4290 "bison_parser.cpp"
    break;

  case 107: /* select_statement: opt_with_clause select_with_paren  */
#line 749 "bison_parser.y"
                                                     {
  (yyval.select_stmt) = (yyvsp[0].select_stmt);
  (yyval.select_stmt)->withDescriptions = (yyvsp[-1].with_description_vec);
}
#line 4299 "bison_parser.cpp"
    break;

  case 108: /* select_statement: opt_with_clause select_no_paren  */
#line 753 "bison_parser.y"
                                  {
  (yyval.select_stmt) = (yyvsp[0].select_stmt);
  (yyval.select_stmt)->withDescriptions = (yyvsp[-1].with_description_vec);
}
#line 4308 "bison_parser.cpp"
    break;

  case 109: /* select_statement: opt_with_clause select_with_paren set_operator select_statement opt_order opt_limit opt_sample  */
#line 757 "bison_parser.y"
                                                                                                 {
  (yyval.select_stmt) = (yyvsp[-5].select_stmt);
  if ((yyval.select_stmt)->setOperations == nullptr) {
    (yyval.select_stmt)->setOperations = new std::vector<SetOperation*>();
  }
  (yyval.select_stmt)->setOperations->push_back((yyvsp[-4].set_operator_t));
  (yyval.select_stmt)->setOperations->back()->nestedSelectStatement = (yyvsp[-3].select_stmt);
  (yyval.select_stmt)->setOperations->back()->resultOrder = (yyvsp[-2].order_vec);
  (yyval.select_stmt)->setOperations->back()->resultLimit = (yyvsp[-1].limit);
  (yyval.select_stmt)->sampleBy = (yyvsp[0].sample);
  (yyval.select_stmt)->withDescriptions = (yyvsp[-6].with_description_vec);
}
#line 4325 "bison_parser.cpp"
    break;

  case 112: /* select_within_set_operation_no_parentheses: select_clause  */
#line 772 "bison_parser.y"
                                                           { (yyval.select_stmt) = (yyvsp[0].select_stmt); }
#line 4331 "bison_parser.cpp"
    break;

  case 113: /* select_within_set_operation_no_parentheses: select_clause set_operator select_within_set_operation  */
#line 773 "bison_parser.y"
                                                         {
  (yyval.select_stmt) = (yyvsp[-2].select_stmt);
  if ((yyval.select_stmt)->setOperations == nullptr) {
    (yyval.select_stmt)->setOperations = new std::vector<SetOperation*>();
  }
  (yyval.select_stmt)->setOperations->push_back((yyvsp[-1].set_operator_t));
  (yyval.select_stmt)->setOperations->back()->nestedSelectStatement = (yyvsp[0].select_stmt);
}
#line 4344 "bison_parser.cpp"
    break;

  case 114: /* select_with_paren: '(' select_no_paren ')'  */
#line 782 "bison_parser.y"
                                            { (yyval.select_stmt) = (yyvsp[-1].select_stmt); }
#line 4350 "bison_parser.cpp"
    break;

  case 115: /* select_with_paren: '(' select_with_paren ')'  */
#line 783 "bison_parser.y"
                            { (yyval.select_stmt) = (yyvsp[-1].select_stmt); }
#line 4356 "bison_parser.cpp"
    break;

  case 116: /* select_no_paren: select_clause opt_order opt_limit opt_sample opt_locking_clause  */
#line 785 "bison_parser.y"
                                                                                  {
  (yyval.select_stmt) = (yyvsp[-4].select_stmt);
  (yyval.select_stmt)->order = (yyvsp[-3].order_vec);

  // Limit could have been set by TOP.
  if ((yyvsp[-2].limit) != nullptr) {
    delete (yyval.select_stmt)->limit;
    (yyval.select_stmt)->limit = (yyvsp[-2].limit);
  }

  if ((yyvsp[-1].sample) != nullptr) {
    (yyval.select_stmt)->sampleBy = (yyvsp[-1].sample);
  }

  if ((yyvsp[0].locking_clause_vec) != nullptr) {
    (yyval.select_stmt)->lockings = (yyvsp[0].locking_clause_vec);
  }
}
#line 4379 "bison_parser.cpp"
    break;

  case 117: /* select_no_paren: select_clause set_operator select_within_set_operation opt_order opt_limit opt_sample opt_locking_clause  */
#line 803 "bison_parser.y"
                                                                                                           {
  (yyval.select_stmt) = (yyvsp[-6].select_stmt);
  if ((yyval.select_stmt)->setOperations == nullptr) {
    (yyval.select_stmt)->setOperations = new std::vector<SetOperation*>();
  }
  (yyval.select_stmt)->setOperations->push_back((yyvsp[-5].set_operator_t));
  (yyval.select_stmt)->setOperations->back()->nestedSelectStatement = (yyvsp[-4].select_stmt);
  (yyval.select_stmt)->setOperations->back()->resultOrder = (yyvsp[-3].order_vec);
  (yyval.select_stmt)->setOperations->back()->resultLimit = (yyvsp[-2].limit);
  (yyval.select_stmt)->sampleBy = (yyvsp[-1].sample);
  (yyval.select_stmt)->lockings = (yyvsp[0].locking_clause_vec);
}
#line 4396 "bison_parser.cpp"
    break;

  case 118: /* set_operator: set_type opt_all  */
#line 816 "bison_parser.y"
                                {
  (yyval.set_operator_t) = (yyvsp[-1].set_operator_t);
  (yyval.set_operator_t)->isAll = (yyvsp[0].bval);
}
#line 4405 "bison_parser.cpp"
    break;

  case 119: /* set_type: UNION  */
#line 821 "bison_parser.y"
                 {
  (yyval.set_operator_t) = new SetOperation();
  (yyval.set_operator_t)->setType = SetType::kSetUnion;
}
#line 4414 "bison_parser.cpp"
    break;

  case 120: /* set_type: INTERSECT  */
#line 825 "bison_parser.y"
            {
  (yyval.set_operator_t) = new SetOperation();
  (yyval.set_operator_t)->setType = SetType::kSetIntersect;
}
#line 4423 "bison_parser.cpp"
    break;

  case 121: /* set_type: EXCEPT  */
#line 829 "bison_parser.y"
         {
  (yyval.set_operator_t) = new SetOperation();
  (yyval.set_operator_t)->setType = SetType::kSetExcept;
}
#line 4432 "bison_parser.cpp"
    break;

  case 122: /* opt_all: ALL  */
#line 834 "bison_parser.y"
              { (yyval.bval) = true; }
#line 4438 "bison_parser.cpp"
    break;

  case 123: /* opt_all: %empty  */
#line 835 "bison_parser.y"
              { (yyval.bval) = false; }
#line 4444 "bison_parser.cpp"
    break;

  case 124: /* select_clause: SELECT opt_top opt_distinct select_list opt_from_clause opt_where opt_expand opt_group opt_ungroup  */
#line 837 "bison_parser.y"
                                                                                                                   {
  (yyval.select_stmt) = new SelectStatement();
  (yyval.select_stmt)->limit = (yyvsp[-7].limit);
  (yyval.select_stmt)->distinct = (yyvsp[-6].distinct_description_t);
  (yyval.select_stmt)->selectList = (yyvsp[-5].expr_vec);
  (yyval.select_stmt)->fromTable = (yyvsp[-4].table);
  (yyval.select_stmt)->whereClause = (yyvsp[-3].whereClause);
  (yyval.select_stmt)->expansion = (yyvsp[-2].expansion);
  (yyval.select_stmt)->groupBy = (yyvsp[-1].group_t);
  (yyval.select_stmt)->unGroupBy = (yyvsp[0].ungroup_t);
}
#line 4460 "bison_parser.cpp"
    break;

  case 125: /* opt_distinct: DISTINCT  */
#line 850 "bison_parser.y"
             {
        (yyval.distinct_description_t) = new DistinctDescription();
    }
#line 4468 "bison_parser.cpp"
    break;

  case 126: /* opt_distinct: DISTINCT ON '(' expr_list ')'  */
#line 853 "bison_parser.y"
                                    {
        (yyval.distinct_description_t) = new DistinctDescription();
        (yyval.distinct_description_t)->distinct_columns = (yyvsp[-1].expr_vec);
    }
#line 4477 "bison_parser.cpp"
    break;

  case 127: /* opt_distinct: %empty  */
#line 857 "bison_parser.y"
                  {
        (yyval.distinct_description_t) = nullptr;
    }
#line 4485 "bison_parser.cpp"
    break;

  case 129: /* opt_from_clause: from_clause  */
#line 863 "bison_parser.y"
                              { (yyval.table) = (yyvsp[0].table); }
#line 4491 "bison_parser.cpp"
    break;

  case 130: /* opt_from_clause: %empty  */
#line 864 "bison_parser.y"
              { (yyval.table) = nullptr; }
#line 4497 "bison_parser.cpp"
    break;

  case 131: /* from_clause: FROM table_ref  */
#line 866 "bison_parser.y"
                             { (yyval.table) = (yyvsp[0].table); }
#line 4503 "bison_parser.cpp"
    break;

  case 132: /* opt_where: WHERE expr  */
#line 868 "bison_parser.y"
                       { (yyval.whereClause) = new hsql::WhereClause((yyvsp[0].expr)); }
#line 4509 "bison_parser.cpp"
    break;

  case 133: /* opt_where: %empty  */
#line 869 "bison_parser.y"
              { (yyval.whereClause) = nullptr; }
#line 4515 "bison_parser.cpp"
    break;

  case 134: /* opt_expand: EXPAND BY INTVAL INTVAL opt_expand_overlap opt_expand_name  */
#line 871 "bison_parser.y"
                                                                        { (yyval.expansion) = new hsql::Expansion((yyvsp[-3].ival), (yyvsp[-2].ival), (yyvsp[0].expr), (yyvsp[-1].expr)); }
#line 4521 "bison_parser.cpp"
    break;

  case 135: /* opt_expand: EXPAND BY INTVAL opt_expand_overlap opt_expand_name  */
#line 872 "bison_parser.y"
                                                      { (yyval.expansion) = new hsql::Expansion((yyvsp[-2].ival), (yyvsp[-2].ival), (yyvsp[0].expr), (yyvsp[-1].expr)); }
#line 4527 "bison_parser.cpp"
    break;

  case 136: /* opt_expand: %empty  */
#line 873 "bison_parser.y"
              { (yyval.expansion) = nullptr; }
#line 4533 "bison_parser.cpp"
    break;

  case 137: /* opt_expand_name: AS IDENTIFIER  */
#line 875 "bison_parser.y"
                                { (yyval.expr) = hsql::Expr::makeLiteral((yyvsp[0].sval)); }
#line 4539 "bison_parser.cpp"
    break;

  case 138: /* opt_expand_name: %empty  */
#line 876 "bison_parser.y"
              { (yyval.expr) = nullptr; }
#line 4545 "bison_parser.cpp"
    break;

  case 139: /* opt_expand_overlap: OVERLAP bool_literal  */
#line 878 "bison_parser.y"
                                          { (yyval.expr) = (yyvsp[0].expr); }
#line 4551 "bison_parser.cpp"
    break;

  case 140: /* opt_expand_overlap: OVERLAP  */
#line 879 "bison_parser.y"
          { (yyval.expr) = Expr::makeLiteral(true); }
#line 4557 "bison_parser.cpp"
    break;

  case 141: /* opt_expand_overlap: %empty  */
#line 880 "bison_parser.y"
              { (yyval.expr) = Expr::makeLiteral(true); }
#line 4563 "bison_parser.cpp"
    break;

  case 142: /* opt_across: ACROSS TIME  */
#line 882 "bison_parser.y"
                          { (yyval.across_type) = AcrossType::Time; }
#line 4569 "bison_parser.cpp"
    break;

  case 143: /* opt_across: ACROSS SPACE  */
#line 883 "bison_parser.y"
                           { (yyval.across_type) = AcrossType::Space; }
#line 4575 "bison_parser.cpp"
    break;

  case 144: /* opt_across: %empty  */
#line 884 "bison_parser.y"
                          { (yyval.across_type) = AcrossType::Time; }
#line 4581 "bison_parser.cpp"
    break;

  case 145: /* opt_group: GROUP BY expr_list opt_having opt_across  */
#line 887 "bison_parser.y"
                                                     {
  (yyval.group_t) = new GroupByDescription();
  (yyval.group_t)->columns = (yyvsp[-2].expr_vec);
  (yyval.group_t)->having = (yyvsp[-1].expr);
  (yyval.group_t)->across = (yyvsp[0].across_type);
}
#line 4592 "bison_parser.cpp"
    break;

  case 146: /* opt_group: %empty  */
#line 893 "bison_parser.y"
              { (yyval.group_t) = nullptr; }
#line 4598 "bison_parser.cpp"
    break;

  case 147: /* opt_ungroup: UNGROUP BY expr  */
#line 895 "bison_parser.y"
                              {
  (yyval.ungroup_t) = new UnGroupByDescription();
  (yyval.ungroup_t)->expr = (yyvsp[0].expr);
}
#line 4607 "bison_parser.cpp"
    break;

  case 148: /* opt_ungroup: UNGROUP BY SPLIT  */
#line 898 "bison_parser.y"
                     {
  (yyval.ungroup_t) = new UnGroupByDescription();
  (yyval.ungroup_t)->split = true;
}
#line 4616 "bison_parser.cpp"
    break;

  case 149: /* opt_ungroup: %empty  */
#line 902 "bison_parser.y"
              { (yyval.ungroup_t) = nullptr; }
#line 4622 "bison_parser.cpp"
    break;

  case 150: /* opt_having: HAVING expr  */
#line 904 "bison_parser.y"
                         { (yyval.expr) = (yyvsp[0].expr); }
#line 4628 "bison_parser.cpp"
    break;

  case 151: /* opt_having: %empty  */
#line 905 "bison_parser.y"
              { (yyval.expr) = nullptr; }
#line 4634 "bison_parser.cpp"
    break;

  case 152: /* opt_sample: sample_desc  */
#line 907 "bison_parser.y"
                         { (yyval.sample) = (yyvsp[0].sample); }
#line 4640 "bison_parser.cpp"
    break;

  case 153: /* opt_sample: %empty  */
#line 908 "bison_parser.y"
  { (yyval.sample) = nullptr; }
#line 4646 "bison_parser.cpp"
    break;

  case 154: /* sample_desc: SAMPLE BY expr REPLACE opt_sample_limit  */
#line 910 "bison_parser.y"
                                                      { (yyval.sample) = new SampleDescription((yyvsp[-2].expr), (yyvsp[0].sample_limit), true); }
#line 4652 "bison_parser.cpp"
    break;

  case 155: /* sample_desc: SAMPLE BY expr opt_sample_limit  */
#line 911 "bison_parser.y"
                                  { (yyval.sample) = new SampleDescription((yyvsp[-1].expr), (yyvsp[0].sample_limit), true); }
#line 4658 "bison_parser.cpp"
    break;

  case 156: /* sample_desc: SAMPLE BY expr REPLACE FALSE opt_sample_limit  */
#line 912 "bison_parser.y"
                                                { (yyval.sample) = new SampleDescription((yyvsp[-3].expr), (yyvsp[0].sample_limit), false); }
#line 4664 "bison_parser.cpp"
    break;

  case 157: /* sample_desc: SAMPLE BY expr REPLACE TRUE opt_sample_limit  */
#line 913 "bison_parser.y"
                                               { (yyval.sample) = new SampleDescription((yyvsp[-3].expr), (yyvsp[0].sample_limit), true); }
#line 4670 "bison_parser.cpp"
    break;

  case 158: /* opt_order: ORDER BY order_list  */
#line 915 "bison_parser.y"
                                { (yyval.order_vec) = (yyvsp[0].order_vec); }
#line 4676 "bison_parser.cpp"
    break;

  case 159: /* opt_order: %empty  */
#line 916 "bison_parser.y"
              { (yyval.order_vec) = nullptr; }
#line 4682 "bison_parser.cpp"
    break;

  case 160: /* order_list: order_desc  */
#line 918 "bison_parser.y"
                        {
  (yyval.order_vec) = new std::vector<OrderDescription*>();
  (yyval.order_vec)->push_back((yyvsp[0].order));
}
#line 4691 "bison_parser.cpp"
    break;

  case 161: /* order_list: order_list ',' order_desc  */
#line 922 "bison_parser.y"
                            {
  (yyvsp[-2].order_vec)->push_back((yyvsp[0].order));
  (yyval.order_vec) = (yyvsp[-2].order_vec);
}
#line 4700 "bison_parser.cpp"
    break;

  case 162: /* order_desc: expr opt_order_type  */
#line 927 "bison_parser.y"
                                 { (yyval.order) = new OrderDescription((yyvsp[0].order_type), (yyvsp[-1].expr)); }
#line 4706 "bison_parser.cpp"
    break;

  case 163: /* opt_order_type: ASC  */
#line 929 "bison_parser.y"
                     { (yyval.order_type) = kOrderAsc; }
#line 4712 "bison_parser.cpp"
    break;

  case 164: /* opt_order_type: DESC  */
#line 930 "bison_parser.y"
       { (yyval.order_type) = kOrderDesc; }
#line 4718 "bison_parser.cpp"
    break;

  case 165: /* opt_order_type: %empty  */
#line 931 "bison_parser.y"
              { (yyval.order_type) = kOrderAsc; }
#line 4724 "bison_parser.cpp"
    break;

  case 166: /* opt_top: TOP int_literal  */
#line 935 "bison_parser.y"
                          { (yyval.limit) = new LimitDescription((yyvsp[0].expr), nullptr); }
#line 4730 "bison_parser.cpp"
    break;

  case 167: /* opt_top: %empty  */
#line 936 "bison_parser.y"
              { (yyval.limit) = nullptr; }
#line 4736 "bison_parser.cpp"
    break;

  case 168: /* opt_limit: LIMIT expr  */
#line 938 "bison_parser.y"
                       { (yyval.limit) = new LimitDescription((yyvsp[0].expr), nullptr); }
#line 4742 "bison_parser.cpp"
    break;

  case 169: /* opt_limit: OFFSET expr  */
#line 939 "bison_parser.y"
              { (yyval.limit) = new LimitDescription(nullptr, (yyvsp[0].expr)); }
#line 4748 "bison_parser.cpp"
    break;

  case 170: /* opt_limit: LIMIT expr OFFSET expr  */
#line 940 "bison_parser.y"
                         { (yyval.limit) = new LimitDescription((yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 4754 "bison_parser.cpp"
    break;

  case 171: /* opt_limit: LIMIT ALL  */
#line 941 "bison_parser.y"
            { (yyval.limit) = new LimitDescription(nullptr, nullptr); }
#line 4760 "bison_parser.cpp"
    break;

  case 172: /* opt_limit: LIMIT ALL OFFSET expr  */
#line 942 "bison_parser.y"
                        { (yyval.limit) = new LimitDescription(nullptr, (yyvsp[0].expr)); }
#line 4766 "bison_parser.cpp"
    break;

  case 173: /* opt_limit: %empty  */
#line 943 "bison_parser.y"
              { (yyval.limit) = nullptr; }
#line 4772 "bison_parser.cpp"
    break;

  case 174: /* opt_sample_limit: LIMIT expr  */
#line 945 "bison_parser.y"
                              { (yyval.sample_limit) = new SampleLimitDescription((yyvsp[0].expr), false); }
#line 4778 "bison_parser.cpp"
    break;

  case 175: /* opt_sample_limit: LIMIT expr PERCENT  */
#line 946 "bison_parser.y"
                     { (yyval.sample_limit) = new SampleLimitDescription((yyvsp[-1].expr), true); }
#line 4784 "bison_parser.cpp"
    break;

  case 176: /* opt_sample_limit: %empty  */
#line 947 "bison_parser.y"
  { (yyval.sample_limit) = nullptr; }
#line 4790 "bison_parser.cpp"
    break;

  case 177: /* expr_list: expr_alias  */
#line 952 "bison_parser.y"
                       {
  (yyval.expr_vec) = new std::vector<Expr*>();
  (yyval.expr_vec)->push_back((yyvsp[0].expr));
}
#line 4799 "bison_parser.cpp"
    break;

  case 178: /* expr_list: expr_list ',' expr_alias  */
#line 956 "bison_parser.y"
                           {
  (yyvsp[-2].expr_vec)->push_back((yyvsp[0].expr));
  (yyval.expr_vec) = (yyvsp[-2].expr_vec);
}
#line 4808 "bison_parser.cpp"
    break;

  case 179: /* expr_pair_list: expr_alias ':' expr_alias  */
#line 961 "bison_parser.y"
                                           {
  (yyval.expr_map) = new std::map<Expr*, Expr*>();
  (yyval.expr_map)->emplace((yyvsp[-2].expr), (yyvsp[0].expr));
}
#line 4817 "bison_parser.cpp"
    break;

  case 180: /* expr_pair_list: expr_pair_list ',' expr_alias ':' expr_alias  */
#line 965 "bison_parser.y"
                                               {
  (yyvsp[-4].expr_map)->emplace((yyvsp[-2].expr), (yyvsp[0].expr));
  (yyval.expr_map) = (yyvsp[-4].expr_map);
}
#line 4826 "bison_parser.cpp"
    break;

  case 181: /* opt_literal_list: literal_list  */
#line 970 "bison_parser.y"
                                { (yyval.expr_vec) = (yyvsp[0].expr_vec); }
#line 4832 "bison_parser.cpp"
    break;

  case 182: /* opt_literal_list: %empty  */
#line 971 "bison_parser.y"
              { (yyval.expr_vec) = nullptr; }
#line 4838 "bison_parser.cpp"
    break;

  case 183: /* literal_list: literal  */
#line 973 "bison_parser.y"
                       {
  (yyval.expr_vec) = new std::vector<Expr*>();
  (yyval.expr_vec)->push_back((yyvsp[0].expr));
}
#line 4847 "bison_parser.cpp"
    break;

  case 184: /* literal_list: literal_list ',' literal  */
#line 977 "bison_parser.y"
                           {
  (yyvsp[-2].expr_vec)->push_back((yyvsp[0].expr));
  (yyval.expr_vec) = (yyvsp[-2].expr_vec);
}
#line 4856 "bison_parser.cpp"
    break;

  case 185: /* expr_alias: expr opt_alias  */
#line 982 "bison_parser.y"
                            {
  (yyval.expr) = (yyvsp[-1].expr);
  if ((yyvsp[0].alias_t)) {
    (yyval.expr)->alias = strdup((yyvsp[0].alias_t)->name);
    delete (yyvsp[0].alias_t);
  }
}
#line 4868 "bison_parser.cpp"
    break;

  case 191: /* operand: '(' expr ')'  */
#line 992 "bison_parser.y"
                       { (yyval.expr) = (yyvsp[-1].expr); }
#line 4874 "bison_parser.cpp"
    break;

  case 201: /* operand: '(' select_no_paren ')'  */
#line 994 "bison_parser.y"
                                         {
  (yyval.expr) = Expr::makeSelect((yyvsp[-1].select_stmt));
}
#line 4882 "bison_parser.cpp"
    break;

  case 204: /* unary_expr: '-' operand  */
#line 1000 "bison_parser.y"
                         { (yyval.expr) = Expr::makeOpUnary(kOpUnaryMinus, (yyvsp[0].expr)); }
#line 4888 "bison_parser.cpp"
    break;

  case 205: /* unary_expr: NOT operand  */
#line 1001 "bison_parser.y"
              { (yyval.expr) = Expr::makeOpUnary(kOpNot, (yyvsp[0].expr)); }
#line 4894 "bison_parser.cpp"
    break;

  case 206: /* unary_expr: operand ISNULL  */
#line 1002 "bison_parser.y"
                 { (yyval.expr) = Expr::makeOpUnary(kOpIsNull, (yyvsp[-1].expr)); }
#line 4900 "bison_parser.cpp"
    break;

  case 207: /* unary_expr: operand IS NULL  */
#line 1003 "bison_parser.y"
                  { (yyval.expr) = Expr::makeOpUnary(kOpIsNull, (yyvsp[-2].expr)); }
#line 4906 "bison_parser.cpp"
    break;

  case 208: /* unary_expr: operand IS NOT NULL  */
#line 1004 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpUnary(kOpNot, Expr::makeOpUnary(kOpIsNull, (yyvsp[-3].expr))); }
#line 4912 "bison_parser.cpp"
    break;

  case 210: /* binary_expr: operand '-' operand  */
#line 1006 "bison_parser.y"
                                              { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpMinus, (yyvsp[0].expr)); }
#line 4918 "bison_parser.cpp"
    break;

  case 211: /* binary_expr: operand '+' operand  */
#line 1007 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpPlus, (yyvsp[0].expr)); }
#line 4924 "bison_parser.cpp"
    break;

  case 212: /* binary_expr: operand '/' operand  */
#line 1008 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpSlash, (yyvsp[0].expr)); }
#line 4930 "bison_parser.cpp"
    break;

  case 213: /* binary_expr: operand '*' operand  */
#line 1009 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpAsterisk, (yyvsp[0].expr)); }
#line 4936 "bison_parser.cpp"
    break;

  case 214: /* binary_expr: operand '%' operand  */
#line 1010 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpPercentage, (yyvsp[0].expr)); }
#line 4942 "bison_parser.cpp"
    break;

  case 215: /* binary_expr: operand '^' operand  */
#line 1011 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpCaret, (yyvsp[0].expr)); }
#line 4948 "bison_parser.cpp"
    break;

  case 216: /* binary_expr: operand LIKE operand  */
#line 1012 "bison_parser.y"
                       { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpLike, (yyvsp[0].expr)); }
#line 4954 "bison_parser.cpp"
    break;

  case 217: /* binary_expr: operand NOT LIKE operand  */
#line 1013 "bison_parser.y"
                           { (yyval.expr) = Expr::makeOpBinary((yyvsp[-3].expr), kOpNotLike, (yyvsp[0].expr)); }
#line 4960 "bison_parser.cpp"
    break;

  case 218: /* binary_expr: operand ILIKE operand  */
#line 1014 "bison_parser.y"
                        { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpILike, (yyvsp[0].expr)); }
#line 4966 "bison_parser.cpp"
    break;

  case 219: /* binary_expr: operand CONCAT operand  */
#line 1015 "bison_parser.y"
                         { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpConcat, (yyvsp[0].expr)); }
#line 4972 "bison_parser.cpp"
    break;

  case 220: /* logic_expr: expr AND expr  */
#line 1017 "bison_parser.y"
                           { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpAnd, (yyvsp[0].expr)); }
#line 4978 "bison_parser.cpp"
    break;

  case 221: /* logic_expr: expr OR expr  */
#line 1018 "bison_parser.y"
               { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpOr, (yyvsp[0].expr)); }
#line 4984 "bison_parser.cpp"
    break;

  case 222: /* in_expr: operand IN '(' expr_list ')'  */
#line 1020 "bison_parser.y"
                                       { (yyval.expr) = Expr::makeInOperator((yyvsp[-4].expr), (yyvsp[-1].expr_vec)); }
#line 4990 "bison_parser.cpp"
    break;

  case 223: /* in_expr: operand NOT IN '(' expr_list ')'  */
#line 1021 "bison_parser.y"
                                   { (yyval.expr) = Expr::makeOpUnary(kOpNot, Expr::makeInOperator((yyvsp[-5].expr), (yyvsp[-1].expr_vec))); }
#line 4996 "bison_parser.cpp"
    break;

  case 224: /* in_expr: operand IN '(' select_no_paren ')'  */
#line 1022 "bison_parser.y"
                                     { (yyval.expr) = Expr::makeInOperator((yyvsp[-4].expr), (yyvsp[-1].select_stmt)); }
#line 5002 "bison_parser.cpp"
    break;

  case 225: /* in_expr: operand NOT IN '(' select_no_paren ')'  */
#line 1023 "bison_parser.y"
                                         { (yyval.expr) = Expr::makeOpUnary(kOpNot, Expr::makeInOperator((yyvsp[-5].expr), (yyvsp[-1].select_stmt))); }
#line 5008 "bison_parser.cpp"
    break;

  case 226: /* case_expr: CASE expr case_list END  */
#line 1027 "bison_parser.y"
                                    { (yyval.expr) = Expr::makeCase((yyvsp[-2].expr), (yyvsp[-1].expr), nullptr); }
#line 5014 "bison_parser.cpp"
    break;

  case 227: /* case_expr: CASE expr case_list ELSE expr END  */
#line 1028 "bison_parser.y"
                                    { (yyval.expr) = Expr::makeCase((yyvsp[-4].expr), (yyvsp[-3].expr), (yyvsp[-1].expr)); }
#line 5020 "bison_parser.cpp"
    break;

  case 228: /* case_expr: CASE case_list END  */
#line 1029 "bison_parser.y"
                     { (yyval.expr) = Expr::makeCase(nullptr, (yyvsp[-1].expr), nullptr); }
#line 5026 "bison_parser.cpp"
    break;

  case 229: /* case_expr: CASE case_list ELSE expr END  */
#line 1030 "bison_parser.y"
                               { (yyval.expr) = Expr::makeCase(nullptr, (yyvsp[-3].expr), (yyvsp[-1].expr)); }
#line 5032 "bison_parser.cpp"
    break;

  case 230: /* case_list: WHEN expr THEN expr  */
#line 1032 "bison_parser.y"
                                { (yyval.expr) = Expr::makeCaseList(Expr::makeCaseListElement((yyvsp[-2].expr), (yyvsp[0].expr))); }
#line 5038 "bison_parser.cpp"
    break;

  case 231: /* case_list: case_list WHEN expr THEN expr  */
#line 1033 "bison_parser.y"
                                { (yyval.expr) = Expr::caseListAppend((yyvsp[-4].expr), Expr::makeCaseListElement((yyvsp[-2].expr), (yyvsp[0].expr))); }
#line 5044 "bison_parser.cpp"
    break;

  case 232: /* exists_expr: EXISTS '(' select_no_paren ')'  */
#line 1035 "bison_parser.y"
                                             { (yyval.expr) = Expr::makeExists((yyvsp[-1].select_stmt)); }
#line 5050 "bison_parser.cpp"
    break;

  case 233: /* exists_expr: NOT EXISTS '(' select_no_paren ')'  */
#line 1036 "bison_parser.y"
                                     { (yyval.expr) = Expr::makeOpUnary(kOpNot, Expr::makeExists((yyvsp[-1].select_stmt))); }
#line 5056 "bison_parser.cpp"
    break;

  case 234: /* comp_expr: operand '=' operand  */
#line 1038 "bison_parser.y"
                                { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpEquals, (yyvsp[0].expr)); }
#line 5062 "bison_parser.cpp"
    break;

  case 235: /* comp_expr: operand EQUALS operand  */
#line 1039 "bison_parser.y"
                         { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpEquals, (yyvsp[0].expr)); }
#line 5068 "bison_parser.cpp"
    break;

  case 236: /* comp_expr: operand NOTEQUALS operand  */
#line 1040 "bison_parser.y"
                            { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpNotEquals, (yyvsp[0].expr)); }
#line 5074 "bison_parser.cpp"
    break;

  case 237: /* comp_expr: operand '<' operand  */
#line 1041 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpLess, (yyvsp[0].expr)); }
#line 5080 "bison_parser.cpp"
    break;

  case 238: /* comp_expr: operand '>' operand  */
#line 1042 "bison_parser.y"
                      { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpGreater, (yyvsp[0].expr)); }
#line 5086 "bison_parser.cpp"
    break;

  case 239: /* comp_expr: operand LESSEQ operand  */
#line 1043 "bison_parser.y"
                         { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpLessEq, (yyvsp[0].expr)); }
#line 5092 "bison_parser.cpp"
    break;

  case 240: /* comp_expr: operand GREATEREQ operand  */
#line 1044 "bison_parser.y"
                            { (yyval.expr) = Expr::makeOpBinary((yyvsp[-2].expr), kOpGreaterEq, (yyvsp[0].expr)); }
#line 5098 "bison_parser.cpp"
    break;

  case 241: /* function_expr: IDENTIFIER '(' ')'  */
#line 1046 "bison_parser.y"
                                   { (yyval.expr) = Expr::makeFunctionRef((yyvsp[-2].sval), new std::vector<Expr*>(), false); }
#line 5104 "bison_parser.cpp"
    break;

  case 242: /* function_expr: IDENTIFIER '(' opt_distinct expr_list ')'  */
#line 1047 "bison_parser.y"
                                            { (yyval.expr) = Expr::makeFunctionRef((yyvsp[-4].sval), (yyvsp[-1].expr_vec), (yyvsp[-2].distinct_description_t)); }
#line 5110 "bison_parser.cpp"
    break;

  case 243: /* function_expr: IDENTIFIER '(' opt_distinct expr_pair_list ')'  */
#line 1048 "bison_parser.y"
                                                 { (yyval.expr) = Expr::makeFunctionRef((yyvsp[-4].sval), (yyvsp[-1].expr_map), (yyvsp[-2].distinct_description_t)); }
#line 5116 "bison_parser.cpp"
    break;

  case 244: /* function_expr: ALL '(' opt_distinct expr_list ')'  */
#line 1049 "bison_parser.y"
                                     { (yyval.expr) = Expr::makeFunctionRef(strdup("ALL"), (yyvsp[-1].expr_vec), (yyvsp[-2].distinct_description_t)); }
#line 5122 "bison_parser.cpp"
    break;

  case 245: /* extract_expr: EXTRACT '(' datetime_field FROM expr ')'  */
#line 1051 "bison_parser.y"
                                                        { (yyval.expr) = Expr::makeExtract((yyvsp[-3].datetime_field), (yyvsp[-1].expr)); }
#line 5128 "bison_parser.cpp"
    break;

  case 246: /* cast_expr: CAST '(' expr AS column_type ')'  */
#line 1053 "bison_parser.y"
                                             { (yyval.expr) = Expr::makeCast((yyvsp[-3].expr), (yyvsp[-1].column_type_t)); }
#line 5134 "bison_parser.cpp"
    break;

  case 247: /* datetime_field: SECONDS  */
#line 1055 "bison_parser.y"
                         { (yyval.datetime_field) = kDatetimeSecond; }
#line 5140 "bison_parser.cpp"
    break;

  case 248: /* datetime_field: MINUTES  */
#line 1056 "bison_parser.y"
          { (yyval.datetime_field) = kDatetimeMinute; }
#line 5146 "bison_parser.cpp"
    break;

  case 249: /* datetime_field: HOURS  */
#line 1057 "bison_parser.y"
        { (yyval.datetime_field) = kDatetimeHour; }
#line 5152 "bison_parser.cpp"
    break;

  case 250: /* datetime_field: DAYS  */
#line 1058 "bison_parser.y"
       { (yyval.datetime_field) = kDatetimeDay; }
#line 5158 "bison_parser.cpp"
    break;

  case 251: /* datetime_field: MONTHS  */
#line 1059 "bison_parser.y"
         { (yyval.datetime_field) = kDatetimeMonth; }
#line 5164 "bison_parser.cpp"
    break;

  case 252: /* datetime_field: YEARS  */
#line 1060 "bison_parser.y"
        { (yyval.datetime_field) = kDatetimeYear; }
#line 5170 "bison_parser.cpp"
    break;

  case 254: /* array_expr: ARRAY '[' expr_list ']'  */
#line 1064 "bison_parser.y"
                                     { (yyval.expr) = Expr::makeArray((yyvsp[-1].expr_vec)); }
#line 5176 "bison_parser.cpp"
    break;

  case 258: /* string_array_index: operand '[' string_literal ']'  */
#line 1068 "bison_parser.y"
                                                    { (yyval.expr) = Expr::makeArrayIndex((yyvsp[-3].expr), (yyvsp[-1].expr)->name); (yyvsp[-1].expr)->name = nullptr; delete (yyvsp[-1].expr); }
#line 5182 "bison_parser.cpp"
    break;

  case 259: /* fancy_array_index: operand '[' fancy_array_index_list ']'  */
#line 1069 "bison_parser.y"
                                                           { (yyval.expr) = Expr::makeArrayIndex((yyvsp[-3].expr), (yyvsp[-1].expr_vec)); }
#line 5188 "bison_parser.cpp"
    break;

  case 260: /* dynamic_array_index_operand: '(' expr ')'  */
#line 1071 "bison_parser.y"
                                           { (yyval.expr) = (yyvsp[-1].expr); }
#line 5194 "bison_parser.cpp"
    break;

  case 269: /* dynamic_array_index: operand '[' dynamic_array_index_operand ']'  */
#line 1072 "bison_parser.y"
                                                                  { (yyval.expr) = Expr::makeArrayIndex((yyvsp[-3].expr), (yyvsp[-1].expr)); }
#line 5200 "bison_parser.cpp"
    break;

  case 270: /* fancy_array_index_list: slice_literal  */
#line 1074 "bison_parser.y"
                                       {
  (yyval.expr_vec) = new std::vector<Expr*>();
  (yyval.expr_vec)->push_back((yyvsp[0].expr));
}
#line 5209 "bison_parser.cpp"
    break;

  case 271: /* fancy_array_index_list: fancy_array_index_list ',' slice_literal  */
#line 1077 "bison_parser.y"
                                             {
  (yyvsp[-2].expr_vec)->push_back((yyvsp[0].expr));
  (yyval.expr_vec) = (yyvsp[-2].expr_vec);
}
#line 5218 "bison_parser.cpp"
    break;

  case 280: /* slice_literal_0_0_0: ':'  */
#line 1084 "bison_parser.y"
                          { (yyval.expr) = Expr::makeSlice(nullptr, nullptr, nullptr); }
#line 5224 "bison_parser.cpp"
    break;

  case 281: /* slice_literal_0_0_0: ':' ':'  */
#line 1084 "bison_parser.y"
                                                                                         { (yyval.expr) = Expr::makeSlice(nullptr, nullptr, nullptr); }
#line 5230 "bison_parser.cpp"
    break;

  case 282: /* slice_literal_0_0_1: ':' ':' int_literal  */
#line 1085 "bison_parser.y"
                                          { (yyval.expr) = Expr::makeSlice(nullptr, nullptr, (yyvsp[0].expr)); }
#line 5236 "bison_parser.cpp"
    break;

  case 283: /* slice_literal_0_1_0: ':' int_literal  */
#line 1086 "bison_parser.y"
                                      { (yyval.expr) = Expr::makeSlice(nullptr, (yyvsp[0].expr), nullptr); }
#line 5242 "bison_parser.cpp"
    break;

  case 284: /* slice_literal_0_1_0: ':' int_literal ':'  */
#line 1086 "bison_parser.y"
                                                                                                            { (yyval.expr) = Expr::makeSlice(nullptr, (yyvsp[-1].expr), nullptr); }
#line 5248 "bison_parser.cpp"
    break;

  case 285: /* slice_literal_0_1_1: ':' int_literal ':' int_literal  */
#line 1087 "bison_parser.y"
                                                      { (yyval.expr) = Expr::makeSlice(nullptr, (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 5254 "bison_parser.cpp"
    break;

  case 288: /* slice_literal_1_0_0: int_literal ':' ':'  */
#line 1088 "bison_parser.y"
                                                                          { (yyval.expr) = Expr::makeSlice((yyvsp[-2].expr), nullptr, nullptr); }
#line 5260 "bison_parser.cpp"
    break;

  case 289: /* slice_literal_1_0_1: int_literal ':' ':' int_literal  */
#line 1089 "bison_parser.y"
                                                      { (yyval.expr) = Expr::makeSlice((yyvsp[-3].expr), nullptr, (yyvsp[0].expr)); }
#line 5266 "bison_parser.cpp"
    break;

  case 290: /* slice_literal_1_1_0: int_literal ':' int_literal  */
#line 1090 "bison_parser.y"
                                                  { (yyval.expr) = Expr::makeSlice((yyvsp[-2].expr), (yyvsp[0].expr), nullptr); }
#line 5272 "bison_parser.cpp"
    break;

  case 291: /* slice_literal_1_1_0: int_literal ':' int_literal ':'  */
#line 1090 "bison_parser.y"
                                                                                                                               { (yyval.expr) = Expr::makeSlice((yyvsp[-3].expr), (yyvsp[-1].expr), nullptr); }
#line 5278 "bison_parser.cpp"
    break;

  case 292: /* slice_literal_1_1_1: int_literal ':' int_literal ':' int_literal  */
#line 1091 "bison_parser.y"
                                                                  { (yyval.expr) = Expr::makeSlice((yyvsp[-4].expr), (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 5284 "bison_parser.cpp"
    break;

  case 293: /* between_expr: operand BETWEEN operand AND operand  */
#line 1093 "bison_parser.y"
                                                   { (yyval.expr) = Expr::makeBetween((yyvsp[-4].expr), (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 5290 "bison_parser.cpp"
    break;

  case 294: /* column_name: IDENTIFIER  */
#line 1095 "bison_parser.y"
                         { (yyval.expr) = Expr::makeColumnRef((yyvsp[0].sval)); }
#line 5296 "bison_parser.cpp"
    break;

  case 295: /* column_name: IDENTIFIER '.' IDENTIFIER  */
#line 1096 "bison_parser.y"
                            { (yyval.expr) = Expr::makeColumnRef((yyvsp[-2].sval), (yyvsp[0].sval)); }
#line 5302 "bison_parser.cpp"
    break;

  case 296: /* column_name: '*'  */
#line 1097 "bison_parser.y"
      { (yyval.expr) = Expr::makeStar(); }
#line 5308 "bison_parser.cpp"
    break;

  case 297: /* column_name: IDENTIFIER '.' '*'  */
#line 1098 "bison_parser.y"
                     { (yyval.expr) = Expr::makeStar((yyvsp[-2].sval)); }
#line 5314 "bison_parser.cpp"
    break;

  case 305: /* string_literal: STRING  */
#line 1102 "bison_parser.y"
                        { (yyval.expr) = Expr::makeLiteral((yyvsp[0].sval)); }
#line 5320 "bison_parser.cpp"
    break;

  case 306: /* bool_literal: TRUE  */
#line 1104 "bison_parser.y"
                    { (yyval.expr) = Expr::makeLiteral(true); }
#line 5326 "bison_parser.cpp"
    break;

  case 307: /* bool_literal: FALSE  */
#line 1105 "bison_parser.y"
        { (yyval.expr) = Expr::makeLiteral(false); }
#line 5332 "bison_parser.cpp"
    break;

  case 308: /* num_literal: FLOATVAL  */
#line 1107 "bison_parser.y"
                       { (yyval.expr) = Expr::makeLiteral((yyvsp[0].fval)); }
#line 5338 "bison_parser.cpp"
    break;

  case 310: /* int_literal: INTVAL  */
#line 1110 "bison_parser.y"
                     { (yyval.expr) = Expr::makeLiteral((yyvsp[0].ival)); }
#line 5344 "bison_parser.cpp"
    break;

  case 311: /* null_literal: NULL  */
#line 1112 "bison_parser.y"
                    { (yyval.expr) = Expr::makeNullLiteral(); }
#line 5350 "bison_parser.cpp"
    break;

  case 312: /* date_literal: DATE STRING  */
#line 1114 "bison_parser.y"
                           {
  int day{0}, month{0}, year{0}, chars_parsed{0};
  // If the whole string is parsed, chars_parsed points to the terminating null byte after the last character
  if (sscanf((yyvsp[0].sval), "%4d-%2d-%2d%n", &day, &month, &year, &chars_parsed) != 3 || (yyvsp[0].sval)[chars_parsed] != 0) {
    free((yyvsp[0].sval));
    yyerror(&yyloc, result, scanner, "Found incorrect date format. Expected format: YYYY-MM-DD");
    YYERROR;
  }
  (yyval.expr) = Expr::makeDateLiteral((yyvsp[0].sval));
}
#line 5365 "bison_parser.cpp"
    break;

  case 313: /* interval_literal: int_literal duration_field  */
#line 1125 "bison_parser.y"
                                              {
  (yyval.expr) = Expr::makeIntervalLiteral((yyvsp[-1].expr)->ival, (yyvsp[0].datetime_field));
  delete (yyvsp[-1].expr);
}
#line 5374 "bison_parser.cpp"
    break;

  case 314: /* interval_literal: INTERVAL STRING datetime_field  */
#line 1129 "bison_parser.y"
                                 {
  int duration{0}, chars_parsed{0};
  // If the whole string is parsed, chars_parsed points to the terminating null byte after the last character
  if (sscanf((yyvsp[-1].sval), "%d%n", &duration, &chars_parsed) != 1 || (yyvsp[-1].sval)[chars_parsed] != 0) {
    free((yyvsp[-1].sval));
    yyerror(&yyloc, result, scanner, "Found incorrect interval format. Expected format: INTEGER");
    YYERROR;
  }
  free((yyvsp[-1].sval));
  (yyval.expr) = Expr::makeIntervalLiteral(duration, (yyvsp[0].datetime_field));
}
#line 5390 "bison_parser.cpp"
    break;

  case 315: /* interval_literal: INTERVAL STRING  */
#line 1140 "bison_parser.y"
                  {
  int duration{0}, chars_parsed{0};
  // 'seconds' and 'minutes' are the longest accepted interval qualifiers (7 chars) + null byte
  char unit_string[8];
  // If the whole string is parsed, chars_parsed points to the terminating null byte after the last character
  if (sscanf((yyvsp[0].sval), "%d %7s%n", &duration, unit_string, &chars_parsed) != 2 || (yyvsp[0].sval)[chars_parsed] != 0) {
    free((yyvsp[0].sval));
    yyerror(&yyloc, result, scanner, "Found incorrect interval format. Expected format: INTEGER INTERVAL_QUALIIFIER");
    YYERROR;
  }
  free((yyvsp[0].sval));

  DatetimeField unit;
  if (strcasecmp(unit_string, "second") == 0 || strcasecmp(unit_string, "seconds") == 0) {
    unit = kDatetimeSecond;
  } else if (strcasecmp(unit_string, "minute") == 0 || strcasecmp(unit_string, "minutes") == 0) {
    unit = kDatetimeMinute;
  } else if (strcasecmp(unit_string, "hour") == 0 || strcasecmp(unit_string, "hours") == 0) {
    unit = kDatetimeHour;
  } else if (strcasecmp(unit_string, "day") == 0 || strcasecmp(unit_string, "days") == 0) {
    unit = kDatetimeDay;
  } else if (strcasecmp(unit_string, "month") == 0 || strcasecmp(unit_string, "months") == 0) {
    unit = kDatetimeMonth;
  } else if (strcasecmp(unit_string, "year") == 0 || strcasecmp(unit_string, "years") == 0) {
    unit = kDatetimeYear;
  } else {
    yyerror(&yyloc, result, scanner, "Interval qualifier is unknown.");
    YYERROR;
  }
  (yyval.expr) = Expr::makeIntervalLiteral(duration, unit);
}
#line 5426 "bison_parser.cpp"
    break;

  case 316: /* param_expr: '?'  */
#line 1172 "bison_parser.y"
                 {
  (yyval.expr) = Expr::makeParameter(yylloc.total_column);
  (yyval.expr)->ival2 = yyloc.param_list.size();
  yyloc.param_list.push_back((yyval.expr));
}
#line 5436 "bison_parser.cpp"
    break;

  case 318: /* table_ref: table_ref_commalist ',' table_ref_atomic  */
#line 1181 "bison_parser.y"
                                                                        {
  (yyvsp[-2].table_vec)->push_back((yyvsp[0].table));
  auto tbl = new TableRef(kTableCrossProduct);
  tbl->list = (yyvsp[-2].table_vec);
  (yyval.table) = tbl;
}
#line 5447 "bison_parser.cpp"
    break;

  case 322: /* nonjoin_table_ref_atomic: '(' select_statement ')' opt_table_alias  */
#line 1190 "bison_parser.y"
                                                                                     {
  auto tbl = new TableRef(kTableSelect);
  tbl->select = (yyvsp[-2].select_stmt);
  tbl->alias = (yyvsp[0].alias_t);
  (yyval.table) = tbl;
}
#line 5458 "bison_parser.cpp"
    break;

  case 323: /* table_ref_commalist: table_ref_atomic  */
#line 1197 "bison_parser.y"
                                       {
  (yyval.table_vec) = new std::vector<TableRef*>();
  (yyval.table_vec)->push_back((yyvsp[0].table));
}
#line 5467 "bison_parser.cpp"
    break;

  case 324: /* table_ref_commalist: table_ref_commalist ',' table_ref_atomic  */
#line 1201 "bison_parser.y"
                                           {
  (yyvsp[-2].table_vec)->push_back((yyvsp[0].table));
  (yyval.table_vec) = (yyvsp[-2].table_vec);
}
#line 5476 "bison_parser.cpp"
    break;

  case 325: /* table_ref_name: table_name opt_table_alias  */
#line 1206 "bison_parser.y"
                                            {
  auto tbl = new TableRef(kTableName);
  tbl->schema = (yyvsp[-1].table_name).schema;
  tbl->name = (yyvsp[-1].table_name).name;
  tbl->alias = (yyvsp[0].alias_t);
  (yyval.table) = tbl;
}
#line 5488 "bison_parser.cpp"
    break;

  case 326: /* table_ref_name_no_alias: table_name  */
#line 1214 "bison_parser.y"
                                     {
  (yyval.table) = new TableRef(kTableName);
  (yyval.table)->schema = (yyvsp[0].table_name).schema;
  (yyval.table)->name = (yyvsp[0].table_name).name;
}
#line 5498 "bison_parser.cpp"
    break;

  case 327: /* table_name: IDENTIFIER  */
#line 1220 "bison_parser.y"
                        {
  (yyval.table_name).schema = nullptr;
  (yyval.table_name).name = (yyvsp[0].sval);
}
#line 5507 "bison_parser.cpp"
    break;

  case 328: /* table_name: IDENTIFIER '.' IDENTIFIER  */
#line 1224 "bison_parser.y"
                            {
  (yyval.table_name).schema = (yyvsp[-2].sval);
  (yyval.table_name).name = (yyvsp[0].sval);
}
#line 5516 "bison_parser.cpp"
    break;

  case 329: /* opt_index_name: IDENTIFIER  */
#line 1229 "bison_parser.y"
                            { (yyval.sval) = (yyvsp[0].sval); }
#line 5522 "bison_parser.cpp"
    break;

  case 330: /* opt_index_name: %empty  */
#line 1230 "bison_parser.y"
              { (yyval.sval) = nullptr; }
#line 5528 "bison_parser.cpp"
    break;

  case 332: /* table_alias: AS IDENTIFIER '(' ident_commalist ')'  */
#line 1232 "bison_parser.y"
                                                            { (yyval.alias_t) = new Alias((yyvsp[-3].sval), (yyvsp[-1].str_vec)); }
#line 5534 "bison_parser.cpp"
    break;

  case 334: /* opt_table_alias: %empty  */
#line 1234 "bison_parser.y"
                                            { (yyval.alias_t) = nullptr; }
#line 5540 "bison_parser.cpp"
    break;

  case 335: /* alias: AS IDENTIFIER  */
#line 1236 "bison_parser.y"
                      { (yyval.alias_t) = new Alias((yyvsp[0].sval)); }
#line 5546 "bison_parser.cpp"
    break;

  case 336: /* alias: IDENTIFIER  */
#line 1237 "bison_parser.y"
             { (yyval.alias_t) = new Alias((yyvsp[0].sval)); }
#line 5552 "bison_parser.cpp"
    break;

  case 338: /* opt_alias: %empty  */
#line 1239 "bison_parser.y"
                                { (yyval.alias_t) = nullptr; }
#line 5558 "bison_parser.cpp"
    break;

  case 339: /* opt_locking_clause: opt_locking_clause_list  */
#line 1245 "bison_parser.y"
                                             { (yyval.locking_clause_vec) = (yyvsp[0].locking_clause_vec); }
#line 5564 "bison_parser.cpp"
    break;

  case 340: /* opt_locking_clause: %empty  */
#line 1246 "bison_parser.y"
              { (yyval.locking_clause_vec) = nullptr; }
#line 5570 "bison_parser.cpp"
    break;

  case 341: /* opt_locking_clause_list: locking_clause  */
#line 1248 "bison_parser.y"
                                         {
  (yyval.locking_clause_vec) = new std::vector<LockingClause*>();
  (yyval.locking_clause_vec)->push_back((yyvsp[0].locking_t));
}
#line 5579 "bison_parser.cpp"
    break;

  case 342: /* opt_locking_clause_list: opt_locking_clause_list locking_clause  */
#line 1252 "bison_parser.y"
                                         {
  (yyvsp[-1].locking_clause_vec)->push_back((yyvsp[0].locking_t));
  (yyval.locking_clause_vec) = (yyvsp[-1].locking_clause_vec);
}
#line 5588 "bison_parser.cpp"
    break;

  case 343: /* locking_clause: FOR row_lock_mode opt_row_lock_policy  */
#line 1257 "bison_parser.y"
                                                       {
  (yyval.locking_t) = new LockingClause();
  (yyval.locking_t)->rowLockMode = (yyvsp[-1].lock_mode_t);
  (yyval.locking_t)->rowLockWaitPolicy = (yyvsp[0].lock_wait_policy_t);
  (yyval.locking_t)->tables = nullptr;
}
#line 5599 "bison_parser.cpp"
    break;

  case 344: /* locking_clause: FOR row_lock_mode OF ident_commalist opt_row_lock_policy  */
#line 1263 "bison_parser.y"
                                                           {
  (yyval.locking_t) = new LockingClause();
  (yyval.locking_t)->rowLockMode = (yyvsp[-3].lock_mode_t);
  (yyval.locking_t)->tables = (yyvsp[-1].str_vec);
  (yyval.locking_t)->rowLockWaitPolicy = (yyvsp[0].lock_wait_policy_t);
}
#line 5610 "bison_parser.cpp"
    break;

  case 345: /* row_lock_mode: UPDATE  */
#line 1270 "bison_parser.y"
                       { (yyval.lock_mode_t) = RowLockMode::ForUpdate; }
#line 5616 "bison_parser.cpp"
    break;

  case 346: /* row_lock_mode: NO KEY UPDATE  */
#line 1271 "bison_parser.y"
                { (yyval.lock_mode_t) = RowLockMode::ForNoKeyUpdate; }
#line 5622 "bison_parser.cpp"
    break;

  case 347: /* row_lock_mode: SHARE  */
#line 1272 "bison_parser.y"
        { (yyval.lock_mode_t) = RowLockMode::ForShare; }
#line 5628 "bison_parser.cpp"
    break;

  case 348: /* row_lock_mode: KEY SHARE  */
#line 1273 "bison_parser.y"
            { (yyval.lock_mode_t) = RowLockMode::ForKeyShare; }
#line 5634 "bison_parser.cpp"
    break;

  case 349: /* opt_row_lock_policy: SKIP LOCKED  */
#line 1275 "bison_parser.y"
                                  { (yyval.lock_wait_policy_t) = RowLockWaitPolicy::SkipLocked; }
#line 5640 "bison_parser.cpp"
    break;

  case 350: /* opt_row_lock_policy: NOWAIT  */
#line 1276 "bison_parser.y"
         { (yyval.lock_wait_policy_t) = RowLockWaitPolicy::NoWait; }
#line 5646 "bison_parser.cpp"
    break;

  case 351: /* opt_row_lock_policy: %empty  */
#line 1277 "bison_parser.y"
              { (yyval.lock_wait_policy_t) = RowLockWaitPolicy::None; }
#line 5652 "bison_parser.cpp"
    break;

  case 353: /* opt_with_clause: %empty  */
#line 1283 "bison_parser.y"
                                            { (yyval.with_description_vec) = nullptr; }
#line 5658 "bison_parser.cpp"
    break;

  case 354: /* with_clause: WITH with_description_list  */
#line 1285 "bison_parser.y"
                                         { (yyval.with_description_vec) = (yyvsp[0].with_description_vec); }
#line 5664 "bison_parser.cpp"
    break;

  case 355: /* with_description_list: with_description  */
#line 1287 "bison_parser.y"
                                         {
  (yyval.with_description_vec) = new std::vector<WithDescription*>();
  (yyval.with_description_vec)->push_back((yyvsp[0].with_description_t));
}
#line 5673 "bison_parser.cpp"
    break;

  case 356: /* with_description_list: with_description_list ',' with_description  */
#line 1291 "bison_parser.y"
                                             {
  (yyvsp[-2].with_description_vec)->push_back((yyvsp[0].with_description_t));
  (yyval.with_description_vec) = (yyvsp[-2].with_description_vec);
}
#line 5682 "bison_parser.cpp"
    break;

  case 357: /* with_description: IDENTIFIER AS select_with_paren  */
#line 1296 "bison_parser.y"
                                                   {
  (yyval.with_description_t) = new WithDescription();
  (yyval.with_description_t)->alias = (yyvsp[-2].sval);
  (yyval.with_description_t)->select = (yyvsp[0].select_stmt);
}
#line 5692 "bison_parser.cpp"
    break;

  case 358: /* join_clause: table_ref_atomic NATURAL JOIN nonjoin_table_ref_atomic  */
#line 1306 "bison_parser.y"
                                                                     {
  (yyval.table) = new TableRef(kTableJoin);
  (yyval.table)->join = new JoinDefinition();
  (yyval.table)->join->type = kJoinNatural;
  (yyval.table)->join->left = (yyvsp[-3].table);
  (yyval.table)->join->right = (yyvsp[0].table);
}
#line 5704 "bison_parser.cpp"
    break;

  case 359: /* join_clause: table_ref_atomic opt_join_type JOIN table_ref_atomic ON join_condition  */
#line 1313 "bison_parser.y"
                                                                         {
  (yyval.table) = new TableRef(kTableJoin);
  (yyval.table)->join = new JoinDefinition();
  (yyval.table)->join->type = (JoinType)(yyvsp[-4].join_type);
  (yyval.table)->join->left = (yyvsp[-5].table);
  (yyval.table)->join->right = (yyvsp[-2].table);
  (yyval.table)->join->condition = (yyvsp[0].expr);
}
#line 5717 "bison_parser.cpp"
    break;

  case 360: /* join_clause: table_ref_atomic opt_join_type JOIN table_ref_atomic USING '(' column_name ')'  */
#line 1321 "bison_parser.y"
                                                                                 {
  (yyval.table) = new TableRef(kTableJoin);
  (yyval.table)->join = new JoinDefinition();
  (yyval.table)->join->type = (JoinType)(yyvsp[-6].join_type);
  (yyval.table)->join->left = (yyvsp[-7].table);
  (yyval.table)->join->right = (yyvsp[-4].table);
  auto left_col = Expr::makeColumnRef(strdup((yyvsp[-1].expr)->name));
  if ((yyvsp[-1].expr)->alias != nullptr) left_col->alias = strdup((yyvsp[-1].expr)->alias);
  if ((yyvsp[-7].table)->getName() != nullptr) left_col->table = strdup((yyvsp[-7].table)->getName());
  auto right_col = Expr::makeColumnRef(strdup((yyvsp[-1].expr)->name));
  if ((yyvsp[-1].expr)->alias != nullptr) right_col->alias = strdup((yyvsp[-1].expr)->alias);
  if ((yyvsp[-4].table)->getName() != nullptr) right_col->table = strdup((yyvsp[-4].table)->getName());
  (yyval.table)->join->condition = Expr::makeOpBinary(left_col, kOpEquals, right_col);
  delete (yyvsp[-1].expr);
}
#line 5737 "bison_parser.cpp"
    break;

  case 361: /* opt_join_type: INNER  */
#line 1337 "bison_parser.y"
                      { (yyval.join_type) = kJoinInner; }
#line 5743 "bison_parser.cpp"
    break;

  case 362: /* opt_join_type: LEFT OUTER  */
#line 1338 "bison_parser.y"
             { (yyval.join_type) = kJoinLeft; }
#line 5749 "bison_parser.cpp"
    break;

  case 363: /* opt_join_type: LEFT  */
#line 1339 "bison_parser.y"
       { (yyval.join_type) = kJoinLeft; }
#line 5755 "bison_parser.cpp"
    break;

  case 364: /* opt_join_type: RIGHT OUTER  */
#line 1340 "bison_parser.y"
              { (yyval.join_type) = kJoinRight; }
#line 5761 "bison_parser.cpp"
    break;

  case 365: /* opt_join_type: RIGHT  */
#line 1341 "bison_parser.y"
        { (yyval.join_type) = kJoinRight; }
#line 5767 "bison_parser.cpp"
    break;

  case 366: /* opt_join_type: FULL OUTER  */
#line 1342 "bison_parser.y"
             { (yyval.join_type) = kJoinFull; }
#line 5773 "bison_parser.cpp"
    break;

  case 367: /* opt_join_type: OUTER  */
#line 1343 "bison_parser.y"
        { (yyval.join_type) = kJoinFull; }
#line 5779 "bison_parser.cpp"
    break;

  case 368: /* opt_join_type: FULL  */
#line 1344 "bison_parser.y"
       { (yyval.join_type) = kJoinFull; }
#line 5785 "bison_parser.cpp"
    break;

  case 369: /* opt_join_type: CROSS  */
#line 1345 "bison_parser.y"
        { (yyval.join_type) = kJoinCross; }
#line 5791 "bison_parser.cpp"
    break;

  case 370: /* opt_join_type: %empty  */
#line 1346 "bison_parser.y"
                       { (yyval.join_type) = kJoinInner; }
#line 5797 "bison_parser.cpp"
    break;

  case 374: /* ident_commalist: IDENTIFIER  */
#line 1357 "bison_parser.y"
                             {
  (yyval.str_vec) = new std::vector<char*>();
  (yyval.str_vec)->push_back((yyvsp[0].sval));
}
#line 5806 "bison_parser.cpp"
    break;

  case 375: /* ident_commalist: ident_commalist ',' IDENTIFIER  */
#line 1361 "bison_parser.y"
                                 {
  (yyvsp[-2].str_vec)->push_back((yyvsp[0].sval));
  (yyval.str_vec) = (yyvsp[-2].str_vec);
}
#line 5815 "bison_parser.cpp"
    break;


#line 5819 "bison_parser.cpp"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == SQL_HSQL_EMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yytoken, &yylloc};
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == -1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *,
                             YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (yymsg)
              {
                yysyntax_error_status
                  = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
                yymsgp = yymsg;
              }
            else
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = YYENOMEM;
              }
          }
        yyerror (&yylloc, result, scanner, yymsgp);
        if (yysyntax_error_status == YYENOMEM)
          YYNOMEM;
      }
    }

  yyerror_range[1] = yylloc;
  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= SQL_YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == SQL_YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc, result, scanner);
          yychar = SQL_HSQL_EMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, yylsp, result, scanner);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  ++yylsp;
  YYLLOC_DEFAULT (*yylsp, yyerror_range, 2);

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (&yylloc, result, scanner, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != SQL_HSQL_EMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc, result, scanner);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, yylsp, result, scanner);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
  return yyresult;
}

#line 1367 "bison_parser.y"

    // clang-format on
    /*********************************
 ** Section 4: Additional C code
 *********************************/

    /* empty */
