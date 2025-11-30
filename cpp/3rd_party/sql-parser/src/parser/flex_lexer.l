/**
 * lexer
 *
 *
 */


/***************************
 ** Section 1: Definitions
 ***************************/
%{

#include "../sql/Expr.h"
#include "bison_parser.h"
#include <climits>
#include <stdio.h>
#include <sstream>

#define TOKEN(name) { return SQL_##name; }

static thread_local std::stringstream strbuf;

%}
%x singlequotedstring

/***************************
 ** Section 2: Rules
 ***************************/

/* Define the output files */
%option header-file="flex_lexer.h"
%option outfile="flex_lexer.cpp"

/* Make reentrant */
%option reentrant
%option bison-bridge

/* performance tweeks */
%option never-interactive
%option batch

/* other flags */
%option noyywrap
%option nounput
%option warn
%option case-insensitive
%option prefix="hsql_"
%option bison-locations
/* %option nodefault */


%s COMMENT

/***************************
 ** Section 3: Rules
 ***************************/
%%

--              BEGIN(COMMENT);
<COMMENT>[^\n]* /* skipping comment content until a end of line is read */;
<COMMENT>\n     BEGIN(INITIAL);

[ \t\n]+        /* skip whitespace */;

DEALLOCATE  TOKEN(DEALLOCATE)
PARAMETERS  TOKEN(PARAMETERS)
INTERSECT   TOKEN(INTERSECT)
TEMPORARY   TOKEN(TEMPORARY)
TIMESTAMP   TOKEN(TIMESTAMP)
DESCRIBE    TOKEN(DESCRIBE)
DISTINCT    TOKEN(DISTINCT)
NVARCHAR    TOKEN(NVARCHAR)
RESTRICT    TOKEN(RESTRICT)
TRUNCATE    TOKEN(TRUNCATE)
ANALYZE     TOKEN(ANALYZE)
BETWEEN     TOKEN(BETWEEN)
CASCADE     TOKEN(CASCADE)
COLUMNS     TOKEN(COLUMNS)
CONTROL     TOKEN(CONTROL)
DEFAULT     TOKEN(DEFAULT)
EXECUTE     TOKEN(EXECUTE)
EXPLAIN     TOKEN(EXPLAIN)
INTEGER     TOKEN(INTEGER)
NATURAL     TOKEN(NATURAL)
PREPARE     TOKEN(PREPARE)
PRIMARY     TOKEN(PRIMARY)
SCHEMAS     TOKEN(SCHEMAS)
SPATIAL     TOKEN(SPATIAL)
VARCHAR     TOKEN(VARCHAR)
TIME        TOKEN(TIME)
DECIMAL     TOKEN(DECIMAL)
REAL        TOKEN(REAL)
SMALLINT    TOKEN(SMALLINT)
BIGINT      TOKEN(BIGINT)
VIRTUAL     TOKEN(VIRTUAL)
BEFORE      TOKEN(BEFORE)
COLUMN      TOKEN(COLUMN)
CREATE      TOKEN(CREATE)
DELETE      TOKEN(DELETE)
DIRECT      TOKEN(DIRECT)
DOUBLE      TOKEN(DOUBLE)
ESCAPE      TOKEN(ESCAPE)
EXCEPT      TOKEN(EXCEPT)
EXISTS      TOKEN(EXISTS)
EXTRACT     TOKEN(EXTRACT)
CAST        TOKEN(CAST)
FORMAT      TOKEN(FORMAT)
GLOBAL      TOKEN(GLOBAL)
HAVING      TOKEN(HAVING)
IMPORT      TOKEN(IMPORT)
INSERT      TOKEN(INSERT)
ISNULL      TOKEN(ISNULL)
OFFSET      TOKEN(OFFSET)
RENAME      TOKEN(RENAME)
SCHEMA      TOKEN(SCHEMA)
SELECT      TOKEN(SELECT)
SORTED      TOKEN(SORTED)
TABLES      TOKEN(TABLES)
UNIQUE      TOKEN(UNIQUE)
UNLOAD      TOKEN(UNLOAD)
UPDATE      TOKEN(UPDATE)
VALUES      TOKEN(VALUES)
AFTER       TOKEN(AFTER)
ALTER       TOKEN(ALTER)
ARRAY       TOKEN(ARRAY)
CROSS       TOKEN(CROSS)
DELTA       TOKEN(DELTA)
FLOAT       TOKEN(FLOAT)
GROUP       TOKEN(GROUP)
UNGROUP     TOKEN(UNGROUP)
SPLIT       TOKEN(SPLIT)
INDEX       TOKEN(INDEX)
INNER       TOKEN(INNER)
LIMIT       TOKEN(LIMIT)
LOCAL       TOKEN(LOCAL)
MERGE       TOKEN(MERGE)
MINUS       TOKEN(MINUS)
ORDER       TOKEN(ORDER)
OUTER       TOKEN(OUTER)
PERCENT     TOKEN(PERCENT)
REPLACE     TOKEN(REPLACE)
RIGHT       TOKEN(RIGHT)
SAMPLE      TOKEN(SAMPLE)
TABLE       TOKEN(TABLE)
UNION       TOKEN(UNION)
USING       TOKEN(USING)
WHERE       TOKEN(WHERE)
EXPAND      TOKEN(EXPAND)
OVERLAP     TOKEN(OVERLAP)
CALL        TOKEN(CALL)
CASE        TOKEN(CASE)
CHAR        TOKEN(CHAR)
COPY        TOKEN(COPY)
DATE        TOKEN(DATE)
DATETIME    TOKEN(DATETIME)
DESC        TOKEN(DESC)
DROP        TOKEN(DROP)
ELSE        TOKEN(ELSE)
FILE        TOKEN(FILE)
FROM        TOKEN(FROM)
FULL        TOKEN(FULL)
HASH        TOKEN(HASH)
HINT        TOKEN(HINT)
INTO        TOKEN(INTO)
JOIN        TOKEN(JOIN)
LEFT        TOKEN(LEFT)
LIKE        TOKEN(LIKE)
ILIKE       TOKEN(ILIKE)
LOAD        TOKEN(LOAD)
LONG        TOKEN(LONG)
NULL        TOKEN(NULL)
PLAN        TOKEN(PLAN)
SHOW        TOKEN(SHOW)
TEXT_INTERNAL_TQL        TOKEN(TEXT_INTERNAL_TQL)
THEN        TOKEN(THEN)
VIEW        TOKEN(VIEW)
WHEN        TOKEN(WHEN)
WITH        TOKEN(WITH)
ADD         TOKEN(ADD)
ALL         TOKEN(ALL)
AND         TOKEN(AND)
ASC         TOKEN(ASC)
END         TOKEN(END)
FOR         TOKEN(FOR)
INT         TOKEN(INT)
KEY         TOKEN(KEY)
NOT         TOKEN(NOT)
OFF         TOKEN(OFF)
SET         TOKEN(SET)
TOP         TOKEN(TOP)
AS          TOKEN(AS)
BY          TOKEN(BY)
IF          TOKEN(IF)
IN          TOKEN(IN)
IS          TOKEN(IS)
OF          TOKEN(OF)
ON          TOKEN(ON)
OR          TOKEN(OR)
TO          TOKEN(TO)
NO          TOKEN(NO)
SECONDS     TOKEN(SECONDS)
MINUTES     TOKEN(MINUTES)
HOURS       TOKEN(HOURS)
DAYS        TOKEN(DAYS)
MONTHS      TOKEN(MONTHS)
YEARS       TOKEN(YEARS)
TRUE        TOKEN(TRUE)
FALSE       TOKEN(FALSE)
TRANSACTION TOKEN(TRANSACTION)
BEGIN       TOKEN(BEGIN)
ROLLBACK    TOKEN(ROLLBACK)
COMMIT      TOKEN(COMMIT)
INTERVAL    TOKEN(INTERVAL)
SHARE       TOKEN(SHARE)
NOWAIT      TOKEN(NOWAIT)
SKIP        TOKEN(SKIP)
LOCKED      TOKEN(LOCKED)
ACROSS      TOKEN(ACROSS)
SPACE       TOKEN(SPACE)

CHARACTER[ \t\n]+VARYING TOKEN(CHARACTER_VARYING)

            /* Allow =/== see https://sqlite.org/lang_expr.html#collateop */
"=="        TOKEN(EQUALS)
"!="        TOKEN(NOTEQUALS)
"<>"        TOKEN(NOTEQUALS)
"<="        TOKEN(LESSEQ)
">="        TOKEN(GREATEREQ)
"||"        TOKEN(CONCAT)

[-+*/(){},.;<>=^%:?[\]|]    { return yytext[0]; }

[0-9]+"."[0-9]* |
"."[0-9]+     |
[0-9]+E[-+]?[0-9]+    |
[0-9]+"."[0-9]*E[-+]?[0-9]+ |
"."[0-9]+E[-+]?[0-9]+ {
  yylval->fval = atof(yytext) ;
  return SQL_FLOATVAL;
}

  /*
   * Regularly, negative literals are treated as <unary minus> <positive literal>. This does not work for LLONG_MIN, as it has no
   * positive equivalent. We thus match for LLONG_MIN specifically. This is not an issue for floats, where
   *   numeric_limits<double>::lowest() == -numeric_limits<double>::max();
   */
-9223372036854775808 {
  yylval->ival = LLONG_MIN;
  return SQL_INTVAL;
}

[0-9]+ {
  errno = 0;
  yylval->ival = strtoll(yytext, nullptr, 0);
  if (errno) {
    return fprintf(stderr, "[SQL-Lexer-Error] Integer cannot be parsed - is it out of range?");
    return 0;
  }
  return SQL_INTVAL;
}

\"[^\"\n]+\" {
  // Crop the leading and trailing quote char
  yylval->sval = hsql::substr(yytext, 1, strlen(yytext)-1);
  return SQL_IDENTIFIER;
}

[A-Za-z][A-Za-z0-9_]* {
  yylval->sval = strdup(yytext);
  return SQL_IDENTIFIER;
}

\'                          { BEGIN singlequotedstring; strbuf.clear(); strbuf.str(""); }  // Clear strbuf manually, see #170
<singlequotedstring>\'\'    { strbuf << '\''; }
<singlequotedstring>[^']*   { strbuf << yytext; }
<singlequotedstring>\'      { BEGIN 0; yylval->sval = strdup(strbuf.str().c_str()); return SQL_STRING; }
<singlequotedstring><<EOF>> { fprintf(stderr, "[SQL-Lexer-Error] Unterminated string\n"); return 0; }

. { fprintf(stderr, "[SQL-Lexer-Error] Unknown Character: %c\n", yytext[0]); return 0; }

%%
/***************************
 ** Section 3: User code
 ***************************/

int yyerror(const char *msg) {
    fprintf(stderr, "[SQL-Lexer-Error] %s\n",msg); return 0;
}
