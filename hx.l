%{
/*
 * hx.l - lexer for the hx hash expression language
 *
 * Tokens: identifiers, numbers, strings, operators, keywords.
 * Newlines are statement separators (SEP) at brace depth 0.
 * Inside { }, newlines are whitespace; use ; to separate statements.
 * Comments: # to end of line.
 */

#include "hx_ast.h"
#include "hx.tab.h"
#include <string.h>
#include <stdlib.h>

int hx_line = 1;
static int brace_depth = 0;
%}

%option noyywrap
%option nounput
%option noinput

%%

"#"[^\n]*           { /* comment — skip to end of line */ }
[ \t\r]+            { /* whitespace */ }

\n                  { hx_line++;
                      if (brace_depth == 0)
                          return SEP;
                      /* inside braces, newline is whitespace */
                    }

";"                 { return SEP; }

 /* ---- keywords ---- */
"for"               { return FOR; }
"to"                { return TO; }
"if"                { return IF; }
"else"              { return ELSE; }

 /* ---- built-in variables (reserved words) ---- */
"pass"              { yylval.str = strdup("pass");    return PASS; }
"salt"              { yylval.str = strdup("salt");    return SALT; }
"salt2"             { yylval.str = strdup("salt2");   return SALT2; }
"pepper"            { yylval.str = strdup("pepper");  return PEPPER; }
"user"              { yylval.str = strdup("user");    return USER; }

 /* ---- literals ---- */
"-"[0-9]+           { yylval.num = strtol(yytext, NULL, 10); return NUMBER; }
[0-9]+              { yylval.num = strtol(yytext, NULL, 10); return NUMBER; }

\"([^\"\n\\]|\\.)*\" {
                      /* strip quotes, handle backslash escapes */
                      yytext[yyleng - 1] = '\0';
                      yylval.str = strdup(yytext + 1);
                      return STRING;
                    }

\'([^\'\n\\]|\\.)*\' {
                      yytext[yyleng - 1] = '\0';
                      yylval.str = strdup(yytext + 1);
                      return STRING;
                    }

 /* ---- identifiers (hash names, variable names) ---- */
[a-zA-Z_][a-zA-Z0-9_]* {
                      yylval.str = strdup(yytext);
                      return IDENT;
                    }

 /* ---- multi-char operators ---- */
"=="                { return EQ; }
"!="                { return NE; }
"<="                { return LE; }
">="                { return GE; }

 /* ---- single-char operators and delimiters ---- */
"."                 { return '.'; }
"^"                 { return '^'; }
"="                 { return '='; }
","                 { return ','; }
"("                 { return '('; }
")"                 { return ')'; }
"{"                 { brace_depth++; return '{'; }
"}"                 { brace_depth--; return '}'; }
"<"                 { return '<'; }
">"                 { return '>'; }
"%"                 { return '%'; }

 /* ---- catch-all ---- */
.                   { fprintf(stderr, "hx:%d: unexpected character '%c'\n",
                              hx_line, yytext[0]); }

%%
