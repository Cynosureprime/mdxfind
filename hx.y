/*
 * hx.y - grammar for the hx hash expression language
 *
 * Grammar overview:
 *   program     = stmt_list
 *   stmt        = assignment | for_stmt | if_stmt | expr
 *   assignment  = IDENT '=' expr
 *   for_stmt    = for IDENT '=' expr to expr { stmt_list }
 *   if_stmt     = if condition { stmt_list } [else { stmt_list }]
 *   expr        = primary | expr '.' primary      (concat)
 *   primary     = IDENT | builtin | NUMBER | STRING | funcall | '(' expr ')'
 *   funcall     = IDENT '(' arglist ')'
 *                 IDENT '^' NUMBER '(' arglist ')' (iteration)
 *
 * Statement separators: newline (at top level) or semicolon (anywhere).
 * Inside { }, newlines are whitespace — use ; to separate statements.
 * "} else {" must be on one line (no newline between } and else).
 *
 * Built-in variables: pass, salt, salt2, pepper
 * Default encoding: hex (lowercase).  Use _bin suffix for raw bytes.
 */

/* %code requires: emitted into both .c and .tab.h */
%code requires {
#include "hx_ast.h"

typedef struct {
	hx_node **items;
	int       count;
	int       cap;
} nodelist;
}

/* %code: emitted into .c only, after all declarations */
%code {
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int  yylex(void);
extern int  hx_line;
void        yyerror(const char *s);

hx_node *hx_parse_result = NULL;

static void nl_init(nodelist *nl)
{
	nl->items = NULL;
	nl->count = 0;
	nl->cap   = 0;
}

static void nl_push(nodelist *nl, hx_node *n)
{
	if (nl->count >= nl->cap) {
		nl->cap = nl->cap ? nl->cap * 2 : 4;
		nl->items = realloc(nl->items, nl->cap * sizeof(hx_node *));
	}
	nl->items[nl->count++] = n;
}
}

/* ---- value types ---- */

%union {
	char    *str;
	int64_t  num;
	hx_node *node;
	nodelist nlist;
}

/* ---- tokens ---- */

%token <str> IDENT STRING PASS SALT SALT2 PEPPER USER
%token <num> NUMBER
%token FOR TO IF ELSE
%token EQ NE LE GE
%token SEP                  /* newline or semicolon */

/* ---- non-terminal types ---- */

%type <node>  program stmt expr primary funcall
%type <node>  assignment for_stmt if_stmt
%type <node>  condition block
%type <nlist> arglist stmt_list

/* ---- precedence (lowest to highest) ---- */

%left '.'

/* 3 shift/reduce conflicts on SEP: leading separators before stmt_list.
   Bison resolves all as shift (absorb into sep_opt), which is correct. */
%expect 3

%%

/* ---- top-level ---- */

program:
      sep_opt stmt_list sep_opt
                        { hx_parse_result = hx_block($2.items, $2.count, 1); }
    ;

stmt_list:
      /* empty */       { nl_init(&$$); }
    | stmt              { nl_init(&$$); nl_push(&$$, $1); }
    | stmt_list seps stmt
                        { $$ = $1; nl_push(&$$, $3); }
    ;

/* ---- separators ---- */

seps:
      SEP
    | seps SEP
    ;

sep_opt:
      /* empty */
    | seps
    ;

/* ---- block: { stmt_list } ---- */

block:
      '{' sep_opt stmt_list sep_opt '}'
                        { $$ = hx_block($3.items, $3.count, hx_line); }
    ;

/* ---- statements ---- */

stmt:
      assignment
    | for_stmt
    | if_stmt
    | expr              { $$ = $1; }
    ;

assignment:
      IDENT '=' expr    { $$ = hx_assign($1, $3, hx_line); }
    ;

for_stmt:
      FOR IDENT '=' expr TO expr block
                        { $$ = hx_for($2, $4, $6, $7, hx_line); }
    ;

/* } else { must be on one line (no newline between } and else) */
if_stmt:
      IF condition block
                        { $$ = hx_if($2, $3, NULL, hx_line); }
    | IF condition block ELSE block
                        { $$ = hx_if($2, $3, $5, hx_line); }
    ;

/* ---- conditions (for if statements) ---- */

condition:
      expr EQ expr      { $$ = hx_binop_node(HX_OP_EQ, $1, $3, hx_line); }
    | expr NE expr      { $$ = hx_binop_node(HX_OP_NE, $1, $3, hx_line); }
    | expr '<' expr     { $$ = hx_binop_node(HX_OP_LT, $1, $3, hx_line); }
    | expr '>' expr     { $$ = hx_binop_node(HX_OP_GT, $1, $3, hx_line); }
    | expr LE expr      { $$ = hx_binop_node(HX_OP_LE, $1, $3, hx_line); }
    | expr GE expr      { $$ = hx_binop_node(HX_OP_GE, $1, $3, hx_line); }
    ;

/* ---- expressions ---- */

expr:
      primary           { $$ = $1; }
    | expr '.' primary  { $$ = hx_concat($1, $3, hx_line); }
    ;

primary:
      IDENT             { $$ = hx_variable($1, hx_line); }
    | PASS              { $$ = hx_variable($1, hx_line); }
    | SALT              { $$ = hx_variable($1, hx_line); }
    | SALT2             { $$ = hx_variable($1, hx_line); }
    | PEPPER            { $$ = hx_variable($1, hx_line); }
    | USER              { $$ = hx_variable($1, hx_line); }
    | NUMBER            { $$ = hx_number($1, hx_line); }
    | STRING            { $$ = hx_literal($1, strlen($1), hx_line); }
    | funcall           { $$ = $1; }
    | '(' expr ')'      { $$ = $2; }
    ;

funcall:
      IDENT '(' arglist ')'
                        { $$ = hx_funcall($1, $3.items, $3.count, hx_line); }
    | IDENT '^' NUMBER '(' arglist ')'
                        { $$ = hx_iter($1, (int)$3, $5.items, $5.count, hx_line); }
    ;

arglist:
      expr              { nl_init(&$$); nl_push(&$$, $1); }
    | arglist ',' expr  { $$ = $1; nl_push(&$$, $3); }
    ;

%%

void yyerror(const char *s)
{
	fprintf(stderr, "hx:%d: %s\n", hx_line, s);
}
