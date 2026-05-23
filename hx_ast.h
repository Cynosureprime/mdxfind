/*
 * hx_ast.h - AST node definitions for the hx hash expression language
 *
 * hx is a domain-specific language for describing and computing
 * cryptographic hash compositions.  Every hash function has two
 * forms: name() returns lowercase hex, name_bin() returns raw bytes.
 *
 * Expression examples:
 *   md5(pass)                       simple MD5
 *   sha1(md5(pass) . salt)          compound with salt
 *   md5^1000(pass)                  iteration shorthand
 *
 * Script example (PHPBB3):
 *   h = md5_bin(salt . pass)
 *   for i = 1 to 2047 {
 *       h = md5_bin(h . pass)
 *   }
 *   hex(h)
 */

#ifndef HX_AST_H
#define HX_AST_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* strndup portability (not available in MinGW/MSVCRT) */
#if defined(_WIN32) && !defined(strndup)
static inline char *hx_strndup(const char *s, size_t n)
{
	size_t len = strlen(s);
	if (len > n) len = n;
	char *r = (char *)malloc(len + 1);
	if (r) { memcpy(r, s, len); r[len] = '\0'; }
	return r;
}
#define strndup hx_strndup
#endif

/* ---- AST node types ---- */

typedef enum {
	HX_LITERAL,         /* string constant: "hello"                */
	HX_NUMBER,          /* integer literal: 1000                   */
	HX_VARIABLE,        /* named variable: pass, salt, h, ...      */
	HX_CONCAT,          /* expr . expr                             */
	HX_FUNCALL,         /* func(args)                              */
	HX_ITER,            /* func^N(args) - shorthand iteration      */
	HX_ASSIGN,          /* ident = expr                            */
	HX_FOR,             /* for ident = expr to expr { block }      */
	HX_IF,              /* if cond { block } [else { block }]      */
	HX_BLOCK,           /* statement list                          */
	HX_BINOP,           /* comparison for if conditions             */
} hx_node_type;

/* Comparison / arithmetic operators (for conditions) */
typedef enum {
	HX_OP_EQ,           /* == */
	HX_OP_NE,           /* != */
	HX_OP_LT,           /* <  */
	HX_OP_GT,           /* >  */
	HX_OP_LE,           /* <= */
	HX_OP_GE,           /* >= */
	HX_OP_MOD,          /* %  (for future use) */
} hx_op;

/* ---- AST node ---- */

typedef struct hx_node hx_node;

struct hx_node {
	hx_node_type type;
	int          line;          /* source line for error messages */
	union {
		/* HX_LITERAL */
		struct {
			char *str;
			int   len;
		} literal;

		/* HX_NUMBER */
		int64_t number;

		/* HX_VARIABLE */
		char *varname;

		/* HX_CONCAT */
		struct {
			hx_node *left;
			hx_node *right;
		} concat;

		/* HX_FUNCALL */
		struct {
			char     *name;     /* "md5", "sha1_bin", etc. */
			hx_node **args;
			int       nargs;
		} call;

		/* HX_ITER */
		struct {
			char     *name;     /* function name           */
			int       count;    /* iteration count         */
			hx_node **args;
			int       nargs;
		} iter;

		/* HX_ASSIGN */
		struct {
			char    *varname;
			hx_node *expr;
		} assign;

		/* HX_FOR */
		struct {
			char    *varname;   /* loop variable           */
			hx_node *from;      /* start value (integer)   */
			hx_node *to;        /* end value (integer)     */
			hx_node *body;      /* HX_BLOCK                */
		} forloop;

		/* HX_IF */
		struct {
			hx_node *cond;      /* HX_BINOP node           */
			hx_node *then_body; /* HX_BLOCK                */
			hx_node *else_body; /* HX_BLOCK or NULL        */
		} ifstmt;

		/* HX_BLOCK */
		struct {
			hx_node **stmts;
			int       nstmts;
		} block;

		/* HX_BINOP */
		struct {
			hx_op    op;
			hx_node *left;
			hx_node *right;
		} binop;
	} u;
};

/* ---- Constructor functions ---- */

hx_node *hx_literal(const char *str, int len, int line);
hx_node *hx_number(int64_t n, int line);
hx_node *hx_variable(const char *name, int line);
hx_node *hx_concat(hx_node *left, hx_node *right, int line);
hx_node *hx_funcall(const char *name, hx_node **args, int nargs, int line);
hx_node *hx_iter(const char *name, int count, hx_node **args, int nargs, int line);
hx_node *hx_assign(const char *varname, hx_node *expr, int line);
hx_node *hx_for(const char *var, hx_node *from, hx_node *to, hx_node *body, int line);
hx_node *hx_if(hx_node *cond, hx_node *then_b, hx_node *else_b, int line);
hx_node *hx_block(hx_node **stmts, int nstmts, int line);
hx_node *hx_binop_node(hx_op op, hx_node *left, hx_node *right, int line);

/* ---- Utilities ---- */

void     hx_free(hx_node *node);
void     hx_dump(hx_node *node, int depth);   /* debug: print AST */

/* ---- Parse result (set by parser) ---- */

extern hx_node *hx_parse_result;

#endif /* HX_AST_H */
