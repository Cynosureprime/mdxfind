/*
 * hx_ast.c - AST node constructors and utilities for hx
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hx_ast.h"

static hx_node *node_new(hx_node_type type, int line)
{
	hx_node *n = calloc(1, sizeof(hx_node));
	if (!n) { fprintf(stderr, "hx: out of memory\n"); exit(1); }
	n->type = type;
	n->line = line;
	return n;
}

hx_node *hx_literal(const char *str, int len, int line)
{
	hx_node *n = node_new(HX_LITERAL, line);
	n->u.literal.str = strndup(str, len);
	n->u.literal.len = len;
	return n;
}

hx_node *hx_number(int64_t val, int line)
{
	hx_node *n = node_new(HX_NUMBER, line);
	n->u.number = val;
	return n;
}

hx_node *hx_variable(const char *name, int line)
{
	hx_node *n = node_new(HX_VARIABLE, line);
	n->u.varname = strdup(name);
	return n;
}

hx_node *hx_concat(hx_node *left, hx_node *right, int line)
{
	hx_node *n = node_new(HX_CONCAT, line);
	n->u.concat.left  = left;
	n->u.concat.right = right;
	return n;
}

hx_node *hx_funcall(const char *name, hx_node **args, int nargs, int line)
{
	hx_node *n = node_new(HX_FUNCALL, line);
	n->u.call.name  = strdup(name);
	n->u.call.args  = args;
	n->u.call.nargs = nargs;
	return n;
}

hx_node *hx_iter(const char *name, int count, hx_node **args, int nargs, int line)
{
	hx_node *n = node_new(HX_ITER, line);
	n->u.iter.name  = strdup(name);
	n->u.iter.count = count;
	n->u.iter.args  = args;
	n->u.iter.nargs = nargs;
	return n;
}

hx_node *hx_assign(const char *varname, hx_node *expr, int line)
{
	hx_node *n = node_new(HX_ASSIGN, line);
	n->u.assign.varname = strdup(varname);
	n->u.assign.expr    = expr;
	return n;
}

hx_node *hx_for(const char *var, hx_node *from, hx_node *to,
                hx_node *body, int line)
{
	hx_node *n = node_new(HX_FOR, line);
	n->u.forloop.varname = strdup(var);
	n->u.forloop.from    = from;
	n->u.forloop.to      = to;
	n->u.forloop.body    = body;
	return n;
}

hx_node *hx_if(hx_node *cond, hx_node *then_b, hx_node *else_b, int line)
{
	hx_node *n = node_new(HX_IF, line);
	n->u.ifstmt.cond      = cond;
	n->u.ifstmt.then_body = then_b;
	n->u.ifstmt.else_body = else_b;
	return n;
}

hx_node *hx_block(hx_node **stmts, int nstmts, int line)
{
	hx_node *n = node_new(HX_BLOCK, line);
	n->u.block.stmts  = stmts;
	n->u.block.nstmts = nstmts;
	return n;
}

hx_node *hx_binop_node(hx_op op, hx_node *left, hx_node *right, int line)
{
	hx_node *n = node_new(HX_BINOP, line);
	n->u.binop.op    = op;
	n->u.binop.left  = left;
	n->u.binop.right = right;
	return n;
}

/* ---- free ---- */

void hx_free(hx_node *node)
{
	int i;
	if (!node) return;

	switch (node->type) {
	case HX_LITERAL:
		free(node->u.literal.str);
		break;
	case HX_VARIABLE:
		free(node->u.varname);
		break;
	case HX_CONCAT:
		hx_free(node->u.concat.left);
		hx_free(node->u.concat.right);
		break;
	case HX_FUNCALL:
		free(node->u.call.name);
		for (i = 0; i < node->u.call.nargs; i++)
			hx_free(node->u.call.args[i]);
		free(node->u.call.args);
		break;
	case HX_ITER:
		free(node->u.iter.name);
		for (i = 0; i < node->u.iter.nargs; i++)
			hx_free(node->u.iter.args[i]);
		free(node->u.iter.args);
		break;
	case HX_ASSIGN:
		free(node->u.assign.varname);
		hx_free(node->u.assign.expr);
		break;
	case HX_FOR:
		free(node->u.forloop.varname);
		hx_free(node->u.forloop.from);
		hx_free(node->u.forloop.to);
		hx_free(node->u.forloop.body);
		break;
	case HX_IF:
		hx_free(node->u.ifstmt.cond);
		hx_free(node->u.ifstmt.then_body);
		hx_free(node->u.ifstmt.else_body);
		break;
	case HX_BLOCK:
		for (i = 0; i < node->u.block.nstmts; i++)
			hx_free(node->u.block.stmts[i]);
		free(node->u.block.stmts);
		break;
	case HX_BINOP:
		hx_free(node->u.binop.left);
		hx_free(node->u.binop.right);
		break;
	case HX_NUMBER:
		break;
	}
	free(node);
}

/* ---- debug dump ---- */

static const char *op_str(hx_op op)
{
	switch (op) {
	case HX_OP_EQ:  return "==";
	case HX_OP_NE:  return "!=";
	case HX_OP_LT:  return "<";
	case HX_OP_GT:  return ">";
	case HX_OP_LE:  return "<=";
	case HX_OP_GE:  return ">=";
	case HX_OP_MOD: return "%";
	}
	return "??";
}

void hx_dump(hx_node *node, int depth)
{
	int i;
	if (!node) return;

	for (i = 0; i < depth; i++) fprintf(stderr, "  ");

	switch (node->type) {
	case HX_LITERAL:
		fprintf(stderr, "LITERAL \"%s\"\n", node->u.literal.str);
		break;
	case HX_NUMBER:
		fprintf(stderr, "NUMBER %lld\n", (long long)node->u.number);
		break;
	case HX_VARIABLE:
		fprintf(stderr, "VAR %s\n", node->u.varname);
		break;
	case HX_CONCAT:
		fprintf(stderr, "CONCAT\n");
		hx_dump(node->u.concat.left,  depth + 1);
		hx_dump(node->u.concat.right, depth + 1);
		break;
	case HX_FUNCALL:
		fprintf(stderr, "CALL %s (nargs=%d)\n",
		        node->u.call.name, node->u.call.nargs);
		for (i = 0; i < node->u.call.nargs; i++)
			hx_dump(node->u.call.args[i], depth + 1);
		break;
	case HX_ITER:
		fprintf(stderr, "ITER %s^%d (nargs=%d)\n",
		        node->u.iter.name, node->u.iter.count,
		        node->u.iter.nargs);
		for (i = 0; i < node->u.iter.nargs; i++)
			hx_dump(node->u.iter.args[i], depth + 1);
		break;
	case HX_ASSIGN:
		fprintf(stderr, "ASSIGN %s\n", node->u.assign.varname);
		hx_dump(node->u.assign.expr, depth + 1);
		break;
	case HX_FOR:
		fprintf(stderr, "FOR %s\n", node->u.forloop.varname);
		for (i = 0; i < depth + 1; i++) fprintf(stderr, "  ");
		fprintf(stderr, "from:\n");
		hx_dump(node->u.forloop.from, depth + 2);
		for (i = 0; i < depth + 1; i++) fprintf(stderr, "  ");
		fprintf(stderr, "to:\n");
		hx_dump(node->u.forloop.to, depth + 2);
		for (i = 0; i < depth + 1; i++) fprintf(stderr, "  ");
		fprintf(stderr, "body:\n");
		hx_dump(node->u.forloop.body, depth + 2);
		break;
	case HX_IF:
		fprintf(stderr, "IF\n");
		for (i = 0; i < depth + 1; i++) fprintf(stderr, "  ");
		fprintf(stderr, "cond:\n");
		hx_dump(node->u.ifstmt.cond, depth + 2);
		for (i = 0; i < depth + 1; i++) fprintf(stderr, "  ");
		fprintf(stderr, "then:\n");
		hx_dump(node->u.ifstmt.then_body, depth + 2);
		if (node->u.ifstmt.else_body) {
			for (i = 0; i < depth + 1; i++) fprintf(stderr, "  ");
			fprintf(stderr, "else:\n");
			hx_dump(node->u.ifstmt.else_body, depth + 2);
		}
		break;
	case HX_BLOCK:
		fprintf(stderr, "BLOCK (%d stmts)\n", node->u.block.nstmts);
		for (i = 0; i < node->u.block.nstmts; i++)
			hx_dump(node->u.block.stmts[i], depth + 1);
		break;
	case HX_BINOP:
		fprintf(stderr, "BINOP %s\n", op_str(node->u.binop.op));
		hx_dump(node->u.binop.left,  depth + 1);
		hx_dump(node->u.binop.right, depth + 1);
		break;
	}
}
