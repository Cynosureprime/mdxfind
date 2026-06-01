/*
 * hx_compile.c - compile hx AST to bytecode
 *
 * Single recursive walk of the AST, emitting instructions into a
 * growable array.  Variable names are resolved to slot indices,
 * function names are looked up in the registry, and the four
 * output-role suffixes (`_bin`, `_hex`, `_b64`, `_mcf`) are stripped
 * and recorded as an `enum hx_role` value on the OP_CALL instruction.
 *
 * Suffix-validation: the registry entry advertises a `supported_roles`
 * bitmask; applying an unsupported suffix yields a compile-time error.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hx_ast.h"
#include "hx_vm.h"

/* ---- compiler state ---- */

typedef struct {
	hx_inst *code;
	int      ncode;
	int      code_cap;

	char   **strings;
	int     *strlens;
	int      nstrings;
	int      str_cap;

	char   **varnames;
	int      nvars;
	int      var_cap;

	int      depth;         /* current stack depth (for max_stack) */
	int      max_depth;
	int      has_emit;      /* set if emit() is called */
} compile_ctx;

/*
 * Does this AST node, when compiled as a statement, push a value onto
 * the runtime stack?  Used by HX_BLOCK compilation to know whether to
 * emit OP_POP after a statement.
 *
 * Value-producing: literals, variables, concatenations, function calls,
 * iter calls.
 *
 * Non-producing: assignments (STORE pops), control-flow (for, if), nested
 * blocks, raw binops (only valid inside conditions, never reached here).
 */
static int node_produces_value(const hx_node *n)
{
	if (!n) return 0;
	switch (n->type) {
	case HX_LITERAL:
	case HX_NUMBER:
	case HX_VARIABLE:
	case HX_CONCAT:
	case HX_FUNCALL:
	case HX_ITER:
		return 1;
	case HX_ASSIGN:
	case HX_FOR:
	case HX_IF:
	case HX_BLOCK:
	case HX_BINOP:
	default:
		return 0;
	}
}

static void emit(compile_ctx *c, hx_inst inst)
{
	if (c->ncode >= c->code_cap) {
		c->code_cap = c->code_cap ? c->code_cap * 2 : 64;
		c->code = realloc(c->code, c->code_cap * sizeof(hx_inst));
	}
	c->code[c->ncode++] = inst;
}

static int add_string(compile_ctx *c, const char *s, int len)
{
	int i;
	/* dedup */
	for (i = 0; i < c->nstrings; i++)
		if (c->strlens[i] == len && memcmp(c->strings[i], s, len) == 0)
			return i;

	if (c->nstrings >= c->str_cap) {
		c->str_cap = c->str_cap ? c->str_cap * 2 : 16;
		c->strings = realloc(c->strings, c->str_cap * sizeof(char *));
		c->strlens = realloc(c->strlens, c->str_cap * sizeof(int));
	}
	c->strings[c->nstrings] = strndup(s, len);
	c->strlens[c->nstrings] = len;
	return c->nstrings++;
}

static int resolve_var(compile_ctx *c, const char *name)
{
	int i;
	/* check built-ins */
	if (strcmp(name, "pass")   == 0) return HX_SLOT_PASS;
	if (strcmp(name, "salt")   == 0) return HX_SLOT_SALT;
	if (strcmp(name, "salt2")  == 0) return HX_SLOT_SALT2;
	if (strcmp(name, "pepper") == 0) return HX_SLOT_PEPPER;
	if (strcmp(name, "user")   == 0) return HX_SLOT_USERID;

	/* search existing user-defined vars */
	for (i = HX_SLOT_USERVARS; i < c->nvars; i++)
		if (strcmp(c->varnames[i], name) == 0)
			return i;

	/* allocate new slot */
	if (c->nvars >= c->var_cap) {
		c->var_cap = c->var_cap ? c->var_cap * 2 : 16;
		c->varnames = realloc(c->varnames, c->var_cap * sizeof(char *));
	}
	i = c->nvars++;
	c->varnames[i] = strdup(name);
	return i;
}

/* Track stack depth for max_stack estimate */
static void push(compile_ctx *c)
{
	c->depth++;
	if (c->depth > c->max_depth)
		c->max_depth = c->depth;
}

static void pop(compile_ctx *c, int n)
{
	c->depth -= n;
}

/*
 * Map a role enum to its short name (without the leading underscore).
 */
static const char *role_name(uint8_t role)
{
	switch (role) {
	case ROLE_BIN: return "bin";
	case ROLE_HEX: return "hex";
	case ROLE_B64: return "b64";
	case ROLE_MCF: return "mcf";
	default:       return "default";
	}
}

/*
 * Render the bitmask of supported roles as a human-readable list,
 * for use in compile-time error messages.  Writes into a static
 * buffer (caller owns no memory; result valid until next call).
 */
static const char *role_caps_str(uint8_t caps)
{
	static char buf[64];
	int o = 0;
	buf[0] = '\0';
	if (caps == 0) {
		strcpy(buf, "(none — bare name only)");
		return buf;
	}
	if (caps & ROLE_CAP_BIN) o += snprintf(buf + o, sizeof(buf) - o,
	                                       "%s_bin", o ? ", " : "");
	if (caps & ROLE_CAP_HEX) o += snprintf(buf + o, sizeof(buf) - o,
	                                       "%s_hex", o ? ", " : "");
	if (caps & ROLE_CAP_B64) o += snprintf(buf + o, sizeof(buf) - o,
	                                       "%s_b64", o ? ", " : "");
	if (caps & ROLE_CAP_MCF) o += snprintf(buf + o, sizeof(buf) - o,
	                                       "%s_mcf", o ? ", " : "");
	return buf;
}

/*
 * Map the canonical default form for a function to a short name
 * (e.g. "default = bare name"). Returned string is for diagnostics.
 */
static const char *default_role_str(uint8_t default_role)
{
	switch (default_role) {
	case ROLE_BIN: return "raw bytes";
	case ROLE_HEX: return "hex";
	case ROLE_B64: return "base64";
	case ROLE_MCF: return "MCF string";
	default:       return "(transform — no canonical encoding)";
	}
}

/*
 * Resolve a function name and any output-role suffix.
 *
 * Suffix grammar (longest match): `_bin`, `_hex`, `_b64`, `_mcf`.
 * Only one suffix is allowed.  If a suffix is present, it is stripped
 * before registry lookup and *role is set accordingly.  If no suffix,
 * *role = ROLE_DEFAULT (the function's `default_role` is applied at
 * dispatch time).
 *
 * Returns NULL if the base name is not in the registry.  Caller must
 * still call validate_role() to enforce supported_roles.
 */
static hx_func_entry *resolve_func(const char *name, uint8_t *role)
{
	char base[128];
	int  len = strlen(name);
	hx_func_entry *e;
	uint8_t r = ROLE_DEFAULT;
	int suffix_len = 0;

	/* longest-match suffix scan: all 4 candidates are length 4 */
	if (len > 4 && name[len - 4] == '_') {
		const char *s = name + len - 3;
		if      (memcmp(s, "bin", 3) == 0) { r = ROLE_BIN; suffix_len = 4; }
		else if (memcmp(s, "hex", 3) == 0) { r = ROLE_HEX; suffix_len = 4; }
		else if (memcmp(s, "b64", 3) == 0) { r = ROLE_B64; suffix_len = 4; }
		else if (memcmp(s, "mcf", 3) == 0) { r = ROLE_MCF; suffix_len = 4; }
	}

	*role = r;

	if (suffix_len > 0) {
		int blen = len - suffix_len;
		if (blen >= (int)sizeof(base)) blen = sizeof(base) - 1;
		memcpy(base, name, blen);
		base[blen] = '\0';
		e = hx_func_lookup(base);
		if (e) return e;
		/*
		 * Suffix recognized, but the stripped base isn't a function.
		 * Fall through to try the full name (this preserves the
		 * possibility of a function literally named e.g. `abc_bin`,
		 * which is unlikely but not forbidden).
		 */
		*role = ROLE_DEFAULT;
	}

	/* try full name */
	return hx_func_lookup(name);
}

/*
 * Verify that the requested role is compatible with the function's
 * advertised supported_roles.  Emits a diagnostic and exits on failure.
 *
 * ROLE_DEFAULT is always permitted (it dispatches to default_role).
 */
static void validate_role(hx_func_entry *fe, uint8_t role,
                          const char *fname, int line)
{
	uint8_t cap;

	if (role == ROLE_DEFAULT) return;

	cap = (uint8_t)(1u << role);
	if (fe->supported_roles & cap) return;

	fprintf(stderr, "hx:%d: function '%s' does not support the '_%s' "
	                "suffix\n", line, fe->name, role_name(role));
	fprintf(stderr, "hx:%d:   supported: %s   (canonical form: %s)\n",
	        line, role_caps_str(fe->supported_roles),
	        default_role_str(fe->default_role));
	(void)fname;
	exit(1);
}

/* ---- recursive compilation ---- */

static void compile_node(compile_ctx *c, hx_node *node);

/*
 * Compile a body block (the body of a for-loop or if-statement, or the
 * else-block of an if).  Every statement's stack value is discarded —
 * unlike the program-level HX_BLOCK which keeps its trailing value for
 * HALT.  Without this, a value-producing statement at the tail of a
 * loop body would leak one stack slot per iteration, eventually
 * overflowing the VM stack and corrupting the heap.
 *
 * If `body` is NULL or not a block, falls through to compile_node()
 * with a single-statement pop (defensive — grammar always passes a
 * block here).
 */
static void compile_body(compile_ctx *c, hx_node *body)
{
	hx_inst inst;
	int i;

	if (!body) return;

	memset(&inst, 0, sizeof(inst));

	if (body->type != HX_BLOCK) {
		/* single statement — compile and pop if it produces a value */
		compile_node(c, body);
		if (node_produces_value(body)) {
			inst.op = OP_POP;
			emit(c, inst);
			pop(c, 1);
		}
		return;
	}

	for (i = 0; i < body->u.block.nstmts; i++) {
		hx_node *stmt = body->u.block.stmts[i];
		compile_node(c, stmt);
		if (node_produces_value(stmt)) {
			inst.op = OP_POP;
			memset(&inst.u, 0, sizeof(inst.u));
			emit(c, inst);
			pop(c, 1);
		}
	}
}

static void compile_node(compile_ctx *c, hx_node *node)
{
	hx_inst inst;
	int i, slot;
	uint8_t role;
	int addr_jump, addr_else;
	int addr_loop;
	hx_func_entry *fe;

	if (!node) return;

	memset(&inst, 0, sizeof(inst));

	switch (node->type) {

	case HX_LITERAL:
		inst.op = OP_PUSH_STR;
		inst.u.stridx = add_string(c, node->u.literal.str,
		                           node->u.literal.len);
		emit(c, inst);
		push(c);
		break;

	case HX_NUMBER:
		inst.op = OP_PUSH_INT;
		inst.u.ival = node->u.number;
		emit(c, inst);
		push(c);
		break;

	case HX_VARIABLE:
		inst.op = OP_PUSH_VAR;
		inst.u.slot = resolve_var(c, node->u.varname);
		emit(c, inst);
		push(c);
		break;

	case HX_CONCAT:
		compile_node(c, node->u.concat.left);
		compile_node(c, node->u.concat.right);
		inst.op = OP_CONCAT;
		emit(c, inst);
		pop(c, 1);   /* two values → one */
		break;

	case HX_FUNCALL:
		/* push args left to right */
		for (i = 0; i < node->u.call.nargs; i++)
			compile_node(c, node->u.call.args[i]);

		fe = resolve_func(node->u.call.name, &role);
		if (!fe) {
			fprintf(stderr, "hx:%d: unknown function '%s'\n",
			        node->line, node->u.call.name);
			exit(1);
		}
		validate_role(fe, role, node->u.call.name, node->line);

		/* detect emit() calls */
		if (strcmp(node->u.call.name, "emit") == 0)
			c->has_emit = 1;

		inst.op = OP_CALL;
		inst.u.call.entry = fe;
		inst.u.call.nargs = node->u.call.nargs;
		inst.u.call.role  = role;
		emit(c, inst);
		pop(c, node->u.call.nargs);
		push(c);      /* result */
		break;

	case HX_ITER:
		/* func^N(args) — push args, emit CALL in a loop */
		for (i = 0; i < node->u.iter.nargs; i++)
			compile_node(c, node->u.iter.args[i]);

		fe = resolve_func(node->u.iter.name, &role);
		if (!fe) {
			fprintf(stderr, "hx:%d: unknown function '%s'\n",
			        node->line, node->u.iter.name);
			exit(1);
		}
		validate_role(fe, role, node->u.iter.name, node->line);

		/*
		 * For iter, we need a hidden loop counter.
		 * Allocate a temp variable slot for it.
		 */
		{
			char tmpname[32];
			int iter_slot, limit;

			snprintf(tmpname, sizeof(tmpname), "__iter_%d",
			         c->ncode);
			iter_slot = resolve_var(c, tmpname);
			limit = node->u.iter.count;

			/* First call: args already on stack */
			inst.op = OP_CALL;
			inst.u.call.entry = fe;
			inst.u.call.nargs = node->u.iter.nargs;
			inst.u.call.role  = role;
			emit(c, inst);
			pop(c, node->u.iter.nargs);
			push(c);

			if (limit > 1) {
				/* i = 1 (counts additional calls after the first) */
				inst.op = OP_PUSH_INT;
				inst.u.ival = 1;
				emit(c, inst);
				inst.op = OP_STORE;
				inst.u.slot = iter_slot;
				emit(c, inst);

				/* LOOP: result is on stack top */
				addr_loop = c->ncode;

				/* call fn(stack_top) — 1 arg (the previous result) */
				inst.op = OP_CALL;
				inst.u.call.entry = fe;
				inst.u.call.nargs = 1;
				inst.u.call.role  = role;
				emit(c, inst);
				/* stack: still has one value (old consumed, new pushed) */

				/* i++ */
				inst.op = OP_INC;
				inst.u.slot = iter_slot;
				emit(c, inst);

				/* push i, push limit-1 (we already did the first call) */
				inst.op = OP_PUSH_VAR;
				inst.u.slot = iter_slot;
				emit(c, inst);
				push(c);

				inst.op = OP_PUSH_INT;
				inst.u.ival = limit - 1;
				emit(c, inst);
				push(c);

				/* jump if i <= (limit-1) → LOOP */
				inst.op = OP_JUMP_LE;
				inst.u.addr = addr_loop;
				emit(c, inst);
				pop(c, 2);
			}
		}
		break;

	case HX_ASSIGN:
		compile_node(c, node->u.assign.expr);
		slot = resolve_var(c, node->u.assign.varname);
		/* DUP so the value remains as the statement's value */
		inst.op = OP_STORE;
		inst.u.slot = slot;
		emit(c, inst);
		pop(c, 1);
		break;

	case HX_FOR:
		/*
		 * for var = from to limit { body }
		 *
		 * The body is a block; its statements' values are discarded
		 * (the for-loop has no result).  compile_block_body() emits
		 * OP_POP after each value-producing statement so the runtime
		 * stack does not grow unboundedly across loop iterations.
		 *
		 * BUG HISTORY: prior to this fix, body statements that produced
		 * values (e.g., a trailing `emit(...)`) leaked one slot per
		 * iteration, causing the VM stack to overflow into adjacent
		 * heap memory after enough iterations.  The corruption typically
		 * surfaced as SIGFPE/SIGSEGV in OpenSSL's atexit cleanup
		 * (`ossl_ht_free`), since libcrypto's heap state was clobbered.
		 */
		slot = resolve_var(c, node->u.forloop.varname);

		/* var = from */
		compile_node(c, node->u.forloop.from);
		inst.op = OP_STORE;
		inst.u.slot = slot;
		emit(c, inst);
		pop(c, 1);

		/* LOOP: */
		addr_loop = c->ncode;

		/* body — values discarded */
		compile_body(c, node->u.forloop.body);

		/* var++ */
		inst.op = OP_INC;
		inst.u.slot = slot;
		emit(c, inst);

		/* push var, push limit */
		inst.op = OP_PUSH_VAR;
		inst.u.slot = slot;
		emit(c, inst);
		push(c);

		compile_node(c, node->u.forloop.to);

		/* if var <= limit, goto LOOP */
		inst.op = OP_JUMP_LE;
		inst.u.addr = addr_loop;
		emit(c, inst);
		pop(c, 2);
		break;

	case HX_IF:
		/* condition puts two values on stack */
		{
			hx_node *cond = node->u.ifstmt.cond;
			uint8_t jump_op;

			compile_node(c, cond->u.binop.left);
			compile_node(c, cond->u.binop.right);

			/* invert the condition for "jump over then-body" */
			switch (cond->u.binop.op) {
			case HX_OP_EQ: jump_op = OP_JUMP_NE; break;
			case HX_OP_NE: jump_op = OP_JUMP_EQ; break;
			case HX_OP_LT: jump_op = OP_JUMP_GE; break;
			case HX_OP_GT: jump_op = OP_JUMP_LE; break;
			case HX_OP_LE: jump_op = OP_JUMP_GT; break;
			case HX_OP_GE: jump_op = OP_JUMP_LT; break;
			default:
				fprintf(stderr, "hx:%d: unsupported condition op\n",
				        node->line);
				exit(1);
			}

			/* placeholder jump — addr filled in after then-body */
			inst.op = jump_op;
			inst.u.addr = 0;
			addr_jump = c->ncode;
			emit(c, inst);
			pop(c, 2);

			/* then body — values discarded */
			compile_body(c, node->u.ifstmt.then_body);

			if (node->u.ifstmt.else_body) {
				/* jump over else */
				inst.op = OP_JUMP;
				inst.u.addr = 0;
				addr_else = c->ncode;
				emit(c, inst);

				/* patch the conditional jump to here */
				c->code[addr_jump].u.addr = c->ncode;

				/* else body — values discarded */
				compile_body(c, node->u.ifstmt.else_body);

				/* patch jump-over-else to here */
				c->code[addr_else].u.addr = c->ncode;
			} else {
				/* patch the conditional jump to here */
				c->code[addr_jump].u.addr = c->ncode;
			}
		}
		break;

	case HX_BLOCK:
		/*
		 * Top-level program block.  Compile each statement; pop the
		 * stack value of every statement except the last (whose value
		 * becomes the program's HALT result, in the !has_emit case).
		 *
		 * Nested blocks inside HX_FOR / HX_IF go through compile_body()
		 * instead, which discards every statement's value.
		 */
		for (i = 0; i < node->u.block.nstmts; i++) {
			hx_node *stmt = node->u.block.stmts[i];
			compile_node(c, stmt);
			if (i + 1 < node->u.block.nstmts &&
			    node_produces_value(stmt)) {
				inst.op = OP_POP;
				memset(&inst.u, 0, sizeof(inst.u));
				emit(c, inst);
				pop(c, 1);
			}
		}
		break;

	case HX_BINOP:
		/* should not appear standalone — handled by HX_IF */
		fprintf(stderr, "hx:%d: unexpected binop outside condition\n",
		        node->line);
		exit(1);
	}
}

/* ---- public interface ---- */

hx_program *hx_compile(hx_node *ast)
{
	compile_ctx ctx;
	hx_program *prog;

	memset(&ctx, 0, sizeof(ctx));

	/* pre-allocate built-in variable slots */
	ctx.var_cap  = 16;
	ctx.varnames = calloc(ctx.var_cap, sizeof(char *));
	ctx.varnames[HX_SLOT_PASS]   = strdup("pass");
	ctx.varnames[HX_SLOT_SALT]   = strdup("salt");
	ctx.varnames[HX_SLOT_SALT2]  = strdup("salt2");
	ctx.varnames[HX_SLOT_PEPPER] = strdup("pepper");
	ctx.varnames[HX_SLOT_USERID] = strdup("user");
	ctx.nvars = HX_SLOT_USERVARS;

	/* compile */
	compile_node(&ctx, ast);

	/* emit HALT */
	{
		hx_inst halt;
		memset(&halt, 0, sizeof(halt));
		halt.op = OP_HALT;
		emit(&ctx, halt);
	}

	/* package into program */
	prog = calloc(1, sizeof(hx_program));
	prog->code      = ctx.code;
	prog->ncode     = ctx.ncode;
	prog->strings   = ctx.strings;
	prog->strlens   = ctx.strlens;
	prog->nstrings  = ctx.nstrings;
	prog->varnames  = ctx.varnames;
	prog->nvars     = ctx.nvars;
	prog->max_stack = ctx.max_depth + 8;  /* some headroom */
	prog->has_emit  = ctx.has_emit;

	return prog;
}

void hx_program_free(hx_program *prog)
{
	int i;
	if (!prog) return;
	free(prog->code);
	for (i = 0; i < prog->nstrings; i++)
		free(prog->strings[i]);
	free(prog->strings);
	free(prog->strlens);
	for (i = 0; i < prog->nvars; i++)
		free(prog->varnames[i]);
	free(prog->varnames);
	free(prog);
}

void hx_program_dump(hx_program *prog)
{
	int i;
	static const char *opnames[] = {
		"PUSH_VAR", "PUSH_STR", "PUSH_INT", "STORE",
		"CALL", "CONCAT", "HALT", "JUMP",
		"INC", "JUMP_LE", "JUMP_EQ", "JUMP_NE",
		"DUP", "POP"
	};

	fprintf(stderr, "--- hx bytecode: %d instructions, %d vars, "
	        "%d strings ---\n", prog->ncode, prog->nvars, prog->nstrings);

	for (i = 0; i < prog->nvars; i++)
		fprintf(stderr, "  var[%d] = %s\n", i, prog->varnames[i]);

	for (i = 0; i < prog->nstrings; i++)
		fprintf(stderr, "  str[%d] = \"%s\" (len=%d)\n",
		        i, prog->strings[i], prog->strlens[i]);

	for (i = 0; i < prog->ncode; i++) {
		hx_inst *ip = &prog->code[i];
		fprintf(stderr, "  %3d  %-10s", i,
		        ip->op < 14 ? opnames[ip->op] : "???");
		switch (ip->op) {
		case OP_PUSH_VAR:
		case OP_STORE:
		case OP_INC:
			fprintf(stderr, " %d (%s)", ip->u.slot,
			        ip->u.slot < prog->nvars
			        ? prog->varnames[ip->u.slot] : "?");
			break;
		case OP_PUSH_STR:
			fprintf(stderr, " %d (\"%s\")", ip->u.stridx,
			        prog->strings[ip->u.stridx]);
			break;
		case OP_PUSH_INT:
			fprintf(stderr, " %lld", (long long)ip->u.ival);
			break;
		case OP_CALL: {
			static const char *role_short[] = {
				"", " [bin]", " [hex]", " [b64]", " [mcf]"
			};
			uint8_t r = ip->u.call.role;
			fprintf(stderr, " %s nargs=%d%s",
			        ip->u.call.entry->name,
			        ip->u.call.nargs,
			        r < 5 ? role_short[r] : " [?role]");
			break;
		}
		case OP_JUMP:
		case OP_JUMP_LE:
		case OP_JUMP_EQ:
		case OP_JUMP_NE:
			fprintf(stderr, " → %d", ip->u.addr);
			break;
		default:
			break;
		}
		fprintf(stderr, "\n");
	}
}
