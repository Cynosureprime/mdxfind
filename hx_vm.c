/*
 * hx_vm.c - bytecode interpreter for hx
 *
 * Stack-based VM.  All string data allocated from a per-password
 * arena that resets between runs.  Integer values (loop counters)
 * are stored inline in hx_val.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hx_vm.h"

/* ---- arena ---- */

void hx_arena_init(hx_arena *a, int cap)
{
	a->buf  = malloc(cap);
	a->used = 0;
	a->cap  = cap;
	if (!a->buf) {
		fprintf(stderr, "hx: arena alloc failed (%d bytes)\n", cap);
		exit(1);
	}
}

char *hx_arena_alloc(hx_arena *a, int size)
{
	char *p;

	/* align to 8 bytes */
	size = (size + 7) & ~7;

	if (a->used + size > a->cap) {
		/* double the arena */
		while (a->used + size > a->cap)
			a->cap *= 2;
		a->buf = realloc(a->buf, a->cap);
		if (!a->buf) {
			fprintf(stderr, "hx: arena realloc failed\n");
			exit(1);
		}
	}
	p = a->buf + a->used;
	a->used += size;
	return p;
}

void hx_arena_reset(hx_arena *a)
{
	a->used = 0;
}

void hx_arena_free(hx_arena *a)
{
	free(a->buf);
	a->buf  = NULL;
	a->used = 0;
	a->cap  = 0;
}

/* ---- VM ---- */

void hx_vm_init(hx_vm *vm, hx_program *prog)
{
	memset(vm, 0, sizeof(*vm));
	vm->prog      = prog;
	hx_arena_init(&vm->arena, HX_ARENA_SIZE);
	vm->stack_cap = prog->max_stack;
	vm->stack     = calloc(vm->stack_cap, sizeof(hx_val));
	vm->vars      = calloc(prog->nvars, sizeof(hx_val));
}

void hx_vm_free(hx_vm *vm)
{
	hx_arena_free(&vm->arena);
	free(vm->stack);
	free(vm->vars);
}

/*
 * Compare two hx_val for equality.
 * Integers compare as integers; strings compare as bytes.
 */
static int val_equal(hx_val *a, hx_val *b)
{
	if (a->is_int && b->is_int)
		return a->ival == b->ival;
	if (a->is_int || b->is_int)
		return 0;       /* type mismatch */
	if (a->len != b->len)
		return 0;
	return memcmp(a->data, b->data, a->len) == 0;
}

/*
 * Compare two values, returning -1/0/+1 like memcmp.
 *
 * Semantics:
 *   int  vs int  : signed integer compare (ival - ival, clamped to {-1,0,+1})
 *   str  vs str  : memcmp on min(a.len, b.len); shorter prefix is less-than
 *   mixed        : runtime error — strict typing, no implicit coercion.
 *                  Catches user mistakes loudly rather than silently
 *                  returning a misleading ordering.
 *
 * The shorter-prefix-is-less-than tiebreak matches C strcmp,
 * Python bytes, and Perl cmp.
 */
static int val_compare(hx_val *a, hx_val *b)
{
	int n, c;

	if (a->is_int && b->is_int) {
		if (a->ival < b->ival) return -1;
		if (a->ival > b->ival) return  1;
		return 0;
	}
	if (a->is_int || b->is_int) {
		fprintf(stderr,
		    "hx: runtime error: cannot order int vs string "
		    "(no implicit type coercion)\n");
		exit(1);
	}
	n = a->len < b->len ? a->len : b->len;
	c = memcmp(a->data, b->data, n);
	if (c != 0)
		return c < 0 ? -1 : 1;
	if (a->len < b->len) return -1;
	if (a->len > b->len) return  1;
	return 0;
}

hx_val hx_vm_run(hx_vm *vm, const char *pass, int passlen,
                  const char *salt, int saltlen,
                  const char *salt2, int salt2len,
                  const char *pepper, int pepperlen,
                  const char *user, int userlen)
{
	hx_inst *code = vm->prog->code;
	hx_val  *stack = vm->stack;
	hx_val  *vars  = vm->vars;
	int      sp = 0;        /* stack pointer (next free slot) */
	int      pc = 0;        /* program counter */
	hx_val   result;

	hx_arena_reset(&vm->arena);

	/* reset all variables */
	memset(vars, 0, vm->prog->nvars * sizeof(hx_val));

	/* set built-in variables */
	vars[HX_SLOT_PASS].data   = (char *)pass;
	vars[HX_SLOT_PASS].len    = passlen;
	vars[HX_SLOT_SALT].data   = (char *)salt;
	vars[HX_SLOT_SALT].len    = saltlen;
	vars[HX_SLOT_SALT2].data  = (char *)salt2;
	vars[HX_SLOT_SALT2].len   = salt2len;
	vars[HX_SLOT_PEPPER].data = (char *)pepper;
	vars[HX_SLOT_PEPPER].len  = pepperlen;
	vars[HX_SLOT_USERID].data = (char *)user;
	vars[HX_SLOT_USERID].len  = userlen;

	memset(&result, 0, sizeof(result));

	for (;;) {
		hx_inst *ip = &code[pc++];

		/*
		 * Stack-overflow guard: the compiler's max_stack estimate
		 * adds 8 slots of headroom.  If runtime sp ever exceeds
		 * stack_cap, something has gone wrong (compiler bug, or a
		 * pathological program shape).  Bail loudly rather than
		 * write past the buffer end and corrupt the heap.
		 */
		if (sp >= vm->stack_cap) {
			fprintf(stderr,
			    "hx: VM stack overflow (sp=%d cap=%d) at pc=%d op=%d\n",
			    sp, vm->stack_cap, (int)(pc - 1), ip->op);
			exit(1);
		}

		switch (ip->op) {

		case OP_PUSH_VAR:
			stack[sp++] = vars[ip->u.slot];
			break;

		case OP_PUSH_STR:
			stack[sp].data   = vm->prog->strings[ip->u.stridx];
			stack[sp].len    = vm->prog->strlens[ip->u.stridx];
			stack[sp].is_int = 0;
			sp++;
			break;

		case OP_PUSH_INT:
			stack[sp].ival   = ip->u.ival;
			stack[sp].is_int = 1;
			stack[sp].data   = NULL;
			stack[sp].len    = 0;
			sp++;
			break;

		case OP_STORE:
			vars[ip->u.slot] = stack[--sp];
			break;

		case OP_CALL: {
			int nargs    = ip->u.call.nargs;
			uint8_t role = ip->u.call.role;
			hx_val *args = &stack[sp - nargs];
			hx_val res;

			memset(&res, 0, sizeof(res));
			ip->u.call.entry->fn(ip->u.call.entry,
			    args, nargs, &res, &vm->arena, role);
			sp -= nargs;
			stack[sp++] = res;
			break;
		}

		case OP_CONCAT: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			int newlen = a->len + b->len;
			char *buf = hx_arena_alloc(&vm->arena, newlen + 1);

			memcpy(buf, a->data, a->len);
			memcpy(buf + a->len, b->data, b->len);
			buf[newlen] = '\0';

			sp--;
			stack[sp - 1].data   = buf;
			stack[sp - 1].len    = newlen;
			stack[sp - 1].is_int = 0;
			break;
		}

		case OP_HALT:
			if (sp > 0)
				result = stack[sp - 1];
			return result;

		case OP_JUMP:
			pc = ip->u.addr;
			break;

		case OP_INC:
			vars[ip->u.slot].ival++;
			vars[ip->u.slot].is_int = 1;
			break;

		case OP_JUMP_LE: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			sp -= 2;
			if (val_compare(a, b) <= 0)
				pc = ip->u.addr;
			break;
		}

		case OP_JUMP_LT: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			sp -= 2;
			if (val_compare(a, b) < 0)
				pc = ip->u.addr;
			break;
		}

		case OP_JUMP_GT: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			sp -= 2;
			if (val_compare(a, b) > 0)
				pc = ip->u.addr;
			break;
		}

		case OP_JUMP_GE: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			sp -= 2;
			if (val_compare(a, b) >= 0)
				pc = ip->u.addr;
			break;
		}

		case OP_JUMP_EQ: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			sp -= 2;
			if (val_equal(a, b))
				pc = ip->u.addr;
			break;
		}

		case OP_JUMP_NE: {
			hx_val *b = &stack[sp - 1];
			hx_val *a = &stack[sp - 2];
			sp -= 2;
			if (!val_equal(a, b))
				pc = ip->u.addr;
			break;
		}

		case OP_DUP:
			stack[sp] = stack[sp - 1];
			sp++;
			break;

		case OP_POP:
			sp--;
			break;

		default:
			fprintf(stderr, "hx: unknown opcode %d at pc=%d\n",
			        ip->op, pc - 1);
			exit(1);
		}
	}
}
