/*
 * hx_vm.h - bytecode VM types for the hx hash expression language
 *
 * Execution model: AST is compiled to a flat instruction stream once,
 * then the VM runs it per-password.  Function pointers and variable
 * slot indices are resolved at compile time.
 *
 * Values are byte strings (hash outputs, concatenations, literals)
 * or integers (loop counters).  All string data is bump-allocated
 * from a per-password arena that resets between runs.
 */

#ifndef HX_VM_H
#define HX_VM_H

#include <stdint.h>
#include "hx_ast.h"

/* ---- arena: bump allocator, reset per password ---- */

#define HX_ARENA_SIZE  (256 * 1024)   /* 256 KB default */

typedef struct {
	char *buf;
	int   used;
	int   cap;
} hx_arena;

void  hx_arena_init(hx_arena *a, int cap);
char *hx_arena_alloc(hx_arena *a, int size);
void  hx_arena_reset(hx_arena *a);
void  hx_arena_free(hx_arena *a);

/* ---- values ---- */

typedef struct {
	char   *data;       /* points into arena, string table, or input */
	int     len;
	int64_t ival;       /* valid when is_int=1 */
	uint8_t is_int;
} hx_val;

/*
 * ---- output-role enum ----
 *
 * Each call site selects one of five output roles via the function-name
 * suffix (`_bin`, `_hex`, `_b64`, `_mcf`).  The bare name (no suffix)
 * resolves to the function's `default_role` — typically ROLE_HEX for
 * digests/HMAC/KDF, ROLE_MCF for crypt-family built-ins (bcrypt, yescrypt).
 *
 * Per spec hx.1 §2.4 rev 1.5:
 *   ROLE_BIN — raw key material (bytes the algorithm derives)
 *   ROLE_HEX — lowercase hex of ROLE_BIN
 *   ROLE_B64 — RFC 4648 base64 (with `=` padding) of ROLE_BIN
 *   ROLE_MCF — full Modular Crypt Format string (crypt families only)
 *
 * Functions advertise which roles they support via `supported_roles`
 * (bitmask of ROLE_CAP_*).  Applying an unsupported suffix is a
 * compile-time error.  ROLE_DEFAULT is always allowed and dispatches
 * to the function's default_role internally.
 */
enum hx_role {
	ROLE_DEFAULT = 0,   /* no suffix — use entry->default_role */
	ROLE_BIN     = 1,
	ROLE_HEX     = 2,
	ROLE_B64     = 3,
	ROLE_MCF     = 4,
};

#define ROLE_CAP_BIN  (1u << ROLE_BIN)
#define ROLE_CAP_HEX  (1u << ROLE_HEX)
#define ROLE_CAP_B64  (1u << ROLE_B64)
#define ROLE_CAP_MCF  (1u << ROLE_MCF)

/* ---- hash function registry ---- */

typedef struct hx_func_entry hx_func_entry;

/*
 * Hash function signature.
 * self:        pointer to this entry (for bridge functions to access ctx)
 * args/nargs:  input arguments (popped from stack)
 * result:      output value (caller provides; data from arena)
 * arena:       arena for allocating result data
 * role:        one of enum hx_role.  ROLE_DEFAULT means "use the
 *              function's default form"; ROLE_BIN/HEX/B64/MCF are the
 *              explicit forms requested via the `_bin/_hex/_b64/_mcf`
 *              suffix.  Each fn maps ROLE_DEFAULT to its canonical role
 *              (for digests/HMAC/KDF that's ROLE_HEX; for bcrypt/yescrypt
 *              it's ROLE_MCF).
 */
typedef void (*hx_hashfn)(hx_func_entry *self, hx_val *args, int nargs,
                          hx_val *result, hx_arena *arena, uint8_t role);

/*
 * Bridge function type: hashpipe's compute_* signature.
 * (pass, passlen, salt, saltlen, dest)
 */
typedef void (*hx_bridge_fn)(const unsigned char *pass, int passlen,
                             const unsigned char *salt, int saltlen,
                             unsigned char *dest);

struct hx_func_entry {
	const char  *name;       /* canonical name, e.g. "md5" */
	hx_hashfn    fn;
	int           max_out;   /* max output bytes (binary form) */
	/* bridge fields — used by hx_bridge_hash() for hashpipe integration */
	hx_bridge_fn  bridge;    /* hashpipe compute_* function, or NULL */
	int           bridge_bytes; /* digest size in bytes */
	/* role-capability metadata (see enum hx_role) */
	uint8_t       supported_roles; /* bitmask: ROLE_CAP_BIN|HEX|B64|MCF */
	uint8_t       default_role;    /* ROLE_HEX, ROLE_MCF, or ROLE_DEFAULT */
};

/* Dynamic registry */
#define HX_MAX_FUNCS 1200
extern hx_func_entry hx_func_table[];
extern int           hx_func_count;

void hx_func_register(const char *name, hx_hashfn fn, int max_out,
                       hx_bridge_fn bridge, int bridge_bytes,
                       uint8_t supported_roles, uint8_t default_role);

/* Bridge wrapper: hashpipe compute_* → hx function */
void hx_bridge_hash(hx_func_entry *self, hx_val *args, int nargs,
                    hx_val *result, hx_arena *arena, uint8_t role);

hx_func_entry *hx_func_lookup(const char *name);

/* ---- opcodes ---- */

enum {
	OP_PUSH_VAR,    /* operand: slot index           → push var[slot]     */
	OP_PUSH_STR,    /* operand: string table index    → push constant     */
	OP_PUSH_INT,    /* operand: int64_t               → push integer      */
	OP_STORE,       /* operand: slot index     top →  store in var[slot]  */
	OP_CALL,        /* operand: func_entry*, nargs, binary                */
	OP_CONCAT,      /*                     a, b →  a.b                   */
	OP_HALT,        /* result = stack top                                 */
	OP_JUMP,        /* operand: addr                                      */
	OP_INC,         /* operand: slot index     var[slot]++               */
	OP_JUMP_LE,     /* operand: addr     a, b → jump if a <= b          */
	OP_JUMP_LT,     /* operand: addr     a, b → jump if a <  b          */
	OP_JUMP_GT,     /* operand: addr     a, b → jump if a >  b          */
	OP_JUMP_GE,     /* operand: addr     a, b → jump if a >= b          */
	OP_JUMP_EQ,     /* operand: addr     a, b → jump if a == b (string) */
	OP_JUMP_NE,     /* operand: addr     a, b → jump if a != b (string) */
	OP_DUP,         /* duplicate stack top                                */
	OP_POP,         /* discard stack top                                  */
};

/* ---- instruction ---- */

typedef struct {
	uint8_t op;
	union {
		int          slot;      /* PUSH_VAR, STORE, INC */
		int          stridx;    /* PUSH_STR */
		int64_t      ival;      /* PUSH_INT */
		int          addr;      /* JUMP, JUMP_LE/LT/GT/GE, JUMP_EQ, JUMP_NE */
		struct {
			hx_func_entry *entry;
			int            nargs;
			uint8_t        role;   /* enum hx_role */
		} call;
	} u;
} hx_inst;

/* ---- compiled program ---- */

typedef struct {
	hx_inst *code;
	int      ncode;

	/* string constant table */
	char   **strings;
	int     *strlens;
	int      nstrings;

	/* variable slots: 0=pass, 1=salt, 2=salt2, 3=pepper, 4+=user */
	char   **varnames;
	int      nvars;

	int      max_stack;     /* estimated max stack depth */
	int      has_emit;      /* 1 if program contains emit() calls */
} hx_program;

/* ---- built-in variable slots ---- */

#define HX_SLOT_PASS    0
#define HX_SLOT_SALT    1
#define HX_SLOT_SALT2   2
#define HX_SLOT_PEPPER  3
#define HX_SLOT_USERID  4
#define HX_SLOT_USERVARS 5  /* first user-defined variable slot */

/* ---- compiler (AST → bytecode) ---- */

hx_program *hx_compile(hx_node *ast);
void        hx_program_free(hx_program *prog);
void        hx_program_dump(hx_program *prog);  /* debug */

/* ---- VM ---- */

/*
 * Run the compiled program with the given inputs.
 * Returns the result string (allocated from arena).
 * The caller must use/copy the result before the next hx_vm_run
 * call, as the arena is reset.
 */
typedef struct {
	hx_program *prog;
	hx_arena    arena;
	hx_val     *stack;
	int         stack_cap;
	hx_val     *vars;       /* variable slots */
} hx_vm;

void    hx_vm_init(hx_vm *vm, hx_program *prog);
hx_val  hx_vm_run(hx_vm *vm, const char *pass, int passlen,
                   const char *salt, int saltlen,
                   const char *salt2, int salt2len,
                   const char *pepper, int pepperlen,
                   const char *user, int userlen);
void    hx_vm_free(hx_vm *vm);

/* ---- high-level entry point (from hx.c) ---- */

hx_program *hx_compile_expr(const char *expr, const char *script_file);

#endif /* HX_VM_H */
