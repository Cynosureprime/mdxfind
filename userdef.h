/*
 * userdef.h - user-defined hash type loader for mdxfind (Milestone 1)
 *
 * $Revision: 1.8 $
 * $Log: userdef.h,v $
 * Revision 1.8  2026/08/31 02:07:28  dlr
 * Add an optional stored-form declaration to userdef.txt, and the split and build routines that consume it. A user-defined type's hx expression says how the digest is COMPUTED and nothing about how it is STORED, and the two differ often enough to matter: KoreLogic's bwtdt stores an 8-hex salt immediately followed by its 32-hex digest in one 40-character field, so the salt never reached the salt slot and the type could not verify its own hashes however it was selected. The new form key reuses hx's concatenation notation so the two read alike, for example salt(8) . digest(32), and supports quoted literals for leading characters, trailing characters and separators, plus optional widths so a field may be variable. At most one variable-width field is allowed per form, because with two the boundary is not determined by the text; the loader refuses rather than guessing. A malformed form is a hard skip rather than a silent ignore, since the type would otherwise load and then fail to read its own hashes, which is the quiet failure this declaration exists to remove. The parsed form is printed in the load report, because a form that parsed but was never mentioned is indistinguishable from one that was dropped. userdef_form_build is the inverse of the split and exists because a form governs OUTPUT as well as input: a found line must come back in the shape it was read, or the emitting tool's output is not accepted by the consuming one. Expect the grammar to grow.
 *
 * Revision 1.7  2026/08/09 20:13:23  dlr
 * Phase A of built-in / user-defined op address separation. JOB_USERDEF_BASE 1000 to 1100 becomes a compile-time DEFAULT only; the live base is derived at runtime from the Types array length and set via the new userdef_set_base before userdef_load. Declares userdef_set_base and userdef_base. Adding a built-in type can no longer grow into the user range - the base moves with it.
 *
 * Revision 1.6  2026/06/10 15:41:40  dlr
 * v1.527: USERDEF_SLOT_* bitmask + uses_salt legacy field; struct userdef_type gains slot_mask. Forward-compat with v2 load grammar.
 *
 * Revision 1.5  2026/05/29 02:11:12  dlr
 * Add extern userdef_verbose (load-message verbosity; default 0 = silent). Leave userdef_gpu_status declaration UNGUARDED (definition stays codegen-gated): mdxfind.c is not compiled with USERDEF_HAVE_CODEGEN, so the guarded prototype was invisible there, making the call default to implicit int and truncate the returned const char pointer to 32 bits on a 64-bit target (latent crash when the static buffer loads above 4 GB).
 *
 * Revision 1.4  2026/05/28 22:08:23  dlr
 * Milestone 3 hashpipe parity: add userdef_get_by_index registry iterator (shared core, no codegen guard) so the hashpipe worker can walk loaded user types per line.
 *
 * Revision 1.3  2026/05/28 21:28:07  dlr
 * *** empty log message ***
 *
 * Revision 1.2  2026/05/28 21:08:52  dlr
 * Milestone 2 sub-phase C+D user-defined hash types: add load-time dedup advisory comparing each compiled user program against the built-in catalog via the shared programs_equal comparator (tools hx_program_cmp.h) over hx_specs_data, emitting a non-fatal nudge toward the equivalent catalog eN for its hand-tuned GPU path; add content-hash identity suggestion (in-tree FNV-1a over canonical bytecode) printed when the user id differs from the stable hash; complete reject-salted slot-inference message wording; record skipped-entry reasons so a selected-but-failed -m u id is fatal with the specific diagnostic; add userdef_gpu_status running hx_detect_pattern on the compiled program for an honest invoke-time GPU-eligibility line. No new link cost: codegen objects already on the mdxfind link line. No new command-line flag per the no-long-options constraint.
 *
 * Revision 1.1  2026/05/28 20:44:34  dlr
 * Milestone 1 user-defined hash types: public interface for the loader. JOB_USERDEF_BASE 1000 USERDEF_MAX 900 chosen below JOB_DONE 2000 and within the 16-bit Hashchain.flags match-flag ceiling. struct userdef_type plus load lookup get name count accessors.
 *
 *
 * Skilled crypto users define custom hash algorithms as hx expressions in
 * a stanza config file ($MDXFIND_CACHE/userdef.txt), with no C code and no
 * recompile, and use them like built-in catalog types: -m u<id> selects the
 * type, output is USER_<name>xNN.
 *
 * Milestone 1 (proof of ability): unsalted, CPU-only execution via the hx VM.
 * The loader parses the stanza file, compiles each hx expression to an
 * hx_program, and registers each under a synthetic op in the range
 * [JOB_USERDEF_BASE, JOB_USERDEF_BASE + USERDEF_MAX).
 */

#ifndef USERDEF_H
#define USERDEF_H

#include "hx_vm.h"

/*
 * Synthetic op base for user-defined types.  Chosen to sit ABOVE the
 * highest built-in JOB_ enum (currently 996) but BELOW JOB_DONE (2000),
 * because mdxfind reuses several [JOB_DONE]-sized per-op arrays
 * (Totalfound, TypeOpts, JudyJ, Foundcnt, Dosalt, ...) AND the loaded-hash
 * match-flag field (struct Hashchain.flags) is a 16-bit unsigned short.
 * Both ceilings forbid the originally-suggested 100000 base.
 */
/*
 * 2026-08-07: raised 1000 -> 1100. The built-in range had actually reached
 * 999 (SHA1CRYPT), leaving ZERO free slots -- the next built-in type would
 * have silently aliased user-defined op 1000. The two ranges are now
 * separated by a deliberate 100-slot gap:
 *
 *      1 .. 1099   built-in types      (Types[] index == op number)
 *   1100 .. 1999   user-defined types  (JOB_USERDEF_BASE + id)
 *
 * mdxfind checks this boundary at startup (see the Numtypes guard in main)
 * so an overrun is a loud, immediate failure rather than silent aliasing.
 *
 * NOTE (design debt): the ranges still share one op number space, which is
 * why this needed a manual bump at all. The durable fix is to give built-in
 * and user-defined types genuinely separate address spaces -- e.g. a tagged
 * op (kind + index) or a dedicated user table -- so a collision becomes
 * structurally impossible instead of a number nobody remembered to raise.
 */
#define JOB_USERDEF_BASE  1100
#define USERDEF_MAX       900   /* ids 1100..1999, leaves headroom below 2000 */

/*
 * Load-time message verbosity (defined in userdef.c).  Default 0 = SILENT:
 * the loader prints nothing on stdout/stderr during a normal run, so it never
 * corrupts a downstream consumer of hashpipe/mdxfind output.  Set to 1 to
 * print the full load report (hashpipe's -U "dump userdef and exit" mode).
 */
extern int userdef_verbose;

/*
 * Slot-usage bitmask (v1.527+).  Each bit set means the compiled program
 * references the corresponding hx VM slot, which in turn drives:
 *   (a) the load-time accept/reject decision (salt2 + pepper still rejected
 *       in v1.527; salt + user accepted),
 *   (b) TypeOpts auto-assignment in mdxfind.c so the standard Typesalt[] /
 *       Typeuser[] per-line loaders populate the right Judy arrays,
 *   (c) the dispatch arm's per-pass iteration over salt + user snapshots.
 */
#define USERDEF_SLOT_SALT    0x01
#define USERDEF_SLOT_SALT2   0x02   /* rejected: needs v2 load grammar */
#define USERDEF_SLOT_PEPPER  0x04   /* rejected: needs v2 load grammar */
#define USERDEF_SLOT_USER    0x08

/*
 * Stored-form declaration -- the optional "form =" key in userdef.txt.
 *
 * A user-defined type's hx expression says how the digest is COMPUTED; it says
 * nothing about how the result is STORED. Those differ often enough to matter:
 * KoreLogic's bwtdt stores an 8-hex salt immediately followed by its 32-hex
 * digest in one 40-character field, so the salt never reaches the salt slot and
 * the type cannot verify its own hashes however it is selected.
 *
 * The declaration reuses hx's concatenation notation so it reads the same way:
 *
 *     form = salt(8) . digest(32)                     bwtdt
 *     form = digest(40) . salt(20)                    Oracle 11g shape
 *     form = "$IPB2$" . salt(10) . "$" . digest(32)   literal prefix/separator
 *     form = salt . "$" . digest(128)                 variable-width field
 *
 * A piece is a quoted literal, or a field name optionally followed by a
 * character width in parentheses. A field with no width is variable, and at
 * most ONE variable piece is allowed per form: with two, the split is not
 * determined by the text. Expect this grammar to grow -- trailing literals,
 * more separators, and further field kinds are all deliberately expressible
 * without changing the shape of the descriptor.
 */
#define USERDEF_FORM_MAX   12   /* pieces per form */
#define USERDEF_LIT_MAX    32   /* bytes per literal piece */

enum userdef_piece_kind {
	UDF_LITERAL = 0,   /* fixed text: prefix, separator or suffix */
	UDF_DIGEST,
	UDF_SALT,
	UDF_USER
};

struct userdef_piece {
	int  kind;                     /* enum userdef_piece_kind        */
	int  len;                      /* width in characters; 0 = variable */
	char lit[USERDEF_LIT_MAX];     /* text when kind == UDF_LITERAL  */
};

struct userdef_form {
	int npieces;                   /* 0 = no form declared           */
	struct userdef_piece piece[USERDEF_FORM_MAX];
};

struct userdef_type {
	char        name[128];   /* stanza header => USER_<name>            */
	char        dispname[160]; /* "USER_<name>" precomputed for output   */
	char        idstr[128];  /* user-supplied id string (freeform key)  */
	char        hx[2048];    /* the verbatim hx expression              */
	hx_program *prog;        /* compiled program (shared, read-only)    */
	int         op;          /* synthetic op = JOB_USERDEF_BASE + seq   */
	int         diglen_hex;  /* hex-string digest length (2 * bytes)    */
	int         uses_salt;   /* legacy: nonzero iff slot_mask != 0       */
	int         slot_mask;   /* USERDEF_SLOT_* bits referenced by prog   */
	struct userdef_form form;   /* stored-form layout; npieces 0 = none */
};

/*
 * Parse $MDXFIND_CACHE/userdef.txt and register all valid types.
 * cache_env is the raw value of getenv("MDXFIND_CACHE") (may be NULL).
 * Returns the number of types successfully loaded (>=0); never fatal on
 * its own (a missing file is fine — user types are optional).
 */
int userdef_load(const char *cache_env);

/* Exact string-keyed lookup by the user-supplied id. Returns op, or -1. */
int userdef_lookup_by_id(const char *idstr);

/* Accessor by op (op >= JOB_USERDEF_BASE). Returns NULL if not a user op. */
struct userdef_type *userdef_get(int op);

/* True if op is in the user-defined range. */
int userdef_is_userop(int op);

/* Display name for a user op (e.g. "USER_Cust1"), or NULL. */
const char *userdef_name(int op);

/* Number of loaded user types. */
int userdef_count(void);

/* Phase A address separation: set the synthetic-op base BEFORE userdef_load().
 * mdxfind derives it from the built-in Types[] length so adding a built-in
 * type can never collide with the user range. Callers that never set it get
 * JOB_USERDEF_BASE. Calling after load() is a no-op (ops already assigned). */
void userdef_set_base(unsigned base);
unsigned userdef_base(void);

/*
 * Milestone 3 (hashpipe parity): registry iterator.  Returns the loaded
 * user type at sequence index idx (0 .. userdef_count()-1), or NULL if idx
 * is out of range.  The returned pointer is into the static registry
 * (read-only after load; the compiled hx_program is shared, so a caller
 * running it must drive a private hx_vm with its own arena -- see
 * hx_vm_init/hx_vm_run).  hashpipe's per-line worker walks this iterator to
 * test each loaded user type's digest length against the input hash length
 * and run the program over the candidate password, so it IDENTIFIES user
 * types (USER_<name>) in pipe mode, not just at load time.
 */
struct userdef_type *userdef_get_by_index(int idx);

/*
 * Split a stored hash field according to a declared form.
 *
 * Returns 1 when the field matches the form and fills the requested pieces
 * (any out parameter may be NULL). Returns 0 when it does not match, which is
 * the normal answer for a line of some other type -- callers must treat a 0 as
 * "not this type", never as an error.
 *
 * Pointers returned point INTO field; nothing is copied or allocated.
 */
int userdef_form_split(const struct userdef_form *f,
                       const char *field, int fieldlen,
                       const char **digest, int *digestlen,
                       const char **salt,   int *saltlen,
                       const char **user,   int *userlen);

/*
 * Assemble a stored field from its parts, the inverse of userdef_form_split().
 *
 * A declared form governs OUTPUT as well as input: a found line must come back
 * in the shape it was read, or the emitting tool's output is not accepted by
 * the consuming one. Returns the length written, or -1 if it does not fit.
 */
int userdef_form_build(const struct userdef_form *f,
                       const char *digest, int digestlen,
                       const char *salt,   int saltlen,
                       const char *user,   int userlen,
                       char *out, int outsz);

/*
 * Milestone 2, sub-phase C2 (fatal-if-selected).  A stanza that fails to
 * load (malformed, unparseable hx, salted/structured, unusable digest,
 * duplicate id) is SKIPPED with a loud per-entry warning so one typo does
 * not kill the other entries.  But if the user SELECTS such an id with
 * -m u<id>, that is fatal -- and the fatal message should carry the
 * specific reason, not a generic "not found".  The loader records each
 * skipped id and its reason here; the selector consults it.
 *
 * Returns the recorded skip reason for idstr (a short human string such
 * as "salted/structured user types are not yet supported (v2)"), or NULL
 * if idstr was never seen as a skipped entry.
 */
const char *userdef_skip_reason(const char *idstr);

/*
 * Milestone 2, sub-phase D4 (GPU eligibility status).  Run the codegen
 * shape detectors on a user op's compiled program and return an honest
 * one-line status string suitable for stderr at invoke time.  Enum-
 * agnostic: it inspects only the hx_program shape.  Returns a pointer to
 * a static/per-call buffer owned by userdef.c (do not free).  Returns
 * NULL if op is not a registered user op.
 *
 * Two outcomes (GPU dispatch for user types is phase 2, so a supported
 * shape still runs on CPU in v1):
 *   - shape NOT a codegen shape -> "GPU not available for this
 *     expression shape; running on CPU."
 *   - shape IS a codegen shape  -> "GPU-eligible shape, but GPU dispatch
 *     for user-defined types is not yet enabled; running on CPU."
 *
 * Milestone 3: this function (and the load-time dedup advisory) depend on
 * the codegen catalog hx_specs_data[] + the GPU-shape detector
 * hx_detect_pattern, which are linked into mdxfind ($(CODEGEN_OBJS)) but
 * NOT into hashpipe (which links only $(HX_OBJS) and has no GPU path).  The
 * definition is therefore compiled only when USERDEF_HAVE_CODEGEN is set
 * (mdxfind's userdef.o); it is unavailable in hashpipe's build.  The
 * DECLARATION, however, is left UNGUARDED (always visible): mdxfind.c is not
 * necessarily compiled with -DUSERDEF_HAVE_CODEGEN, and without a visible
 * prototype the call defaults to implicit int and TRUNCATES the returned
 * const char * to 32 bits on a 64-bit target -- a latent crash when the
 * static buffer loads above 4 GB.  hashpipe never calls it, so an
 * always-visible declaration is harmless there (no undefined reference).
 */
const char *userdef_gpu_status(int op);

#endif /* USERDEF_H */
