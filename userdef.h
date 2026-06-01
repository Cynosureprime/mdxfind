/*
 * userdef.h - user-defined hash type loader for mdxfind (Milestone 1)
 *
 * $Revision: 1.5 $
 * $Log: userdef.h,v $
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
#define JOB_USERDEF_BASE  1000
#define USERDEF_MAX       900   /* ids 1000..1899, leaves headroom below 2000 */

/*
 * Load-time message verbosity (defined in userdef.c).  Default 0 = SILENT:
 * the loader prints nothing on stdout/stderr during a normal run, so it never
 * corrupts a downstream consumer of hashpipe/mdxfind output.  Set to 1 to
 * print the full load report (hashpipe's -U "dump userdef and exit" mode).
 */
extern int userdef_verbose;

struct userdef_type {
	char        name[128];   /* stanza header => USER_<name>            */
	char        dispname[160]; /* "USER_<name>" precomputed for output   */
	char        idstr[128];  /* user-supplied id string (freeform key)  */
	char        hx[2048];    /* the verbatim hx expression              */
	hx_program *prog;        /* compiled program (shared, read-only)    */
	int         op;          /* synthetic op = JOB_USERDEF_BASE + seq   */
	int         diglen_hex;  /* hex-string digest length (2 * bytes)    */
	int         uses_salt;   /* slot-inference: references salt/etc      */
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
