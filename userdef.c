/*
 * userdef.c - user-defined hash type loader for mdxfind (Milestone 1)
 *
 * $Revision: 1.6 $
 * $Log: userdef.c,v $
 * Revision 1.6  2026/05/29 02:13:28  dlr
 * Comment fix: the load-report opt-in is the -Y validate-and-exit option (in hashpipe and mdxfind), not the briefly-used -U.
 *
 * Revision 1.5  2026/05/29 02:11:12  dlr
 * User-defined hash types silent-by-default loader: route all load-time messages through a new UD_MSG macro gated on userdef_verbose (default 0). The loader now emits nothing on stdout or stderr during a normal run, so it never corrupts a downstream consumer of hashpipe/mdxfind output. The report is turned on only by the -Y validate-and-exit option.
 *
 * Revision 1.4  2026/05/28 22:08:29  dlr
 * Milestone 3 hashpipe parity: add userdef_get_by_index registry iterator returning the loaded user type at a sequence index, for the hashpipe per-line worker to walk all loaded user types. Shared core (no codegen guard) so it links into both mdxfind and hashpipe.
 *
 * Revision 1.3  2026/05/28 21:27:42  dlr
 * *** empty log message ***
 *
 * Revision 1.2  2026/05/28 21:08:52  dlr
 * Milestone 2 sub-phase C+D user-defined hash types: add load-time dedup advisory comparing each compiled user program against the built-in catalog via the shared programs_equal comparator (tools hx_program_cmp.h) over hx_specs_data, emitting a non-fatal nudge toward the equivalent catalog eN for its hand-tuned GPU path; add content-hash identity suggestion (in-tree FNV-1a over canonical bytecode) printed when the user id differs from the stable hash; complete reject-salted slot-inference message wording; record skipped-entry reasons so a selected-but-failed -m u id is fatal with the specific diagnostic; add userdef_gpu_status running hx_detect_pattern on the compiled program for an honest invoke-time GPU-eligibility line. No new link cost: codegen objects already on the mdxfind link line. No new command-line flag per the no-long-options constraint.
 *
 * Revision 1.1  2026/05/28 20:44:34  dlr
 * Milestone 1 user-defined hash types loader for mdxfind: parse INI stanza config from MDXFIND_CACHE userdef.txt (header name, id, hx verbatim to EOL), compile each hx expression to an hx_program, register under synthetic op JOB_USERDEF_BASE plus sequence. Exact-string id lookup, display name USER_name, minimal slot-inference reject-salted guard for the unsalted M1 proof. Built with HX_STANDALONE so the hx registry has OpenSSL md5 sha1 etc directly.
 *
 *
 * Loads custom hash-algorithm definitions from a stanza config file
 * ($MDXFIND_CACHE/userdef.txt), compiles each hx expression to an
 * hx_program via the existing hx compiler, and registers each under a
 * synthetic op in the range [JOB_USERDEF_BASE, JOB_USERDEF_BASE+USERDEF_MAX).
 *
 * Stanza format (INI-style, forward-compatible with the v2 load grammar):
 *
 *   [Cust1]                       # header => USER_Cust1
 *   id = 47                       # freeform id; invoked as -m u47
 *   hx = sha1(md5(pass)."register")   # expression, VERBATIM to EOL
 *
 *   # '#' comments and blank lines are ignored.
 *
 * The hx = value is read VERBATIM to end-of-line: hx expressions contain
 * . ( ) " ' : ^ etc., so the stanza reader must NOT tokenize it -- the
 * whole remainder of the line is handed to the hx compiler, which owns its
 * own quoting.
 *
 * Milestone 1 is unsalted/CPU-only.  This module is built for mdxfind with
 * -DHX_STANDALONE so the hx function registry includes the OpenSSL-backed
 * md5/sha1/sha256/... primitives directly (mdxfind, unlike hashpipe, does
 * not register hash functions dynamically from a Hashtypes[] table).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "userdef.h"
#include "hx_vm.h"

/*
 * Load-time message verbosity.  Default 0 = SILENT: the loader emits NOTHING
 * on stdout or stderr during a normal run, so it never corrupts the output of
 * a downstream consumer of hashpipe (or mdxfind).  Set to 1 (e.g. by the
 * -Y "validate userdef.txt and exit" option in hashpipe or mdxfind) to print
 * the full load report
 * (each type loaded, content-hash suggestions, dedup advisories, skipped
 * entries with reasons, and the count).  Declared extern in userdef.h.
 * NOTE: a selected-but-failed type (mdxfind -m u<id>) is still fatal with its
 * reason via userdef_skip_reason() in the caller -- that is an error, not
 * load chatter, and is NOT gated by this flag.
 */
int userdef_verbose = 0;
#define UD_MSG(...) do { if (userdef_verbose) fprintf(stderr, __VA_ARGS__); } while (0)

/*
 * USERDEF_HAVE_CODEGEN gates the two Milestone 2 "nicety" features that
 * depend on the codegen catalog + GPU-shape detector: the load-time dedup
 * advisory (dedup_advisory) and the invoke-time GPU-eligibility status
 * (userdef_gpu_status).  mdxfind links $(CODEGEN_OBJS) so its userdef.o is
 * compiled WITH -DUSERDEF_HAVE_CODEGEN; hashpipe links only $(HX_OBJS)
 * (NOT the codegen catalog -- and has no GPU path, so GPU-eligibility is
 * meaningless there and the dedup nudge is unneeded), so its userdef.o is
 * compiled WITHOUT the macro.  The CORE of this module -- parse, compile-
 * via-hx, register, reject-salted, skip-registry, content-hash -- needs
 * only $(HX_OBJS) and compiles into BOTH binaries.
 *
 * When the macro IS set, Milestone 2 reuses three already-linked-into-
 * mdxfind facilities (no new link cost -- $(CODEGEN_OBJS) is on the mdxfind
 * link line, providing the catalog table hx_specs_data[] and the pattern
 * detector hx_detect_pattern):
 *
 *   tools/hx_program_cmp.h   -- the shared programs_equal() comparator the
 *                               standalone tools/hx_dedup_check uses (Tier
 *                               1 bytecode + role-canon + Tier 2 commutative
 *                               operand canonicalization).  Header-only
 *                               static inline -> zero new .o.
 *   codegen/hx_spec_entry.h  -- the hx_specs_data[]/hx_specs_count catalog
 *                               of compiled built-in algorithms (already an
 *                               .o on the mdxfind link line).
 *   codegen/hx_patterns.h    -- hx_detect_pattern() GPU-shape detector
 *                               (already an .o on the mdxfind link line).
 *
 * All three are quote-includes that resolve their own "../hx_vm.h" relative
 * to their directory, so no -I change is needed in the userdef.o rule.
 */
#ifdef USERDEF_HAVE_CODEGEN
#include "tools/hx_program_cmp.h"
#include "codegen/hx_spec_entry.h"
#include "codegen/hx_patterns.h"
#endif

static struct userdef_type Userdefs[USERDEF_MAX];
static int Userdef_count = 0;

/*
 * Milestone 2, sub-phase C2: skipped-entry registry for fatal-if-selected.
 * A bounded ring is plenty -- the file is human-authored and the cap is
 * advisory.  Each entry records the id string and a short reason.
 */
#define USERDEF_SKIP_MAX 256
static struct {
	char idstr[128];
	char reason[160];
} Userdef_skips[USERDEF_SKIP_MAX];
static int Userdef_skip_count = 0;

static void record_skip(const char *idstr, const char *reason)
{
	int i;
	if (!idstr || !*idstr) return;          /* no id -> not selectable */
	/* dedup: keep the FIRST reason recorded for an id */
	for (i = 0; i < Userdef_skip_count; i++)
		if (strcmp(Userdef_skips[i].idstr, idstr) == 0)
			return;
	if (Userdef_skip_count >= USERDEF_SKIP_MAX) return;
	strncpy(Userdef_skips[Userdef_skip_count].idstr, idstr,
	        sizeof(Userdef_skips[0].idstr) - 1);
	Userdef_skips[Userdef_skip_count].idstr[sizeof(Userdef_skips[0].idstr) - 1] = '\0';
	strncpy(Userdef_skips[Userdef_skip_count].reason, reason,
	        sizeof(Userdef_skips[0].reason) - 1);
	Userdef_skips[Userdef_skip_count].reason[sizeof(Userdef_skips[0].reason) - 1] = '\0';
	Userdef_skip_count++;
}

const char *userdef_skip_reason(const char *idstr)
{
	int i;
	if (!idstr) return NULL;
	for (i = 0; i < Userdef_skip_count; i++)
		if (strcmp(Userdef_skips[i].idstr, idstr) == 0)
			return Userdef_skips[i].reason;
	return NULL;
}

int userdef_count(void) { return Userdef_count; }

struct userdef_type *userdef_get_by_index(int idx)
{
	if (idx < 0 || idx >= Userdef_count) return NULL;
	return &Userdefs[idx];
}

int userdef_is_userop(int op)
{
	return op >= JOB_USERDEF_BASE && op < JOB_USERDEF_BASE + USERDEF_MAX;
}

struct userdef_type *userdef_get(int op)
{
	int i;
	if (!userdef_is_userop(op)) return NULL;
	for (i = 0; i < Userdef_count; i++)
		if (Userdefs[i].op == op)
			return &Userdefs[i];
	return NULL;
}

const char *userdef_name(int op)
{
	struct userdef_type *u = userdef_get(op);
	return u ? u->dispname : NULL;
}

int userdef_lookup_by_id(const char *idstr)
{
	int i;
	if (!idstr) return -1;
	for (i = 0; i < Userdef_count; i++)
		if (strcmp(Userdefs[i].idstr, idstr) == 0)
			return Userdefs[i].op;
	return -1;
}

/*
 * Derive the directory that holds userdef.txt from MDXFIND_CACHE.
 * MDXFIND_CACHE is conventionally a file path (the sqlite db); the GPU
 * kernel cache uses its DIRECTORY COMPONENT.  We mirror that: take the
 * directory part of MDXFIND_CACHE and join "userdef.txt".  If there is no
 * directory component, look in the current directory.
 *
 * Returns 0 and fills path[] on success; -1 if MDXFIND_CACHE is unset.
 */
static int derive_userdef_path(const char *cache_env, char *path, size_t pathlen)
{
	size_t dlen;
	const char *p;

	if (!cache_env || !*cache_env) return -1;

	/* find last '/' (or '\\' for portability) */
	dlen = 0;
	for (p = cache_env; *p; p++)
		if (*p == '/' || *p == '\\')
			dlen = (size_t)(p - cache_env) + 1; /* include the sep */

	if (dlen == 0) {
		/* no directory component -- current directory */
		if (snprintf(path, pathlen, "userdef.txt") >= (int)pathlen)
			return -1;
	} else {
		if (snprintf(path, pathlen, "%.*suserdef.txt",
		             (int)dlen, cache_env) >= (int)pathlen)
			return -1;
	}
	return 0;
}

/* trim leading/trailing ASCII whitespace, in place; returns start ptr */
static char *trim(char *s)
{
	char *end;
	while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
	if (!*s) return s;
	end = s + strlen(s) - 1;
	while (end > s && (*end == ' ' || *end == '\t' ||
	                   *end == '\r' || *end == '\n'))
		*end-- = '\0';
	return s;
}

/* case-insensitive key match for the part before '=' (already trimmed) */
static int keyis(const char *k, const char *want)
{
	while (*k && *want) {
		if (tolower((unsigned char)*k) != tolower((unsigned char)*want))
			return 0;
		k++; want++;
	}
	return *k == '\0' && *want == '\0';
}

/*
 * Slot-inference (minimal, for M1): does the compiled program reference any
 * salt/salt2/pepper/user slot?  M1 is unsalted-only; a salted user type is
 * rejected with a clear message (full v2 support is later).  This also
 * exercises the inference plumbing per the spec.
 */
static int program_uses_salt(hx_program *prog)
{
	int i;
	if (!prog || !prog->code) return 0;
	for (i = 0; i < prog->ncode; i++) {
		if (prog->code[i].op == OP_PUSH_VAR) {
			int slot = prog->code[i].u.slot;
			if (slot == HX_SLOT_SALT  || slot == HX_SLOT_SALT2 ||
			    slot == HX_SLOT_PEPPER || slot == HX_SLOT_USERID)
				return 1;
		}
	}
	return 0;
}

/*
 * Compute the digest hex length (number of hex chars) the program emits for
 * a known short password.  The default hx role for digests is ROLE_HEX, so
 * the VM returns a hex string; its length / mirrors checkhash's len arg.
 */
static int probe_diglen_hex(hx_program *prog)
{
	hx_vm vm;
	hx_val r;
	int n;
	static const char probe[] = "abc";

	hx_vm_init(&vm, prog);
	r = hx_vm_run(&vm, probe, (int)sizeof(probe) - 1,
	              "", 0, "", 0, "", 0, "", 0);
	n = (r.data && r.len > 0) ? r.len : 0;
	hx_vm_free(&vm);
	return n;
}

/*
 * Milestone 2, sub-phase D5: content-hash identity suggestion.
 *
 * Compute a STABLE hash of the compiled bytecode so a shared userdef.txt
 * can use a self-describing content-hash id (per spec 2.6: an id that
 * identifies the algorithm independent of the local stanza name).  We use
 * an in-tree 64-bit FNV-1a over a canonical serialization of the opcode
 * stream -- no OpenSSL header dependency in this TU, fully deterministic.
 *
 * The serialization is canonical (same shape => same hash regardless of
 * string-table ordering / temporary spelling): for each instruction we mix
 * the opcode, then an operand discriminator:
 *   PUSH_VAR/STORE/INC : the slot index
 *   PUSH_INT           : the integer value
 *   PUSH_STR           : the string CONTENT + length (not the table index)
 *   CALL               : nargs, role, and the callee NAME bytes
 *   JUMP*              : the absolute addr
 * This mirrors the fields programs_equal() compares (minus the Layer-2
 * commutative canonicalization, which is not needed for an id nudge).
 */
static void fnv1a_mix(uint64_t *h, const void *p, size_t n)
{
	const unsigned char *b = (const unsigned char *)p;
	size_t i;
	for (i = 0; i < n; i++) {
		*h ^= (uint64_t)b[i];
		*h *= 1099511628211ULL;
	}
}

static void content_hash_hex(hx_program *prog, char *out, size_t outlen)
{
	uint64_t h = 1469598103934665603ULL;   /* FNV offset basis */
	int i;

	out[0] = '\0';
	if (!prog || !prog->code) return;

	for (i = 0; i < prog->ncode; i++) {
		const hx_inst *ip = &prog->code[i];
		uint8_t op = ip->op;
		fnv1a_mix(&h, &op, 1);
		switch (op) {
		case OP_PUSH_VAR:
		case OP_STORE:
		case OP_INC: {
			int32_t s = (int32_t)ip->u.slot;
			fnv1a_mix(&h, &s, sizeof(s));
			break;
		}
		case OP_PUSH_INT: {
			int64_t v = ip->u.ival;
			fnv1a_mix(&h, &v, sizeof(v));
			break;
		}
		case OP_PUSH_STR: {
			int idx = ip->u.stridx;
			if (idx >= 0 && idx < prog->nstrings) {
				int32_t sl = (int32_t)prog->strlens[idx];
				fnv1a_mix(&h, &sl, sizeof(sl));
				if (sl > 0 && prog->strings[idx])
					fnv1a_mix(&h, prog->strings[idx], (size_t)sl);
			}
			break;
		}
		case OP_CALL: {
			int32_t na = (int32_t)ip->u.call.nargs;
			uint8_t role = ip->u.call.role;
			const char *nm = (ip->u.call.entry && ip->u.call.entry->name)
			                 ? ip->u.call.entry->name : "?";
			fnv1a_mix(&h, &na, sizeof(na));
			fnv1a_mix(&h, &role, 1);
			fnv1a_mix(&h, nm, strlen(nm));
			break;
		}
		case OP_JUMP:    case OP_JUMP_LE: case OP_JUMP_LT:
		case OP_JUMP_GT: case OP_JUMP_GE: case OP_JUMP_EQ:
		case OP_JUMP_NE: {
			int32_t a = (int32_t)ip->u.addr;
			fnv1a_mix(&h, &a, sizeof(a));
			break;
		}
		default:
			break;   /* CONCAT/HALT/DUP/POP: opcode alone is enough */
		}
	}

	snprintf(out, outlen, "%016llx", (unsigned long long)h);
}

#ifdef USERDEF_HAVE_CODEGEN
/*
 * Build a call_names sidecar so programs_equal() can compare a freshly-
 * compiled user program against catalog entries.  Mirrors the helper in
 * tools/hx_dedup_check.c: snapshot each OP_CALL's entry->name.  Caller
 * frees with free().
 */
static const char **build_call_names_sidecar(const hx_program *prog)
{
	const char **names;
	int i;
	if (!prog || prog->ncode <= 0) return NULL;
	names = (const char **)calloc((size_t)prog->ncode, sizeof(*names));
	if (!names) return NULL;
	for (i = 0; i < prog->ncode; i++)
		if (prog->code[i].op == OP_CALL && prog->code[i].u.call.entry)
			names[i] = prog->code[i].u.call.entry->name;
	return names;
}

/*
 * Milestone 2, sub-phase C3: load-time dedup advisory.
 *
 * Compare the user program's compiled bytecode against every live catalog
 * entry via the SHARED programs_equal() comparator (the same one
 * tools/hx_dedup_check uses).  On a match, emit a NON-FATAL advisory
 * nudging the user toward the catalog type -- which has a hand-tuned GPU
 * path.  Never blocks: it is the user's namespace.  Best-effort: outlier /
 * compile-failed catalog entries (NULL .program) are simply skipped.
 *
 * Catalog cost is ZERO new link surface: hx_specs_data[] + the comparator
 * are already linked into mdxfind via $(CODEGEN_OBJS) + the header-only
 * comparator.
 */
static void dedup_advisory(const struct userdef_type *u)
{
	const char **ucn;
	int i;

	if (!u || !u->prog) return;
	ucn = build_call_names_sidecar(u->prog);

	for (i = 0; i < hx_specs_count; i++) {
		const struct hx_spec_entry *e = &hx_specs_data[i];
		if (!e->program) continue;             /* outlier / compile-fail */
		if (programs_equal(u->prog, ucn, e->program, e->call_names)) {
			UD_MSG(
			        "userdef: %s (-m u%s) is equivalent to catalog e%d "
			        "%s -- consider -m e%d for GPU acceleration.\n",
			        u->dispname, u->idstr, e->job_enum,
			        e->name ? e->name : "?", e->job_enum);
			break;     /* one advisory is enough; first match wins */
		}
	}

	free(ucn);
}

const char *userdef_gpu_status(int op)
{
	struct userdef_type *u = userdef_get(op);
	hx_pattern_id pid;

	if (!u || !u->prog) return NULL;

	pid = hx_detect_pattern(u->prog);
	if (pid == HX_PATTERN_UNKNOWN)
		return "GPU not available for this expression shape; "
		       "running on CPU.";
	return "GPU-eligible shape, but GPU dispatch for user-defined types "
	       "is not yet enabled; running on CPU.";
}
#endif /* USERDEF_HAVE_CODEGEN */

/* finalize one accumulated stanza into the registry */
static void finalize_stanza(const char *name, const char *idstr,
                            const char *hx, const char *path, int lineno)
{
	struct userdef_type *u;
	hx_program *prog;
	int i, diglen;

	if (!name[0]) return;             /* no header seen yet */

	if (!idstr[0]) {
		UD_MSG( "userdef: %s: stanza [%s] (near line %d) has no "
		        "'id =' key; skipping\n", path, name, lineno);
		return;
	}
	if (!hx[0]) {
		UD_MSG( "userdef: %s: stanza [%s] (near line %d) has no "
		        "'hx =' expression; skipping\n", path, name, lineno);
		return;
	}

	/* duplicate-id guard within this load */
	for (i = 0; i < Userdef_count; i++) {
		if (strcmp(Userdefs[i].idstr, idstr) == 0) {
			UD_MSG( "userdef: %s: stanza [%s] (near line %d) "
			        "reuses id '%s' (already used by [%s]); skipping\n",
			        path, name, lineno, idstr, Userdefs[i].name);
			/* NB: the FIRST loaded type owns this id; the duplicate
			 * is the skipped one.  -m u<id> still resolves to the
			 * loaded one, so we do NOT record a skip reason here
			 * (that would wrongly fatalize a valid selection). */
			return;
		}
	}

	if (Userdef_count >= USERDEF_MAX) {
		UD_MSG( "userdef: too many user types (max %d); "
		        "skipping [%s]\n", USERDEF_MAX, name);
		record_skip(idstr, "too many user types (registry full)");
		return;
	}

	prog = hx_compile_expr(hx, NULL);
	if (!prog) {
		UD_MSG( "userdef: %s: stanza [%s] (near line %d): hx "
		        "expression failed to compile: %s\n",
		        path, name, lineno, hx);
		record_skip(idstr, "hx expression failed to compile (parse error)");
		return;
	}

	/*
	 * Sub-phase C1: slot-inference reject-salted.  v1 supports only
	 * unsalted/unstructured expressions; any reference to
	 * salt/salt2/pepper/user is rejected at load with a clear message.
	 * This exercises the inference plumbing and de-risks v2.
	 */
	if (program_uses_salt(prog)) {
		UD_MSG( "userdef: %s (u%s): salted/structured user types "
		        "are not yet supported (v2); skipping\n",
		        name, idstr);
		record_skip(idstr, "salted/structured user types are not yet "
		            "supported (v2)");
		hx_program_free(prog);
		return;
	}

	diglen = probe_diglen_hex(prog);
	if (diglen <= 0 || (diglen & 1)) {
		UD_MSG( "userdef: %s: stanza [%s] (id %s): expression "
		        "produced an unusable digest (len %d); skipping\n",
		        path, name, idstr, diglen);
		record_skip(idstr, "expression produced an unusable digest");
		hx_program_free(prog);
		return;
	}

	u = &Userdefs[Userdef_count];
	memset(u, 0, sizeof(*u));
	strncpy(u->name, name, sizeof(u->name) - 1);
	strncpy(u->idstr, idstr, sizeof(u->idstr) - 1);
	strncpy(u->hx, hx, sizeof(u->hx) - 1);
	snprintf(u->dispname, sizeof(u->dispname), "USER_%s", name);
	u->prog       = prog;
	u->op         = JOB_USERDEF_BASE + Userdef_count;
	u->diglen_hex = diglen;
	u->uses_salt  = 0;
	Userdef_count++;

	UD_MSG( "userdef: loaded %s (-m u%s, op %d, %d-char digest): "
	        "%s\n", u->dispname, u->idstr, u->op, u->diglen_hex, u->hx);

	/*
	 * Sub-phase D5: content-hash identity suggestion.  Compute a stable
	 * bytecode hash and, when the user's id is NOT already that hash,
	 * suggest it as a self-describing shared id.  Soft-encourage only --
	 * never enforce, never rewrite the file.
	 */
	{
		char chash[24];
		content_hash_hex(u->prog, chash, sizeof(chash));
		if (chash[0] && strcmp(u->idstr, chash) != 0)
			UD_MSG( "userdef: %s (-m u%s): suggested stable "
			        "shared id = %s.\n", u->dispname, u->idstr, chash);
	}

#ifdef USERDEF_HAVE_CODEGEN
	/*
	 * Sub-phase C3: non-fatal dedup advisory against the built-in catalog.
	 * mdxfind-only: it depends on the codegen catalog hx_specs_data[] +
	 * the programs_equal() comparator, which are NOT linked into hashpipe
	 * (hashpipe links $(HX_OBJS) but not $(CODEGEN_OBJS)).  Guarded by
	 * USERDEF_HAVE_CODEGEN so the shared loader stays linkable into both.
	 */
	dedup_advisory(u);
#endif
}

int userdef_load(const char *cache_env)
{
	char path[1200];
	FILE *fp;
	char line[2200];
	char name[128] = {0}, idstr[128] = {0}, hx[2048] = {0};
	int lineno = 0, stanza_line = 0;

	if (derive_userdef_path(cache_env, path, sizeof(path)) != 0)
		return 0;   /* MDXFIND_CACHE unset; user types optional */

	fp = fopen(path, "r");
	if (!fp)
		return 0;   /* no file is fine; user types are optional */

	while (fgets(line, sizeof(line), fp)) {
		char *s, *eq;
		lineno++;

		/*
		 * Strip a trailing newline first.  Comment handling below is
		 * line-level: a '#' starting a (trimmed) line is a comment.
		 * We deliberately do NOT strip inline '#': the hx value is
		 * verbatim and may legitimately contain characters we must
		 * not interpret.  Full-line comments are the documented form.
		 */
		s = line;
		while (*s == ' ' || *s == '\t') s++;

		if (*s == '#' || *s == '\0' || *s == '\n' || *s == '\r')
			continue;   /* comment or blank */

		if (*s == '[') {
			/* new stanza header => finalize the previous one */
			char *close;
			finalize_stanza(name, idstr, hx, path, stanza_line);
			name[0] = idstr[0] = hx[0] = '\0';
			stanza_line = lineno;

			s++;
			close = strchr(s, ']');
			if (!close) {
				UD_MSG( "userdef: %s line %d: malformed "
				        "stanza header (no ']'); skipping\n",
				        path, lineno);
				continue;
			}
			*close = '\0';
			{
				char *hn = trim(s);
				strncpy(name, hn, sizeof(name) - 1);
				name[sizeof(name) - 1] = '\0';
			}
			continue;
		}

		/* key = value line */
		eq = strchr(s, '=');
		if (!eq) {
			UD_MSG( "userdef: %s line %d: not a "
			        "'key = value' line; ignoring\n", path, lineno);
			continue;
		}

		{
			char keybuf[64];
			char *key;
			int klen = (int)(eq - s);
			if (klen >= (int)sizeof(keybuf)) klen = sizeof(keybuf) - 1;
			memcpy(keybuf, s, klen);
			keybuf[klen] = '\0';
			key = trim(keybuf);

			if (keyis(key, "hx")) {
				/*
				 * VERBATIM value to end of line.  Take
				 * everything after '=' , strip exactly one
				 * leading space (the conventional "hx = ")
				 * and the trailing newline; do NOT tokenize.
				 */
				char *val = eq + 1;
				size_t vlen;
				if (*val == ' ') val++;
				vlen = strlen(val);
				while (vlen > 0 && (val[vlen-1] == '\n' ||
				                    val[vlen-1] == '\r'))
					val[--vlen] = '\0';
				strncpy(hx, val, sizeof(hx) - 1);
				hx[sizeof(hx) - 1] = '\0';
			} else if (keyis(key, "id")) {
				char *val = trim(eq + 1);
				strncpy(idstr, val, sizeof(idstr) - 1);
				idstr[sizeof(idstr) - 1] = '\0';
			} else {
				/* forward-compat: unknown key warn-and-ignore */
				UD_MSG( "userdef: %s line %d: unknown "
				        "key '%s' ignored (forward-compat)\n",
				        path, lineno, key);
			}
		}
	}

	/* finalize the last stanza */
	finalize_stanza(name, idstr, hx, path, stanza_line);

	fclose(fp);

	if (Userdef_count > 0)
		UD_MSG( "userdef: %d user-defined hash type(s) loaded "
		        "from %s\n", Userdef_count, path);

	return Userdef_count;
}
