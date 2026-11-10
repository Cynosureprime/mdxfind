/*
 * mdxint.h -- declarations shared by mdxfind.c (the hash engine: procjob and
 * everything it calls) and mdxmain.c (main, the hash and wordlist loaders,
 * statistics).  Definitions live in exactly one .c file; this header carries
 * only includes, macros, types, static inline helpers and extern declarations.
 *
 * $Revision: 1.4 $
 * $Log: mdxint.h,v $
 * Revision 1.4  2026/11/09 23:00:16  dlr
 * Include md6.h in mdxfind.c only. md6.h defines compression_hook instead of declaring it, and since the split both mdxfind.c and mdxmain.c included it through mdxint.h. gcc merges the two as a common symbol, so Linux, FreeBSD and macOS linked; llvm-mingw compiles with -fno-common and the Windows link failed with duplicate symbol compression_hook. mdxint.h now includes md6.h unless MDX_NO_MD6 is defined, and mdxmain.c, which uses only the JOB_MD6 constants, defines it. Verified with the Windows x86_64 release objects: mdxmain.o no longer defines the symbol and the release link line produces a 13,590,528-byte PE32+ executable.
 *
 * Revision 1.3  2026/11/09 20:54:20  dlr
 * Read pipeline: four read buffers, a reader thread, and an indexer that handles every line ending in a 64-byte block in one loop. A wordlist or hash file now passes through three stages, each owning one buffer of the NREADBUF=4 ring at a time: rb_reader reads, cacheline indexes on the caller thread, and the workers hash. On a cheap hash the run goes at the speed of the slowest stage rather than the sum of reading and indexing. Four rather than three because hashing usually holds two buffers, one draining its last jobs and one active. The reader hands cacheline whole lines only: after each read rb_cut finds the cut just past the last newline, or past the RINDEXSIZE-1 th newline if that comes first, and the rest is carried to the front of the next buffer, so the index cannot overflow. The cut applies at end of file too; a buffer is the last only when all that remains fits. As before, a newline in the last byte read waits for the next read, an unterminated tail of MAXLINE or more is dropped, and the last line of a file needs no newline. A buffer returns to the reader only when cacheline has moved past it and its job count, ReadBufL[i], is zero. ReadBuf0 and ReadBuf1 are replaced by the ReadBufL array and readbuf_lock(). gzread errors are now FATAL with file:line. Measured on mmt, reader-only e426 runs: 29m.pass x8 from NVMe, 232M lines, 2.31-2.48 s to 1.66-1.82 s at 36 threads and 2.92-3.04 s to 2.15-2.17 s at 72; hashmob.net_2026.found from ZFS, 2.49G lines, 51.9 s to 39.6-40.2 s, where the reader thread is now the limit and 83 percent of its time is ZFS LZ4 decompression and copy in the kernel; MD5 best64 on 10m.pass x3, 9.94-10.71 s to 9.24-9.35 s, 581,687 found both. -z output and line counts identical to the previous reader on arm64 and x86 across no trailing newline, empty, a lone newline, CRLF, $HEX[] edge cases, 300,000 lines of length 0-1000, lines at and past MAXLINE, a 30 MB line with and without a final newline, 20M lines plain and gzipped, stdin, and 30M one-byte lines that force cuts at the index cap. The one difference is -d on the one-byte files, 20 against 21 lines, which is expected: -d deduplicates within a buffer, so its output depends on where buffers are cut. ALSO: seconds hashing is no longer quantised to whole seconds. ReportStats waited out a sleep(1) before seeing the stop signal, and shutdown joins it before taking the end time; it now waits on a condition variable with a one-second timeout that stats_wake ends at once. MD4 skips the hex conversion after the last iteration, 18.15 s to 17.15 s on dev3 with identical results at x01-x04. The 1.666 log sentence in the $Log comment is corrected to match the rewritten RCS log.
 *
 * Revision 1.2  2026/11/09 16:10:46  dlr
 * Declare TypeDigest with TYPEDIGEST_LEN and TYPEDIGEST_FLAGS, the typedigest() accessor that derives a user-defined type length from userdef (0 when it declares a stored form), and the Extmatch, Precheck_off and Precheck_maxlen globals.
 *
 * Revision 1.1  2026/11/09 04:21:36  dlr
 * New. The includes, shared macros, types, static inline helpers and extern declarations for the split of mdxfind.c into the hash engine and mdxmain.c.
 *
 */
#ifndef MDXINT_H
#define MDXINT_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <wctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <signal.h>
#endif

/* Windows stat() overflows on files > 2GB; use 64-bit version */
#ifdef _WIN32
#define stat _stat64
#define fstat _fstat64
#include <windows.h>
#endif

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#endif

#include <sys/file.h>  /* GPU salt resume fix: gpu_opencl.c r1.53, gpu_metal.m r1.42 */
#include <dirent.h>
#include <errno.h>
#ifndef _WIN32
#include <sys/mman.h>     /* mlock() for pinned host buffers used by GPU upload */
#endif
#include <stdint.h>
#include <stdatomic.h>
#include "mdxhost.h"   /* mdx_gethostname() -- gethostname() is Winsock-only on Windows */

#ifdef SPARC
#define NOTINTEL 1
#endif
#ifdef ARM
#define NOTINTEL 1
#endif
#ifdef POWERPC
#define NOTINTEL 1
#endif
#ifndef NOTINTEL

#include <emmintrin.h>
#include <xmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <cpuid.h>

#endif

#include <Judy.h>
#include <pcre.h>
#include <mhash.h>
#include <rhash.h>
#include "yarn.h"
/* md6.h DEFINES compression_hook rather than declaring it, so only one
 * translation unit may include it.  mdxfind.c, which hashes with MD6, does;
 * mdxmain.c defines MDX_NO_MD6.  gcc merges the duplicate as a common symbol,
 * but a -fno-common toolchain (llvm-mingw) refuses the link. */
#ifndef MDX_NO_MD6
#include "md6.h"
#endif
#include <sph_blake.h>
#include "gosthash/gost2012/streebog.h"
#include "gosthash/gosthash.h"
#include "openssl/des.h"
#include "openssl/aes.h"
#include "openssl/rc4.h"
#include "openssl/hmac.h"
#include "openssl/evp.h"
#include "yescrypt/yescrypt.h"
#include <sph_bmw.h>
#include <sph_cubehash.h>
#include <sph_echo.h>
#include <sph_fugue.h>
#include <sph_groestl.h>
#include <sph_hamsi.h>
#include <sph_haval.h>
#include <sph_jh.h>
#include <sph_keccak.h>
#include <sph_luffa.h>
#include <sph_md2.h>
#include <sph_md4.h>
#include <sph_md5.h>
#include <sph_panama.h>
#include <sph_radiogatun.h>
#include <sph_ripemd.h>
#include <sph_sha1.h>
#include <sph_sha2.h>
#include <sph_sha3.h>
#include <sph_shabal.h>
#include <sph_shavite.h>
#include <sph_simd.h>
#include <sph_skein.h>
#include <sph_tiger.h>
#include <sph_types.h>
#include <sph_whirlpool.h>

#include <zlib.h>
#include <lzma.h>
#include <bzlib.h>
#include <zstd.h>
#include "sqlite3.h"

#include "mdxfind.h"
/* AFTER mdxfind.h, which defines MAXLINE: ruleproc32.h carries a fallback
 * MAXLINE for the standalone test tools, and this order makes mdxfind's
 * win, exactly as procrule.c does it. */
#include "rule_ops.h"   /* RULE_OP_* opcode numbers, shared with both engines */
#include "ruleproc32.h"
#include "classify_utf8.h"
#include "userdef.h"  /* unknown-key forward-compat fix: userdef.c r1.11 */


#include "job_types.h"   /* the JOB_ numbers, generated from mdxfind.c */

#define TYPENAME(op) (userdef_is_userop(op) ? userdef_name(op) : Types[op])
#ifndef O_BINARY
#define O_BINARY 0
#endif
extern unsigned Userdef_base;
/* op <-> user-id mapping. Valid only after Userdef_base is derived. */
static inline int      op_is_user(unsigned op) { return op >= Userdef_base; }
static inline unsigned op_user_id(unsigned op) { return op - Userdef_base; }
static inline unsigned user_id_op(unsigned id) { return Userdef_base + id; }
extern struct Linehints Userlinehints[USERDEF_MAX];
static inline struct Linehints *linehints_ref(unsigned op) {
    return op_is_user(op) ? &Userlinehints[op_user_id(op)] : &linehints[op];
}
#define LINEHINTS(op) (*linehints_ref(op))
/* Per-type compare length in bytes -- see TypeDigest[] in mdxfind.c.  A
 * user-defined type takes its length from userdef's load-time probe; one with
 * a stored-form declaration matches through JUDYJ, not the compact table, so
 * it reports 0 like a NEEDSJ built-in. */
extern unsigned int TypeDigest[JOB_DONE];
#define TYPEDIGEST_LEN(v)   ((v) & 0xffffu)
#define TYPEDIGEST_FLAGS(v) ((v) >> 16)
static inline unsigned int typedigest(unsigned op) {
    if (op_is_user(op)) {
        struct userdef_type *u = userdef_get((int)op);
        return (u && u->form.npieces == 0 && u->diglen_hex > 0) ? (unsigned)u->diglen_hex / 2 : 0;
    }
    return op < JOB_DONE ? TypeDigest[op] : 0;
}
extern int Extmatch, Precheck_off, Precheck_maxlen;
#if (defined(__APPLE__) && defined(METAL_GPU)) || defined(CUDA_GPU) || defined(OPENCL_GPU)
#define GPU_ENABLED 1
#endif
#if defined(OPENCL_GPU) || (defined(__APPLE__) && defined(METAL_GPU))
#define GPU_BF_CHUNK_PRODUCER 1
#endif
#ifdef GPU_ENABLED
#if defined(__APPLE__) && defined(METAL_GPU)
#include "gpu_metal.h"  /* Metal iter-axis dispatch budget: gpu_metal.m r1.133, gpu/metal_common.metal r1.34 */
#elif defined(CUDA_GPU)
#include "cuda_md5salt.h"
#elif defined(OPENCL_GPU)
#include "gpu_opencl.h"
#include "gpu_kernel_cache.h"
#endif
/* hx codegen sub-phase 2a.1 (2026-05-21): in-process P4 state-machine
 * walker + trivial-spec harness. OpenCL only at 2a.1; Metal harness
 * arrived in 2a.4 so the codegen headers are now included on BOTH
 * GPU-enabled backend builds (they are pure C; no backend symbols). */
#if defined(OPENCL_GPU) || (defined(__APPLE__) && defined(METAL_GPU))
#include "codegen/hx_spec.h"
#include "codegen/hx_walker.h"
#include "codegen/hx_spec_entry.h"
/* hx codegen sub-phase 5a.5 (2026-05-22): family admit-predicate helper
 * for routing the seven 5a-eligible MAKE_MD5PASS family JOB enums through
 * the production kernel-B codegen dispatcher. Pure C; usable from both
 * OpenCL and Metal backend builds. */
#include "gpu/gpu_codegen_eligible.h"
#endif
#define NO_JOB_TYPES 1
#include "gpujob.h"
#endif
#ifdef GPU_ENABLED
/* Backend-neutral device enumeration, same shim shape and for the same
 * reason as mdx_kernel_a_proto_enabled above: calling a gpu_opencl_*
 * symbol unguarded breaks the Metal build at link time (that is what
 * rev 1.488 did).
 *
 * These three exist so the brute-force chunk-as-job producer can stop
 * being OpenCL-only. The producer was never OpenCL-specific in its
 * logic -- it had been gated on OPENCL_GPU because these were the only
 * backend-bound calls inside it. The Metal kernel side
 * (metal_kernel_a_bruteforce, mask_start / mask_offset_per_word /
 * bf_num_masks / bf_inner_iter honoured in dispatch, chunk base re-added
 * in the hit decode) was already complete and correct.
 *
 * Contracts, which both backends must keep:
 *   num_devices          enumerated devices, >= 1 when a backend is up.
 *   active_device_count  devices that will actually receive a dispatch.
 *                        The BF bootstrap wait waits on THIS count; using
 *                        the enumerated count instead is what stalled
 *                        every multi-chunk BF run to the 180 s cap on a
 *                        rig with one disabled device.
 *   dev_uuid             stable per-device key for the BF rate sidecar,
 *                        FNV-1a 64-bit as "%016llx", out_sz >= 17. A key
 *                        that moved between runs would silently disable
 *                        persistence rather than fail. */
static inline int mdx_gpu_num_devices(void)
{
#if defined(__APPLE__) && defined(METAL_GPU)
    return gpu_metal_num_devices();
#elif defined(OPENCL_GPU)
    return gpu_opencl_num_devices();
#else
    return 0;
#endif
}
#endif
#ifndef NOTINTEL
extern int HasSSSE3;
#endif
#ifdef ARM
extern int Neon;
#endif
#ifdef ARM
#define mysha1 SHA1
#endif
#ifdef POWERPC
#define mysha1 SHA1
#endif
#ifdef SPARC
#define mysha1 SHA1
#endif
#ifdef SPARC
#define mymd5 MD5
#endif
#ifdef AIX
#define mysha1 SHA1
#endif
extern char *Version;
__attribute__((unused)) const char *mdxfind_rev_string(void);
extern void WHIRLPOOL(char *cur, int len, unsigned char *dest);
extern void mysha1(void *cur, int len, unsigned char *dest);
extern void mysha256(char *cur, int len, unsigned char *dest);
extern void mysha512(char *cur, int len, unsigned char *dest);
extern void RIPEMD160(char *cur, int len, unsigned char *dest);
extern void RIPEMD128(char *cur, int len, unsigned char *dest);
extern void MD4(char *cur, int len, unsigned char *dest);
extern void gosthash(char *cur, int len, unsigned char *dest);
extern const struct cppenc_params CPPENC_LEGACY;
extern const struct cppenc_params CPPENC_DEFAULT;
int cppenc_reclen_ok(const struct cppenc_params *p, int reclen);
/* ------------------------------------------------------------------
 * 7-Zip (7zAES) archive records -- JOB_SEVENZIP / e1000 / hashcat 11600.
 *
 * Stage 1 (the only thing in the hot path) needs just the last TWO
 * ciphertext blocks and the pad size, so that is all we keep. The KDF
 * output depends solely on (salt, log2_iter), so archives are grouped by
 * that and the 2^19-round key derivation is paid ONCE per group per
 * candidate no matter how many archives share it. With the CMIYC
 * challenge_2 set (all saltlen=0, all 2^19) every archive lands in one
 * group -- five archives for the price of one derivation.
 * ------------------------------------------------------------------ */
struct SevenZipRec {
  char group[96];            /* KDF group key: saltlen:salthex:log2 */
  unsigned char tail[32];    /* final two ciphertext blocks; tail[0..15] is the
                              * CBC IV for tail[16..31] */
  unsigned char head[32];    /* first two ciphertext blocks (tier 1) */
  unsigned char iv[16];      /* archive IV -- decrypts from offset 0 */
  unsigned char *attrs;      /* coder properties, NULL when absent */
  unsigned char *data;       /* whole ciphertext stream, NULL when tail-only */
  char *line;                /* full input line: JudyJ key + output */
  long datalen;              /* bytes in data */
  long packed, unpacked;     /* declared lengths */
  long crc_len;              /* bytes the stored CRC32 covers; 0 when absent */
  unsigned int crc;          /* stored CRC32 */
  int attrlen;
  int type;                  /* $7z$ data type indicator -- see hx.8 / the spec */
  int padsize;               /* packedlen - unpackedlen, 0..15 */
  int bits;                  /* tier-0 evidence strength, 8 * padsize */
  int havehead;              /* head[] is populated */
  int tier2;                 /* whole stream retained -> decompress+CRC possible */
};
extern struct SevenZipRec *SevenZipRecs;
extern int SevenZipCount , SevenZipCap;
extern long SevenZipRejected , SevenZipWarned , SevenZipNoisy;
/* Largest log2(iterations) the KDF loop will actually run. 2^40 already takes
 * geological time; anything above it is a corrupt or hostile record, and 0x3F
 * is the format's "no hashing" special case, which needs a different KDF. */
#define SEVENZIP_MAX_LOG2 40
extern long SevenZipMinBits;
extern int SevenZipStrict;
void sevenzip_reject(const char *why, const char *line);
struct SevenZipRec *sevenzip_addrec(void);
extern size_t Argon2_maxmem;
extern void mymd5(void *message, int len, unsigned char *hash);
extern int Rulequiet;
extern const char *Ruleecho;
extern void getcpuinfo();
extern lock *Memlock;
void *malloc_lock(size_t size, const char *reason);
extern lock *TsfLock;
extern struct timespec tsfprintf_origin;
extern int tsfprintf_origin_set;
struct LineInfo {
  unsigned int offset;
  unsigned short int len;
  /* -8: the word's encoding class from classify_utf8(), one of the U8C_*
   * masks, or U8C_NONE when -8 is off and no classification was done.
   *
   * This costs NOTHING: offset + len is 6 bytes in an 8-byte-aligned struct,
   * so two padding bytes were already here.  The static assert below is the
   * guard -- if a future field pushes sizeof past 8, every wordlist read in
   * the program gets 33% wider, and the compile should stop rather than let
   * that happen quietly. */
  unsigned char enc;
};
extern char *Curfile;
extern unsigned long long int Lowline, LowSkip, TotLines, Fileline;
extern Pvoid_t JuniperIVE;
extern Pvoid_t DoUser;
extern Pvoid_t Dohash, Doload, SaltArray, UserArray, KeyArray, PepperArray;
extern struct rulestore Rulestore, Rulefile;
/* Hybrid compact hash table (replaces Bloom + JudyL HashL) */
#define COMPACT_MAX_PROBE 16
extern unsigned char *HashDataBuf;
extern size_t HashDataBufUsed, HashDataBufCap;
extern size_t *HashDataOff;
extern unsigned short *HashDataLen;
extern unsigned short *HashDataFlags;
extern size_t HashDataCount, HashDataCap;
extern uint32_t *CompactFP;
extern uint32_t *CompactIdx;
extern uint64_t CompactSize, CompactMask, CompactUsed;
extern Pvoid_t OverflowHash;
static inline uint64_t compact_mix(uint64_t k) {
    return k ^ (k >> 32);
}
extern int NoMarkSalt, Hexkey, Rotatehash, Unicode, Dedupe, XMLchar, Email, Printall;
extern int Contfound;
extern int EmailLocal;
/* -a dot-sweep width ceiling.  A local part of length L has 2^L dottings, so
 * this is the one axis in the tool whose cost per input line spans eight orders
 * of magnitude.  24 was an unnamed literal; naming it makes the cost visible
 * and the decision reviewable.  It is NOT free: PLAN-email-address-mode section
 * 2 measured 226,707 dotted locals carrying five or more dots, and the
 * 30-character thefamousdogandcatshowyoulike1 is swept only to position 24, so
 * the clamp has already cost real recoveries.  Raising it is plan section 9
 * decision 3, still open -- what the width's syntax and default should be.
 * Until then the clamp stays and the end-of-run summary says how often it bit,
 * because a sweep that covered 24 of 30 positions looks exactly like one that
 * covered all of them. */
#define EMAIL_DOT_WIDTH_MAX 24
/* -a: the narrowest dot sweep worth a GPU dispatch.
 *
 * 5, not 14.  The first value came from arithmetic about dispatch size and was
 * WRONG, because the variable it tuned is not the one that matters.  Measured on
 * fpga (GTX 1080) with UNIFORM-width synthetic lists, device candidate rate:
 *
 *     width  6 ->  8.4e8/s     width 10 -> 1.01e9/s
 *     width  8 -> 1.01e9/s     width 11 ->  9.8e8/s
 *     width  9 -> 1.03e9/s     width 13 ->  9.2e8/s
 *
 * Flat from 6 upward, against a CPU that does ~5.8e7 candidates/s -- the device
 * is ~17x faster at EVERY width, so there is no crossover to find.  The ~2 s of
 * idle in those runs is fixed start-up, identical in all of them, not
 * per-dispatch overhead.
 *
 * What actually destroys -a throughput is the batch KEY changing, not the width
 * being small.  Same 4M words, same width range, on the same card:
 *
 *     uniform width 6        2.30 s wall, 0.31 s busy,        246 batches
 *     widths 5 or 6 at random  130.06 s wall, 87.00 s busy, 2,000,719 batches
 *
 * 44x, one batch per two words.  That is the section 15.1 fragmentation case,
 * and the cure is ordering the input by DOTLESS local length -- which is what
 * utils/emailprep does and what the warning below asks for -- not declining
 * narrow words.  5 remains only as a floor against a pathological micro-sweep;
 * nothing measured argues for more. */
#define EMAIL_GPU_MIN_DOT_W 5
extern long Email_clamped;
extern long Email_batch_splits;
extern long Email_cpu_cheap;
#ifdef GPU_ENABLED
extern int gpu_dot_width_used;
#endif
#ifdef GPU_ENABLED
extern int gpu_dot_width_asked;
#endif
extern long Email_unsorted;
/* Bytes to hand classify_utf8() for the -8 engine choice.  Byte 0x40 cannot
 * occur inside a multi-byte UTF-8 sequence, so finding the '@' before the
 * decode is exactly as correct as finding it after. */
static inline int enclen_E(const char *w, int n)
{
    const char *atp;
    if (!EmailLocal) return n;
    atp = memchr(w, '@', (size_t)n);
    return atp ? (int)(atp - w) : n;
}
#ifdef GPU_ENABLED
extern int NoMetal;
#endif
#ifdef GPU_ENABLED
#else
extern int NoMetal;
#endif
#ifdef GPU_ENABLED
extern int gpu_device_allowed[64];
#endif
extern int Debug, DoHistogram;
struct RuleHist {
  char *rule;
  unsigned long long val;
  unsigned short rlen;   /* rule[] is NOT NUL-terminated: Rulesrc is packed */
};
extern char *MD5_user;
extern char *combo_pepper;
extern Pvoid_t *Typesalt;
extern Pvoid_t *UserTypesalt;
extern Pvoid_t *DashSSalt;
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline Pvoid_t *typesalt_ref(unsigned op) {
    return op_is_user(op) ? &UserTypesalt[op_user_id(op)] : &Typesalt[op];
}
#define TYPESALT(op) (*typesalt_ref(op))
extern Pvoid_t *Typehashsalt;
extern int *Typesaltcnt;
extern long long *Typesaltbytes;
struct saltentry {
    char *salt;      /* points into thread-local saltpool (copy of Judy key) */
    Word_t *PV;      /* cached PV pointer — stable for Judy lifetime */
    int saltlen;     /* cached strlen */
    char *hashsalt;  /* precomputed hex(MD5(salt)) from Typehashsalt, or NULL */
    int hashlen;     /* length of hashsalt string (typically 32) */
    uint32_t iter;   /* internal iteration count for GPU Tothash accounting;
                      * 0 = not set (gpu_compute_iter_sum falls back to
                      * default_iter_for_op). Populated by gpu_compute_iter_sum
                      * at GPU dispatch accounting time; not set by CPU paths
                      * (unused there). For DESCRYPT/MD5CRYPT/SHA1DRU: fixed
                      * constant; for SHACRYPT: parsed from rounds= prefix in
                      * salt string; for BCRYPT: 1<<cost from salt prefix;
                      * for PHPBB3: 1<<i64hex[salt[3]]. */
};
extern char **Typesalt2;
extern void **Typeuser;
extern void **UserTypeuser;
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline void **typeuser_ref(unsigned op) {
    return op_is_user(op) ? &UserTypeuser[op_user_id(op)] : &Typeuser[op];
}
#define TYPEUSER(op) (*typeuser_ref(op))
extern Pvoid_t UseridJudy;
extern char **Typepepper;
extern int XOR_key_len;
extern char *XOR_key;
extern char *Keys, *Readbuf, *Rules, *Peppers;
extern char *Keys_end;
extern long Keycount;
extern int Keymaxlen;
extern int Rulesize;
extern unsigned char *gpu_rule_program;
extern uint32_t *gpu_rule_offsets;
extern int *gpu_rule_slot;
extern uint32_t gpu_rule_program_len;
extern int gpu_rule_count;
extern char *Rulesrc;
extern uint32_t *Rulesrcoff;
extern unsigned short *Rulesrclen;
extern int Utf32Rules;
#define U32LOC_ROOT     1   /* -L c|root|posix|none: variant 0, chosen      */
#define U32LOC_TR       2   /* -L tr|az|turkish|azeri: variant 1            */
extern int Utf32Locale;
extern long Rule_u32_only;
extern long Rule_both_bad;
extern unsigned char *Rule_u32ok;
extern char *Rule_u32src;
extern uint32_t *Rule_u32srcoff;
extern unsigned char *Rule_byteok;
extern volatile long Unreachable;
extern volatile long Variants_dropped;
int validrule_u32(const char *plainrule);
#if defined(OPENCL_GPU) || defined(METAL_GPU)
extern int gpu_u32_membership_prefilter(const unsigned char *byte_prog, uint32_t byte_len, const uint32_t *byte_offs, int n_rules);
#endif
extern unsigned char *gpu_rule_membership;
extern int gpu_legacy_slot_unused;
extern int gpu_rules_engine_active;
extern struct LineInfo *Readindex;
extern struct timespec starthash, current;
extern unsigned int Numsalts, Totsalts, Numrules, Numpeppers;
extern volatile unsigned int *Livesalts;
extern volatile unsigned int *UserLivesalts;
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline volatile unsigned int *livesalts_ref(unsigned op) {
    return op_is_user(op) ? &UserLivesalts[op_user_id(op)] : &Livesalts[op];
}
#define LIVESALTS(op) (*livesalts_ref(op))
extern unsigned int Minhashlen, Maxdepth;
extern unsigned long long Dupcnt, Insize;
extern unsigned char *inhash1, *inhash2, *inhashbuf, *inhashstart;
extern int inhash, inhashcnt;
void current_utc_time(struct timespec *ts);
char *memndup(void *source, int len);
extern char phpitoa64[];
extern char b64[];
extern signed char phpatoi64[256];

void init_phpatoi64(void);
int md5crypt_b64decode(const char *b64str, unsigned char *out);
void sha1crypt_b64encode(const unsigned char *d, char *out);
int sha1crypt_b64decode(const char *s, unsigned char *out);
int phpass_b64decode(const char *b64str, unsigned char *out);
int b64_decode(char *b64src, char *clrdst, int *incount);
int lotus64_decode(const char *in, unsigned char *out, int nchars);
int lotus64_encode(const unsigned char *in, int inlen, char *out, int outmax);
__attribute__((noinline)) int b64_encode(char *clrstr, char *b64dst, int inlen);
int crypt64_val(int c);
int crypt64_uint32(const char *s, int nchar, unsigned int *out);
int crypt64_bytes(const char *s, int nchar, unsigned char *out, int outmax);
#ifdef NOTINTEL
char *findeol(char *s,int l);
#endif
#ifdef NOTINTEL
#else
char *findeol(char *s, int l);
#endif
size_t mystrlen(const void *v);
void gen_userid();
extern int (*get32)(char *iline, unsigned char *dest, int len);
extern unsigned char i64hex[];
extern unsigned char trhex[];
char *commify(unsigned long long source);
static char hextab[16] = "0123456789abcdef";
static char hextabUC[16] = "0123456789ABCDEF";

/* 512-byte LUT: hexlut_lc[byte*2] gives two lowercase ASCII hex chars */
static const char hexlut_lc[512] =
    "000102030405060708090a0b0c0d0e0f"
    "101112131415161718191a1b1c1d1e1f"
    "202122232425262728292a2b2c2d2e2f"
    "303132333435363738393a3b3c3d3e3f"
    "404142434445464748494a4b4c4d4e4f"
    "505152535455565758595a5b5c5d5e5f"
    "606162636465666768696a6b6c6d6e6f"
    "707172737475767778797a7b7c7d7e7f"
    "808182838485868788898a8b8c8d8e8f"
    "909192939495969798999a9b9c9d9e9f"
    "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
    "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
    "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
    "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
    "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
    "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";

static const char hexlut_uc[512] =
    "000102030405060708090A0B0C0D0E0F"
    "101112131415161718191A1B1C1D1E1F"
    "202122232425262728292A2B2C2D2E2F"
    "303132333435363738393A3B3C3D3E3F"
    "404142434445464748494A4B4C4D4E4F"
    "505152535455565758595A5B5C5D5E5F"
    "606162636465666768696A6B6C6D6E6F"
    "707172737475767778797A7B7C7D7E7F"
    "808182838485868788898A8B8C8D8E8F"
    "909192939495969798999A9B9C9D9E9F"
    "A0A1A2A3A4A5A6A7A8A9AAABACADAEAF"
    "B0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF"
    "C0C1C2C3C4C5C6C7C8C9CACBCCCDCECF"
    "D0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF"
    "E0E1E2E3E4E5E6E7E8E9EAEBECEDEEEF"
    "F0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF";
__attribute__((noinline)) char *prmd5(unsigned char *md5, char *out, int len);
/*
 * Fixed-width hex: prmd5_N(in, out) is prmd5(in, out, N), output and NUL
 * identical.  914 of the 1,156 prmd5 calls pass a constant width, and as a
 * loop each one was inlined and re-unrolled by the compiler: loop passes were
 * half of procjob's optimize time.  These have no loop, read exactly N/2
 * input bytes and write N+1, and do 16 bytes at a time with SSE2 or NEON,
 * 2.9x (i9-9900K) to 6.8x (M1) faster than the table loop for 32.  The
 * generic forms remain for variable widths.
 */
static inline void hex2_lut(const char *lut, const unsigned char *in, char *out, int n) {
  for (int x = 0; x < n; x++)
    memcpy(out + 2 * x, &lut[in[x] * 2], 2);
}

#if !defined(NOTINTEL)
static inline void hex16(const unsigned char *in, char *out, int uc) {
  __m128i v = _mm_loadu_si128((const __m128i *)in), m = _mm_set1_epi8(0x0f);
  __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m), lo = _mm_and_si128(v, m);
  __m128i a = _mm_unpacklo_epi8(hi, lo), b = _mm_unpackhi_epi8(hi, lo);
  __m128i nine = _mm_set1_epi8(9), z = _mm_set1_epi8('0');
  __m128i adj = _mm_set1_epi8(uc ? 'A' - '0' - 10 : 'a' - '0' - 10);
  a = _mm_add_epi8(_mm_add_epi8(a, z), _mm_and_si128(_mm_cmpgt_epi8(a, nine), adj));
  b = _mm_add_epi8(_mm_add_epi8(b, z), _mm_and_si128(_mm_cmpgt_epi8(b, nine), adj));
  _mm_storeu_si128((__m128i *)out, a);
  _mm_storeu_si128((__m128i *)(out + 16), b);
}
#elif defined(ARM) && defined(__aarch64__)
static inline void hex16(const unsigned char *in, char *out, int uc) {
  const uint8x16_t t = vld1q_u8((const uint8_t *)(uc ? hextabUC : hextab));
  uint8x16_t v = vld1q_u8(in);
  uint8x16_t h = vqtbl1q_u8(t, vshrq_n_u8(v, 4));
  uint8x16_t l = vqtbl1q_u8(t, vandq_u8(v, vdupq_n_u8(0x0f)));
  vst1q_u8((uint8_t *)out, vzip1q_u8(h, l));
  vst1q_u8((uint8_t *)out + 16, vzip2q_u8(h, l));
}
#else
static inline void hex16(const unsigned char *in, char *out, int uc) {
  hex2_lut(uc ? hexlut_uc : hexlut_lc, in, out, 16);
}
#endif

/* Straight-line, no loop.  A width that is not a multiple of 16 bytes is two
 * OVERLAPPING 16-byte conversions -- 20 bytes is 0..15 then 4..19 -- which
 * write the same characters twice over the overlap and never read past
 * the input. */
#define HEXPAIR(L, in, out, i) memcpy((out) + 2 * (i), &(L)[(in)[i] * 2], 2)
#define PRHEX_FNS(SUF, UC, L)                                                         \
  static inline char *prmd5##SUF##_16(const unsigned char *in, char *out) {           \
    HEXPAIR(L, in, out, 0); HEXPAIR(L, in, out, 1); HEXPAIR(L, in, out, 2);           \
    HEXPAIR(L, in, out, 3); HEXPAIR(L, in, out, 4); HEXPAIR(L, in, out, 5);           \
    HEXPAIR(L, in, out, 6); HEXPAIR(L, in, out, 7);                                   \
    out[16] = 0; return out; }                                                        \
  static inline char *prmd5##SUF##_32(const unsigned char *in, char *out) {           \
    hex16(in, out, UC); out[32] = 0; return out; }                                    \
  static inline char *prmd5##SUF##_40(const unsigned char *in, char *out) {           \
    hex16(in, out, UC); hex16(in + 4, out + 8, UC); out[40] = 0; return out; }        \
  static inline char *prmd5##SUF##_56(const unsigned char *in, char *out) {           \
    hex16(in, out, UC); hex16(in + 12, out + 24, UC); out[56] = 0; return out; }      \
  static inline char *prmd5##SUF##_64(const unsigned char *in, char *out) {           \
    hex16(in, out, UC); hex16(in + 16, out + 32, UC); out[64] = 0; return out; }      \
  static inline char *prmd5##SUF##_96(const unsigned char *in, char *out) {           \
    hex16(in, out, UC); hex16(in + 16, out + 32, UC); hex16(in + 32, out + 64, UC);   \
    out[96] = 0; return out; }                                                        \
  static inline char *prmd5##SUF##_128(const unsigned char *in, char *out) {          \
    hex16(in, out, UC); hex16(in + 16, out + 32, UC);                                 \
    hex16(in + 32, out + 64, UC); hex16(in + 48, out + 96, UC);                       \
    out[128] = 0; return out; }
PRHEX_FNS(, 0, hexlut_lc)
PRHEX_FNS(UC, 1, hexlut_uc)
extern int Maxiter, ThreadCount, Maxniter;
struct MapHashcat {
  unsigned int hc, mdx;
};
extern struct MapHashcat Maphashcat[];
extern int Arg_have_salt;
extern int Arg_have_z;
extern int Arg_have_user;
extern char *Types[];
/* Windows default iteration count, and the value the $DCC2$ loader already
 * falls back to when the wrapper carries no usable count. Used when the
 * username arrives via -u/-U, which carry no iteration count of their own. */
#define DCC2_DEFAULT_ITER 10240
extern Pvoid_t JudyJ[JOB_DONE];
extern Pvoid_t UserJudyJ[USERDEF_MAX];
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline Pvoid_t *judyj_ref(unsigned op) {
    return op_is_user(op) ? &UserJudyJ[op_user_id(op)] : &JudyJ[op];
}
#define JUDYJ(op) (*judyj_ref(op))
extern char Typedone[JOB_DONE];
extern char UserTypedone[USERDEF_MAX];
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline char *typedone_ref(unsigned op) {
    return op_is_user(op) ? &UserTypedone[op_user_id(op)] : &Typedone[op];
}
#define TYPEDONE(op) (*typedone_ref(op))
#define TYPEOPT_NEEDSJ     1
#define TYPEOPT_NEEDSF     2
#define TYPEOPT_SALTJUDY   4
#define TYPEOPT_USERJUDY   8
#define TYPEOPT_NEEDSALT   32
#define TYPEOPT_NEEDUSER   64
#define TYPEOPT_NEEDPEPPER 128
extern unsigned short TypeOpts[JOB_DONE];
extern unsigned short UserTypeOpts[USERDEF_MAX];
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline unsigned short *typeopts_ref(unsigned op) {
    return op_is_user(op) ? &UserTypeOpts[op_user_id(op)] : &TypeOpts[op];
}
#define TYPEOPTS(op) (*typeopts_ref(op))
/* Mask-based hybrid attack support.
 *
 * MAX_MASK_POS is the CPU per-pattern position cap (one side: -n or -N
 * alone, or one side of -N + -n combined). Raised from 16 to 256 on
 * 2026-05-26 to accommodate long literal-suffixed customer masks such
 * as "?d?d?d?d?d?d?d?d?d?d@myecou.com" (21 positions). Memory cost is
 * negligible: each struct MaskPos is 8 bytes; the 3 pattern arrays
 * grow from 128 B to 2 KB total.
 *
 * GPU per-side caps (2026-05-30 long-mask amendment, supersedes the
 * single MAX_MASK_POS_GPU_SIDE=16 cap):
 *
 *   GPU_MASK_VAR_CAP       16   maximum variable (placeholder ?d/?l/?a
 *                                /?[abc]) positions per side. Matches
 *                                pre-amendment cap; per-thread private
 *                                scratch arrays for placeholder expansion
 *                                stay at 16 entries.
 *
 *   GPU_MASK_LIT_BYTES_CAP 224  maximum literal bytes per side. Decoupled
 *                                from variable count. Lifts the 2026-05-30
 *                                gate fixture
 *                                  -N "This is triumph, I'm making a note here: ?a?a?a?a?a?d?d"
 *                                (42 literal + 7 variable = 49 positions)
 *                                onto the GPU path. Total per-side <=
 *                                lit_bytes_cap + var_cap = 240; combined
 *                                with input word <=15 stays under
 *                                RULE_BUF_LIMIT=255.
 *
 *   GPU_MASK_DESC_BYTES_CAP 320 sizing for the wire-format run-descriptor
 *                                stream (see below). Per-side persistent
 *                                cl_mem allocation is 320 B.
 *
 * Wire format (Design Y, amendment D1.b): the prepend / append mask
 * buffers are interleaved run-descriptor streams of:
 *
 *   GPU_MASK_DESC_TAG_LIT  (0x00) lit_len:u16 (little-endian) bytes[lit_len]
 *   GPU_MASK_DESC_TAG_VAR  (0x01) classid:u8     (one placeholder position)
 *   GPU_MASK_DESC_TAG_END  (0xFF) terminator (no payload)
 *
 * Worst-case packing per side (16 var + 17 lit + END): 308 B (well under
 * the 320 B persistent allocation). gpu_pack_mask_descriptor() in this
 * file is the host-side encoder; mask_expand_run_into_gpu() in
 * gpu_kernel_a_masks.cl is the GPU walker.
 *
 * Backward compat: MAX_MASK_POS_GPU_SIDE retained as a deprecated alias
 * for GPU_MASK_VAR_CAP (16). Pre-amendment short-mask workloads (e.g.
 * -n ?d?d, 0 lit + 2 var) take the same unified walker as long-mask
 * workloads, producing byte-identical multisets.
 *
 * Backend coverage: the OpenCL A2/A3/A4 dispatchers consume the
 * descriptor wire format. The Metal twins follow per amendment §9.
 * The template/slab kernel path (gpu_template.cl / metal_template.metal)
 * is on a separate upload wire (gpu_opencl_set_mask /
 * gpu_metal_set_mask) and is UNAFFECTED by this amendment; that path
 * retains its own MASK_POS_CAP=16 per-side gate.
 *
 * The GPU upload sites (mdxfind.c, the OpenCL+Metal A2/A3/A4 family
 * blocks below) now check (var_count <= GPU_MASK_VAR_CAP &&
 * lit_bytes <= GPU_MASK_LIT_BYTES_CAP) per side; on overflow they
 * fall back to CPU iteration. The template/slab path still emits its
 * own MAX_MASK_POS_GPU_SIDE warning. */
#define MAX_MASK_POS 256
#define MAX_MASK_POS_GPU_SIDE 16        /* deprecated alias for GPU_MASK_VAR_CAP */
#define GPU_MASK_VAR_CAP        16      /* placeholder positions per side */
/* Template / rules-engine mask SLOT cap per side (B7.9, 2026-10-26: 16 -> 32).
 * Distinct from GPU_MASK_VAR_CAP above, which bounds the kernel-A descriptor
 * walker's PLACEHOLDER count and is backed by per-thread private scratch.
 * This one bounds the template path's per-position charset rows, which live in
 * a shared __global buffer (gpu_template.cl mask_charsets), so widening it
 * costs global memory and divmod iterations rather than registers. Literals
 * occupy a row each here -- that is why "?a?a?a?a?a?a@gmail10p.com" needs 19.
 * Must move together with: gpu_opencl.c MAX_MASK_POS + MASK_POS_CAP,
 * gpu_metal.m METAL_MASK_POS_CAP, gpujob_metal.m prepend_chars/append_chars,
 * and the 32u clamps in gpu_template.cl + metal_template.metal. */
#define GPU_MASK_SLOT_CAP       32      /* template-path positions per side */
#define GPU_MASK_LIT_BYTES_CAP  224     /* literal bytes per side */
#define GPU_MASK_DESC_BYTES_CAP 320     /* run-descriptor stream cap per side */
#define MASK_LITERAL -1
#define MASK_MAX_CLASSES 16
struct MaskClass { unsigned char chars[256]; int count; };
struct MaskPos   { int classid; unsigned char literal; };
extern struct MaskClass MaskClasses[MASK_MAX_CLASSES];
extern struct MaskPos MaskPattern[MAX_MASK_POS];
extern int MaskLen;
extern int MaskPrepend;
extern unsigned long long MaskTotal;
extern unsigned long long MaskChunkSize;
extern struct MaskPos MaskAppendPattern[MAX_MASK_POS];
extern int MaskAppendLen;
extern unsigned long long MaskAppendTotal;
extern struct MaskPos MaskPrependPattern[MAX_MASK_POS];
extern int MaskPrependLen;
extern unsigned long long MaskPrependTotal;
extern unsigned char gpu_kern_a_prep_desc[GPU_MASK_DESC_BYTES_CAP];
extern unsigned char gpu_kern_a_app_desc [GPU_MASK_DESC_BYTES_CAP];
extern int gpu_kern_a_prep_dlen;
extern int gpu_kern_a_app_dlen;
extern int gpu_kern_a_mask_eligible;
void mask_init_classes(void);
int parse_mask_into(const char *s, struct MaskPos *pattern, int *len_out, unsigned long long *total_out);
/* Expand a mask pattern at a given index into buf. Returns pattern length. */
static inline int mask_expand_into(unsigned long long index,
                                   const struct MaskPos *pattern, int patlen,
                                   char *buf) {
    int i;
    for (i = patlen - 1; i >= 0; i--) {
        if (pattern[i].classid == MASK_LITERAL)
            buf[i] = pattern[i].literal;
        else {
            struct MaskClass *mc = &MaskClasses[pattern[i].classid];
            buf[i] = mc->chars[index % mc->count];
            index /= mc->count;
        }
    }
    return patlen;
}
int gpu_pack_mask_descriptor(const struct MaskPos *pattern, int plen, unsigned char *desc_out, int desc_cap);
void gpu_mask_count_vars_lit_bytes(const struct MaskPos *pattern, int plen, int *out_vars, int *out_lit_bytes);
extern struct job *FreeHead, **FreeTail;
extern struct job *WorkHead, **WorkTail;
extern lock *FreeWaiting, *WorkWaiting, *TestVecWaiting, *Stats;
/* Read buffers: a ring of NREADBUF, each READBUFSZ bytes of Readbuf with its
 * own RINDEXSIZE stretch of Readindex.  ReadBufL[i] counts the jobs still
 * using buffer i; the reader reuses a buffer only when it is zero.  Four, not
 * three: reading, indexing and hashing each hold one, and hashing usually
 * holds a second while the last jobs of the previous buffer drain. */
#define NREADBUF 4
#define READBUFSZ (MAXCHUNK / 2)
extern lock *ReadBufL[NREADBUF], *HashWaiting;
static inline lock *readbuf_lock(const char *buf) {
  return ReadBufL[(size_t)(buf - Readbuf) / READBUFSZ];
}
extern lock *ServerState;
extern atomic_ullong Tothash;
extern atomic_ullong Totfound;
extern unsigned long long Totrules;
extern atomic_ullong Totrules_gpu;
/* Phase 1.4: BF adaptive chunk-sizing servo state.
 *
 * Per-device dispatch feedback. After every BF chunk dispatch's clFinish,
 * gpu/gpujob_opencl.c writes the wall_us and exact candidate count of the
 * chunk to these atomics (one slot per active GPU device). The BF chunk
 * producer in the main thread reads them before each new chunk to refresh
 * its rate-EMA estimate; chunks_produced ramps target_seconds 1s -> 5s
 * over the first 10 chunks. Both arrays are sized BF_MAX_GPU_SLOTS (=64,
 * matches MAX_GPU_SLOTS local to gpujob_opencl.c). atomic_store_explicit
 * with memory_order_relaxed is sufficient — producer doesn't need
 * happens-before, just a recent enough sample.
 *
 * rate_ema is single-producer (main thread BF activation loop only), so
 * a plain uint64 is fine; chunks_produced likewise. Reset to initial
 * values at the start of each BF op so the servo restarts per op rather
 * than carrying state across ops with potentially different rates. */
#define BF_MAX_GPU_SLOTS 64
extern _Atomic uint64_t bf_dev_wall_us[BF_MAX_GPU_SLOTS];
extern _Atomic uint64_t bf_dev_chunk_total[BF_MAX_GPU_SLOTS];
extern _Atomic int bf_dev_first_dispatch_done[BF_MAX_GPU_SLOTS];
extern int BfMaskGpuUsable;
extern uint64_t bf_rate_ema;
extern uint32_t bf_chunks_produced;
extern _Atomic int bf_first_feedback_seen;
extern unsigned long long EstimatedTotalLines;
extern volatile unsigned long long AutoCountTotalLines;
extern volatile unsigned long long ProjectedTotalLines;
extern volatile int AutoCountDone;
extern int AutoCountRequested;
extern char **LineCountArgv;
extern int LineCountArgc;
extern char LineCountCachePath[4096];
extern int BruteForceMode;
extern unsigned long long BruteForceTotal;
extern atomic_ullong *Totalfound[JOB_DONE], *RuleCnt;
extern atomic_ullong *UserTotalfound[USERDEF_MAX];
/* ---- Phase B per-op accessor -------------------------------------------
 * All access to this per-op array goes through this reference function.
 * The body is a plain redirect today (storage is still one shared range);
 * Phase C flips built-in vs user-defined to genuinely separate storage by
 * changing THIS FUNCTION ONLY, not the call sites. */
/* Phase C: user-defined types now live in SEPARATE storage, indexed by user
 * id (0..USERDEF_MAX-1), not by synthetic op. Built-in and user state can no
 * longer alias each other even if the op ranges were somehow to overlap --
 * they are different objects. This routing is the ONLY place that knows the
 * split; all 2,417 call sites go through the macro. */
static inline atomic_ullong **totalfound_ref(unsigned op) {
    return op_is_user(op) ? &UserTotalfound[op_user_id(op)] : &Totalfound[op];
}
#define TOTALFOUND(op) (*totalfound_ref(op))
extern volatile int MDXpause;
extern volatile int MDXpaused_count;
extern atomic_ullong MDXlowest_line;
extern volatile char *MDXlowest_file;
extern atomic_ullong CurfileBytesTotal;
extern atomic_ullong CurfileBytesRead;
#ifdef GPU_ENABLED
void sig_gpu_exit(int sig);
#endif
#ifndef _WIN32
void sig_pause(int sig);
#endif
#ifndef _WIN32
void sig_resume(int sig);
#endif
#ifndef _WIN32
#else
void win_pause_init(void);
#endif
int compact_insert(void);
void gen_salts();
void init_default_salts(void);
int user_canon(const char *in, int inlen, char *out);
extern char Ucanon[2 * MAXLINE + 16];

int user_decode(const char *key, char *dst);
#ifdef MDX_BIT32
#ifndef NOTINTEL
#define MDXALIGN __attribute__((force_align_arg_pointer))
#endif
#endif
#ifdef MDX_BIT32
#ifndef NOTINTEL
#else
#define MDXALIGN
#endif
#endif
#ifdef MDX_BIT32
#else
#define MDXALIGN
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
/* MIN_CHUNKS: producer-iteration count required before persisting. Each
 * iteration of adaptive_bf_chunk_size increments bf_chunks_produced;
 * after the bootstrap wait (chunk 0) the EMA(alpha=0.7) sees per-device
 * feedback on every subsequent call. By chunk 5 the EMA has integrated
 * 4-5 real samples (each weighted ~0.7^k), which is enough convergence
 * to be a useful next-run seed. Lower thresholds (e.g. 2) would persist
 * a still-warming-up value; higher (e.g. 20) miss short BF runs entirely
 * (10-digit MD5 BF on a 1080 = 6 chunks total). 5 is the practical floor. */
#define BF_SIDECAR_MIN_CHUNKS    5u
#endif
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
#define BF_SIDECAR_RATE_MIN      ((uint64_t)100000000ull)        /* 100 MH/s */
#endif
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
#define BF_SIDECAR_RATE_MAX      ((uint64_t)100000000000ull)     /* 100 GH/s */
#endif
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
int bf_sidecar_path(int dev_idx, int op, char *out, size_t cap);
#endif
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
int bf_sidecar_read_rate(const char *path, uint64_t *out_rate);
#endif
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
int bf_sidecar_write_rate(const char *path, uint64_t rate);
#endif
#endif
#ifdef GPU_ENABLED
#if defined(GPU_BF_CHUNK_PRODUCER)
int adaptive_bf_chunk_size(uint64_t mask_total, uint64_t chunk_cursor, int num_devs, int op, uint64_t *out_chunk_total, uint32_t *out_num_words, uint32_t *out_mask_size_per_word, uint32_t *out_inner_iter);
#endif
#endif
MDXALIGN void procjob(void *dummy);

#endif /* MDXINT_H */
