/*
 * $Log: mdxfind.h,v $
 * Revision 1.23  2026/05/11 03:48:16  dlr
 * BF Phase 1.9 A1: struct job +unsigned char bf_fast_eligible after bf_inner_iter (~line 127). Host BF chunk producer populates; procjob short-circuit copies to jobg.
 *
 * Revision 1.22  2026/05/10 21:20:25  dlr
 * BF Phase 1.8: struct job +unsigned int bf_inner_iter at 117-123, adjacent to bf_offset_per_word/bf_num_masks. Main-thread BF chunk producer populates; procjob short-circuit copies to jobg.
 *
 * Revision 1.21  2026/05/10 14:03:58  dlr
 * BF Tranche 3 plumbing: add bf_offset_per_word + bf_num_masks (uint) fields to struct job. Together with existing MaskIndex (carries bf_mask_start) and MaskCount (carries chunk_total), procjob short-circuit translates these to jobg fields at chokepoint entry. Default 0 = not a BF chunk.
 *
 * Revision 1.20  2026/05/10 05:11:16  dlr
 * BF Tranche 1 plumbing: define JOBFLAG_BF_CHUNK (0x80) for the chunk-as-job migration. Set on jobg slots produced by main thread BF chunk producer; procjob short-circuits at chokepoint entry and submits to gpujob queue. Inert until Tranche 3 sets the flag.
 *
 * Revision 1.19  2026/05/04 14:32:59  dlr
 * Add extern decls for tsfprintf() + tsfprintf_pin_summary() — startup-phase diagnostic instrumentation. Implementation in mdxfind.c above malloc_pinned. Used by gpu_opencl.c (forward-declared inline there since it doesn't include mdxfind.h) and gpujob_opencl.c (which does). Always-on with a single static lock for stderr serialization.
 *
 * Revision 1.18  2026/05/03 21:46:54  dlr
 * Widening + cache-poison fix.
 *
 * Widening (forward-looking; Shooter's hashmob.net.found.v7 is 3.16B lines, fits in 32-bit but no headroom):
 * - struct job::startline,numline: unsigned int -> unsigned long long; reordered (8-byte first, 4-byte after); prefix[] aligned(16) to keep buffer alignment after the +8 bytes
 * - struct Linehints::curline,numline: widened + reordered
 * - MDXlowest_line: 'volatile unsigned int = 0xFFFFFFFF' -> 'atomic_ullong = ULLONG_MAX'. C11 atomics auto-handle platform conditionalization (x86-64/ARM64: plain mov/ldr/str; i386: LOCK CMPXCHG8B; ARMv7: LDREXD/STREXD). All 4 read/write sites use atomic_load_explicit/atomic_store_explicit with memory_order_relaxed. lowest_line/restart locals widened to match. Sentinels migrated to ULLONG_MAX.
 * - cacheline()::nextline static + curline local widened (the actual chunk-position counter)
 * - procjob locals (curline,numline,sl) widened
 * - read-loop locals widened
 * - numline = UINT_MAX -> ULLONG_MAX (lineswanted stays 32-bit — per-job target)
 * - struct jobg fully reorganized (8-byte first / 4-byte / 1-byte slot_kind / 4-byte arrays / 2-byte arrays / aligned union last); 'unsigned int line_num' deleted (vestigial — never compared, only assigned)
 * - gpujob_get_free* dropped 'startline' parameter (the parameter was already discarded with '(void)startline;')
 * - 7 dead 'g->line_num = job->startline;' lines deleted
 * - 5 callers updated to drop startline arg
 *
 * Cache-poison fix (root cause of Shooter's wrong-ETA bug 2026-05-03: cache held 12M lines for hashmob.net.found.v7 vs 3.16B actual):
 * - Fix 1: linecount_file() gains 'int *complete' out-param. Default *complete=0; set to 1 only via natural fall-through after read-loop runs to EOF. linecount_thread() now skips cache store when complete=0 — partial counts from HashWaiting early-bail no longer poison future sessions.
 * - Fix 2a: New atomic_ullong CurfileBytesRead/CurfileBytesTotal globals. cacheline() accumulates readlen into CurfileBytesRead per chunk. Main read loop sets CurfileBytesTotal from sb.st_size at file open; resets CurfileBytesRead.
 * - Fix 2b: After 'Fileline += Linecount' in main read loop, if Fileline > AutoCountTotalLines (cache was wrong), project new estimate from byte-position ratio (Fileline * total_bytes / read_bytes) and CAS-loop-bump AutoCountTotalLines monotonically. ReportStats's ETA self-corrects without code change to ReportStats.
 * - Fix 2c: New linecount_cache_finalize(filepath, size, mtime, actual_lines) function. Called after gzclose per wordlist (skipping stdin + Fileline==0). Opens own SQLite connection, INSERT OR REPLACE under WAL+busy_timeout, PRAGMA wal_checkpoint(TRUNCATE), close — overwrites poisoned partial entry with the authoritative count discovered at EOF.
 * - comfort line: 'line N' position now appears in the percentage branch too (not just the no-estimate fallback).
 *
 * Revision 1.17  2026/04/27 21:53:26  dlr
 * GPU rule engine Phase 0 classifier (project_gpu_rule_engine_design.md rev 3, §6). Adds gpu_rule_safe_phase0() — single-stage op-based predicate accepting only Tier-1 ops {l, u, r, :, space, tab} in the post-packrules bytecode — and classify_rules() — partitions a rule array into full / gpu / cpu lists preserving original order. struct rule_lists declared in mdxfind.h alongside applyrule. Verified against synthetic mixed input (7 GPU + 5 CPU partition correct). Empirical note: HashMob.{100,1k,5k,100k}.rule classify as 0% GPU-eligible at Phase 0 — they all use ops beyond l/u/r — so Phase 0 validation will need a synthetic test fixture, not HashMob, to exercise the GPU path.
 *
 * Revision 1.16  2026/04/22 22:02:53  dlr
 * struct rule_workspace and extern applyrule in header
 *
 * Revision 1.15  2026/04/22 18:23:53  dlr
 * Add struct rule_workspace for heap-allocated applyrule buffers
 *
 * Revision 1.14  2026/04/14 04:46:11  dlr
 * GPU brute-force: timing probe, per-chunk dispatch, uint64 mask_start, base-offset decomposition, immediate hit processing, MD5SHA256SHA256 (e996)
 *
 * Revision 1.13  2026/04/05 03:55:52  dlr
 * Include emmintrin.h under NOTINTEL guard, MAXCHUNK 50MB for Apple Silicon (not embedded ARM)
 *
 * Revision 1.12  2026/04/04 18:53:45  dlr
 * Per-algorithm dispatch with linehints: rate-based lineswanted from bench_rates.h, EMA feedback in ReportStats, per-algorithm curline tracking, GPU lineswanted=UINT_MAX for ordering, Lowline from min(curline), struct job reorder + fileno + JOBFLAG_GPU, FAM enum moved to gpujob.h
 *
 * Revision 1.11  2026/03/25 23:11:05  dlr
 * Move Hashchain struct to header
 *
 * Revision 1.10  2026/03/23 02:51:54  dlr
 * Replace -n digit hack with mask-based hybrid attack: -n "?l?d" append, -N prepend, ?[0-9a-f] custom classes
 *
 * Revision 1.9  2025/08/24 22:08:56  dlr
 * changes for atomic
 *
 * Revision 1.8  2025/08/23 22:26:25  dlr
 * Move to new outbuf
 *
 * Revision 1.7  2020/03/11 02:49:29  dlr
 * SSSE modifications complete.  About to start on fastrule
 *
 * Revision 1.6  2017/10/19 03:38:44  dlr
 * Add rule counter
 *
 * Revision 1.5  2017/08/25 05:09:54  dlr
 * minor change for ARM6
 *
 * Revision 1.4  2017/08/25 04:16:03  dlr
 * Porting for ARM/POWERPC.  Fix SQL5
 *
 * Revision 1.3  2017/06/30 13:35:32  dlr
 * fix for ARM
 *
 * Revision 1.2  2017/06/30 13:23:13  dlr
 * Added SVAL
 *
 * Revision 1.1  2017/06/29 14:09:29  dlr
 * Initial revision
 *
 *
 */
#if ARM > 6
#include <arm_neon.h>
#endif
#ifndef NOTINTEL
#include <emmintrin.h>
#endif

#define MAXLINE (40*1024)
struct job {
    /* 8-byte fields grouped first */
    struct job *next;
    char *readbuf,*outbuf,*pass;
    unsigned int *found;
    struct LineInfo *readindex;
    char *filename;
    int *doneprint;
    unsigned long long Numbers;
    unsigned long long MaskIndex;
    unsigned long long MaskCount;
    unsigned long long startline, numline;   /* widened from unsigned int — wordlist line positions can exceed 4.29B */
    /* 4-byte fields */
    int op,len,clen,flags;
    int Ruleindex,digits,outlen,fileno;
    /* BF chunk-as-job (Tranche 3, 2026-05-09): when JOBFLAG_BF_CHUNK is set,
     * MaskIndex carries bf_mask_start (chunk's base cursor in the global
     * keyspace), MaskCount carries chunk_total (candidates in this chunk),
     * and these two fields carry the per-word stride / mask range. The
     * procjob short-circuit translates these into jobg fields at chokepoint
     * entry. Default 0 = not a BF chunk. */
    unsigned int bf_offset_per_word;
    unsigned int bf_num_masks;
    /* BF Phase 1.8 (2026-05-10): kernel inner iteration count for this chunk.
     * 0 or 1 = today's behavior (bit-identical). Cap=16. Set by
     * adaptive_bf_chunk_size servo; procjob short-circuit copies into
     * jobg.bf_inner_iter. Unsalted BF only; servo forces 1 on salted ops. */
    unsigned int bf_inner_iter;
    /* Phase 1.9 Tranche A1 (2026-05-10): when 1, the chunk producer has
     * pre-qualified this BF chunk for the BF-fast MD5 template kernel
     * (gpu_md5_bf.cl). Conditions: op==JOB_MD5, Numrules<=1, unsalted,
     * append-only mask (npre==0, napp in [1,8]), and env
     * MDXFIND_GPU_FAST_DISABLE is unset. Procjob short-circuit copies
     * this into jobg.bf_fast_eligible. Default 0 = slow template path.
     * Wider eligibility (multi-rule, prepend, salted) is intentionally
     * out of A1 scope; A2-A4 do not widen this gate. */
    unsigned int bf_fast_eligible;
    /* Buffers — explicit 16-byte alignment for SIMD; widening startline+numline
     * pushed prefix off natural 16-alignment, so we mark it explicitly. */
    char prefix[MAXLINE] __attribute__((aligned(16)));
    char line[MAXLINE+MAXLINE];
};
#define JOBFLAG_PRINT 1
#define JOBFLAG_HEX 2
#define JOBFLAG_NUMBERS 4
#define JOBFLAG_IP 8
#define JOBFLAG_PREPEND 16
#define JOBFLAG_GPU 32
#define JOBFLAG_BRUTEFORCE 64
#define JOBFLAG_BF_CHUNK 128  /* BF chunk-as-job: produced by main thread for procjob short-circuit fill */

union HashU {
    unsigned char h[256];
    uint32_t i[64];
    unsigned long long v[32];
#ifndef NOTINTEL
    __m128i x[16];
#endif
#if ARM > 6
    uint32x4_t x[16];
#endif
#ifdef POWERPC
    vector unsigned int x[16];
#endif
};

struct Hashchain {
    struct Hashchain *next;
    unsigned short int flags, len;
    unsigned char hash[1];
};

#ifdef ARM
union sse_value {
#if ARM > 6
    uint32x4_t sse;
#else
    uint64_t sse,sse1;
#endif
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif
#ifdef POWERPC
union sse_value {
   vector unsigned int sse;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#ifdef SPARC
union sse_value {
   uint64_t sse,sse1;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#ifndef NOTINTEL
union sse_value {
    __m128i sse;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

/* Rule processing workspace — defined here before any includes,
 * so it's available for both mdxfind and procrule builds. */
#define RULE_WORKSPACE_SIZE ((40*1024) + 16)
struct rule_workspace {
    char Memory[RULE_WORKSPACE_SIZE];
    char Base64buf[RULE_WORKSPACE_SIZE];
};

extern int applyrule(char *line, char *pass, int len, char *rule, struct rule_workspace *ws);
extern int packrules(char *line);

/* GPU rule engine — three-list partition (Phase 0 design memo).
 * `full` is the original (caller-owned) array; gpu and cpu are owned
 * pointer arrays into the same packed-rule strings — no copy. */
struct rule_lists {
    char **full;     int nfull;
    char **gpu;      int ngpu;
    char **cpu;      int ncpu;
};

extern int classify_rules(char **rules, int nrules, struct rule_lists *out);
extern void rule_lists_free(struct rule_lists *rl);

/* Startup-phase diagnostic instrumentation (Shooter 12-GPU rig).
 * Always-on. Prefixes a "[T+ S.SSSs] " stamp to the formatted message,
 * thread-safe via a private lock so the prefix and message stay on a
 * single line under multi-thread emission. tsfprintf_pin_summary()
 * emits a consolidated end-of-init pin tally (per-(reason, size_class)
 * malloc_pinned() outcomes). MDXFIND_PIN_TRACE=1 in the environment
 * additionally enables per-attempt tsfprintf() lines. Defined in
 * mdxfind.c above malloc_pinned. */
#include <stdio.h>
extern void tsfprintf(FILE *fp, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
extern void tsfprintf_pin_summary(void);

#define BCRYPT_HASHSIZE 64
#define MAXVECSIZE 2000000  /* Maximum test vector size */

#define MAXTHREADS 8

#define LDAP_MAX_UTF8_LEN  ( sizeof(wchar_t) * 3/2 )
#define FLOOR_LOG2(x) (31 - __builtin_clz((x) | 1))
static inline int log2i(uint64_t n) {
#define S(k) if (n >= ((uint64_t)1 << k)) { i += k; n >>= k; }
    int i = -(n == 0); S(32); S(16); S(8); S(4); S(2); S(1); return i;
#undef S
}

/* MAXCHUNK sets the maximum amount of memory used for each chunk.
   As I write this, typical hard drive speeds are 100 Mbytes/sec, so
   100M represents about 1 seconds of data.  Increase as appropriate.
*/
#if defined(ARM) && !defined(MACOSX)
/* INPUTCHUNK - maximum number of hashes to process at once from stdin */
#define INPUTCHUNK (100000)
#define MAXCHUNK (5*1024*1024)
#else
/* INPUTCHUNK - maximum number of hashes to process at once from stdin */
#define INPUTCHUNK (10000000)
#define MAXCHUNK (50*1024*1024)
#endif

#define MAXLINEPERCHUNK (MAXCHUNK/2/8)

#define MAXLJOB (32)



