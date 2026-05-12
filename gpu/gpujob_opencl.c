/*
 * gpujob_cuda.c — GPU worker thread for mdxfind CUDA acceleration
 *
 * Same architecture as gpujob.m (Metal), but uses gpu_opencl.h API.
 * Compiled only on Linux with OPENCL_GPU defined.
 */
#include <time.h>
/*
 */

#if defined(OPENCL_GPU)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef bswap_32
#define bswap_32(x) __builtin_bswap32(x)
#endif
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <stdatomic.h>
#ifndef NOTINTEL
#include <emmintrin.h>
#endif
#include <iconv.h>
#include "mdxfind.h"
#include "job_types.h"
#include "gpujob.h"
#include "gpu_opencl.h"
#include "yarn.h"
#include <Judy.h>

extern int Printall, Maxiter;
extern volatile int MDXpause, MDXpaused_count;
extern int hybrid_check(const unsigned char *, int, int *, unsigned short **);
extern void md5crypt_b64encode(const unsigned char *, char *);
extern void prfound(struct job *, char *);
extern int checkhashbb(union HashU *, int, char *, struct job *);
extern void mymd5(char *, int, unsigned char *);
extern void mysha1(char *, int, unsigned char *);
extern void mysha256(char *, int, unsigned char *);
struct saltentry;
extern int build_hashsalt_snapshot(struct saltentry *, char *, Pvoid_t, char *, int);
extern Pvoid_t *Typehashsalt;
extern char Typedone[];
extern void **Typesalt;
extern void **Typeuser;
extern void *OverflowHash;
extern Pvoid_t JudyJ[];
extern char phpitoa64[];
extern atomic_ullong *Totalfound[];
extern atomic_ullong *RuleCnt;
extern atomic_ullong Tothash;
extern atomic_ullong Totfound;
/* Phase 1.4 BF adaptive servo telemetry: per-device feedback channel.
 * Producer (mdxfind.c BF activation loop) reads these atomics to compute
 * the next chunk size; this file writes them after each clFinish. Storage
 * is sized to MAX_GPU_SLOTS in mdxfind.c (matches our local constant).
 * Atomic stores are memory_order_relaxed — sloppy is fine; producer just
 * wants any recent rate sample. */
extern _Atomic uint64_t bf_dev_wall_us[];
extern _Atomic uint64_t bf_dev_chunk_total[];
/* Phase 1.7c (2026-05-09): per-slot first-dispatch flag — see mdxfind.c
 * for the rationale. The very first BF dispatch on a slot includes the
 * per-op template JIT (~12 s); skip the atomic_store on that first
 * dispatch so the rate-EMA bootstraps from clean (post-JIT) samples. */
extern _Atomic int bf_dev_first_dispatch_done[];
extern int checkhash(union HashU *curin, int len, int x, struct job *job);
extern int checkhashkey(union HashU *curin, int len, char *key, struct job *job);
extern int checkhashsalt(union HashU *curin, int len, char *salt, int saltlen, int x, struct job *job);
extern int build_salt_snapshot(void *snap, char *pool,
                void *judy, char *keybuf, int printall);
extern int *Typesaltcnt;
extern long long *Typesaltbytes;

/* SHA256CRYPT carrier (2026-05-08) -- hit-replay helper.
 *
 * Encode a 32-byte SHA-256 digest into the 43-char SHA256CRYPT base64
 * format (glibc crypt-sha256 layout). Ports the per-byte permutation
 * table from mdxfind.c:12753-12980 (cryptlen=32 path of the shared
 * SHA-CRYPT serialization). Output 43 chars + NUL.
 *
 * Layout: 10 groups of 3 bytes -> 4 chars (40 chars), final 2 bytes ->
 * 3 chars (43 total). The 11 byte-tuples are:
 *   {0,10,20}, {21,1,11}, {12,22,2}, {3,13,23}, {24,4,14},
 *   {15,25,5}, {6,16,26}, {27,7,17}, {18,28,8}, {9,19,29},
 *   {31,30}                  (final 2-byte tuple -> 3 chars)
 *
 * The mapping is the 32-byte specific permutation from glibc's
 * sha256-crypt.c. Each (a, b, c) tuple builds val = (a << 16) |
 * (b << 8) | c, then emits 4 phpitoa64 chars LSB-first.
 *
 * NOT named sha256_b64encode to avoid clash with any future sha256-
 * specific encoder; the "crypt" suffix marks it as the glibc-specific
 * permutation. */
static inline void sha256crypt_b64encode(const unsigned char *in, char *out) {
    /* The 10 three-byte tuples that produce 4 b64 chars each. */
    static const unsigned char order3[10][3] = {
        { 0, 10, 20}, {21,  1, 11}, {12, 22,  2}, { 3, 13, 23},
        {24,  4, 14}, {15, 25,  5}, { 6, 16, 26}, {27,  7, 17},
        {18, 28,  8}, { 9, 19, 29}
    };
    int j = 0;
    for (int i = 0; i < 10; i++) {
        unsigned int v = ((unsigned int)in[order3[i][0]] << 16)
                       | ((unsigned int)in[order3[i][1]] << 8)
                       |  (unsigned int)in[order3[i][2]];
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f];
    }
    /* Final 2-byte tuple -> 3 b64 chars. (high << 8) | low, in[31] = high,
     * in[30] = low (matches mdxfind.c:12964 cas = (h[31] << 8) | h[30]). */
    {
        unsigned int v = ((unsigned int)in[31] << 8) | (unsigned int)in[30];
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f];
    }
    out[j] = 0;  /* j == 43 */
}

/* SHA512CRYPT carrier (2026-05-08) -- hit-replay helper.
 *
 * Encode a 64-byte SHA-512 digest into the 86-char SHA512CRYPT base64
 * format (glibc crypt-sha512 layout). Ports the per-byte permutation
 * table from mdxfind.c:12361-12780 (cryptlen=64 path of the shared
 * SHA-CRYPT serialization). Output 86 chars + NUL.
 *
 * Layout: 21 groups of 3 bytes -> 4 chars (84 chars), final 1 byte ->
 * 2 chars (86 total). The 21 three-byte tuples + final 1-byte tuple
 * match the compact-table loader at mdxfind.c:37036-37041:
 *   {0,21,42}, {22,43,1}, {44,2,23}, {3,24,45}, {25,46,4},
 *   {47,5,26}, {6,27,48}, {28,49,7}, {50,8,29}, {9,30,51},
 *   {31,52,10}, {53,11,32}, {12,33,54}, {34,55,13}, {56,14,35},
 *   {15,36,57}, {37,58,16}, {59,17,38}, {18,39,60}, {40,61,19},
 *   {62,20,41}            (21 three-byte tuples)
 *   {63}                  (final 1-byte tuple -> 2 chars)
 *
 * The mapping is the 64-byte specific permutation from glibc's
 * sha512-crypt.c. Each (a, b, c) tuple builds val = (a << 16) |
 * (b << 8) | c, then emits 4 phpitoa64 chars LSB-first. Final byte
 * emits 2 phpitoa64 chars LSB-first (no third char -- only the bottom
 * 2 b64 digits are written; the high 6 bits of the 8-bit value
 * occupy 0..3 of the second char).
 *
 * NOT named sha512_b64encode to avoid clash with any future sha512-
 * specific encoder; the "crypt" suffix marks it as the glibc-specific
 * permutation. Phase 3 of the Unix-crypt ladder. */
static inline void sha512crypt_b64encode(const unsigned char *in, char *out) {
    /* The 21 three-byte tuples that produce 4 b64 chars each.
     * Mirrors mdxfind.c:37036-37041 sha512_perm[21][3] verbatim. */
    static const unsigned char order3[21][3] = {
        { 0, 21, 42}, {22, 43,  1}, {44,  2, 23}, { 3, 24, 45},
        {25, 46,  4}, {47,  5, 26}, { 6, 27, 48}, {28, 49,  7},
        {50,  8, 29}, { 9, 30, 51}, {31, 52, 10}, {53, 11, 32},
        {12, 33, 54}, {34, 55, 13}, {56, 14, 35}, {15, 36, 57},
        {37, 58, 16}, {59, 17, 38}, {18, 39, 60}, {40, 61, 19},
        {62, 20, 41}
    };
    int j = 0;
    for (int i = 0; i < 21; i++) {
        unsigned int v = ((unsigned int)in[order3[i][0]] << 16)
                       | ((unsigned int)in[order3[i][1]] << 8)
                       |  (unsigned int)in[order3[i][2]];
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f];
    }
    /* Final 1-byte tuple -> 2 b64 chars. Just in[63] (matches mdxfind.c
     * cryptlen==64 path's final cas = curin.h[63] at line 12775). */
    {
        unsigned int v = (unsigned int)in[63];
        out[j++] = phpitoa64[v & 0x3f]; v >>= 6;
        out[j++] = phpitoa64[v & 0x3f];
    }
    out[j] = 0;  /* j == 86 */
}

/* GPU rule engine globals (set by classify_rules block in mdxfind.c main()).
 * Used by rules_engine dispatch path to recover word->rule->hash triples. */
extern unsigned char *gpu_rule_program;   /* packed bytecodes, NUL-separated */
extern uint32_t      *gpu_rule_offsets;   /* byte offset of each GPU rule in program */
extern int           *gpu_rule_origin;    /* gpu_rule_origin[i] = original Rules[] index */
extern int            gpu_rule_count;     /* number of GPU-eligible rules uploaded */
extern char          *Rules;              /* length-prefixed rule buffer (mdxfind.c) */
extern unsigned int   Numrules;           /* total rules loaded */
extern atomic_ullong  Totrules_gpu;       /* GPU-side simulated (word×rule) candidate count */

/* Set by gpujob_overflow_preload_all() — when 1, the worker thread
 * skips its lazy load_overflow() call (mdxfind.c already pushed the
 * data at session start). Diagnosed via GAP DEBUG: lazy upload from
 * the worker contended for CPU cycles with procjob threads, inflating
 * tiny work to seconds (3s on 16-core fpga, ~30s suspected on 64-core
 * mmt). Pre-loading at session start happens before any procjob thread
 * launches, so the system is quiet and the upload completes in µs. */
int gpu_overflow_preloaded = 0;

/* Return the correct salt/user Judy for a given op.
 * HMAC key=$salt types whose TypeOpts has TYPEOPT_NEEDUSER store their
 * "salt" (the HMAC key) in Typeuser. The Streebog HMAC KSALT ops
 * (JOB_HMAC_STREEBOG256_KSALT, JOB_HMAC_STREEBOG512_KSALT) are
 * EXCEPTIONS — their TypeOpts has TYPEOPT_NEEDSALT (not TYPEOPT_-
 * NEEDUSER), so the salt-data lives in Typesalt[op] (loaded from -F
 * via the structured-hash loader). The CPU paths at mdxfind.c:30822
 * (KSALT 256) + mdxfind.c:30937 (KSALT 512) build their salt snapshot
 * from Typesalt[job->op]. Falling through to the default Typesalt arm
 * matches CPU behavior. */
static void *gpu_salt_judy(int op) {
    switch (op) {
    case JOB_HMAC_MD5:
    case JOB_HMAC_SHA1:
    case JOB_HMAC_SHA224:
    case JOB_HMAC_SHA256:
    case JOB_HMAC_SHA384:
    case JOB_HMAC_SHA512:
    case JOB_HMAC_RMD160:
    case JOB_HMAC_RMD320:
        return Typeuser ? Typeuser[op] : NULL;
    default:
        return Typesalt ? Typesalt[op] : NULL;
    }
}

/* Decode phpass iteration count from salt[3] */
static int phpbb3_iter_count(const char *salt, int saltlen) {
    static const char itoa64[] = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    if (saltlen < 4) return 2048;
    char c = salt[3];
    for (int k = 0; k < 64; k++)
        if (itoa64[k] == c) return 1 << k;
    return 2048;
}

/* Mask table helpers — gpu_mask_desc has [sizes][tables[256]] packed by gpu_opencl_set_mask.
 * gpu_mask_sizes[] has the per-position character counts for hit reconstruction. */
extern uint8_t gpu_mask_desc[];
extern uint8_t gpu_mask_sizes[];
extern int gpu_mask_n_prepend, gpu_mask_n_append;

/* Reconstruct mask characters from mask_idx into buf.
 * pos_offset = starting position index (0 for prepend, n_prepend for append).
 * Uses table-based format: sizes in gpu_mask_sizes, char tables in gpu_mask_desc. */
static int mask_decode(uint64_t mask_idx, int pos_offset, int npos, char *buf) {
    uint64_t idx = mask_idx;
    int n_total = gpu_mask_n_prepend + gpu_mask_n_append;
    for (int i = npos - 1; i >= 0; i--) {
        int pos = pos_offset + i;
        int sz = gpu_mask_sizes[pos];
        int ci = (int)(idx % sz);
        idx /= sz;
        buf[i] = (char)gpu_mask_desc[n_total + pos * 256 + ci];
    }
    return npos;
}

extern uint64_t gpu_mask_total;

#define PV_DEC(pv) { unsigned long _old = *(pv); \
  while (_old > 0) { \
    if (__sync_bool_compare_and_swap((pv), _old, _old - 1)) break; \
    _old = *(pv); } }

struct saltentry {
    char *salt;
    unsigned long *PV;
    int saltlen;
    char *hashsalt;
    int hashlen;
};

/* ---- GPU work queue ---- */
struct jobg *GPUWorkHead, **GPUWorkTail;
lock *GPUWorkWaiting;
/* Two disjoint slot free-lists. A slot's `slot_kind` field determines
 * which list it returns to; cross-pool reuse is structurally impossible. */
struct jobg *GPULegacyFreeHead, **GPULegacyFreeTail;
struct jobg *GPURulesFreeHead,  **GPURulesFreeTail;
lock *GPULegacyFreeWaiting, *GPURulesFreeWaiting;
static int _gpujob_ready = 0;
static int _gpujob_count = 1;       /* device count (stats-array index bound) */
static int _n_gpujob_workers = 0;   /* actually-spawned worker thread count
                                     * (== _gpujob_count - disabled_devices).
                                     * Used by shutdown sentinel loop so we
                                     * send the right number of poison pills. */
static int _num_legacy_jobg = 0;
static int _num_rules_jobg = 0;

/* Per-GPU dispatch counters for diagnostics. Three separate accumulators
 * because the three dispatch paths use different units:
 *   _gpu_words        — rules-engine path, += g->packed_count (input words).
 *                       Each kernel-launch hashes (packed_count × rule_count) candidates.
 *   _gpu_legacy_ent   — legacy packed path, += chunk_end - word_start (entries).
 *                       Each entry is a CPU-pre-expanded candidate; one hash apiece.
 *   _gpu_slab_hashes  — batch-slab path, += g->count (hashes).
 *                       Each is one hash, possibly iterated × Maxiter.
 * The end-of-run print computes total hashes correctly per path, instead
 * of the prior single counter that mixed units and forced a wrong
 * × gpu_rule_count multiplier across the board. */
#define MAX_GPU_SLOTS 64
static uint64_t _gpu_words[MAX_GPU_SLOTS];        /* rules-engine words (raw word count, no fan-out) */
static uint64_t _gpu_rules_hashes[MAX_GPU_SLOTS]; /* rules-engine candidates evaluated = words × rules × masks × salts × iters; mirrors slab _gpu_slab_hashes semantics. Per-device companion to Tothash for hash_Gh/s rendering. Added 2026-05-09 to fix the salt-axis under-count surfaced by e31 MD5SALT real-workload run on fpga (1.6M reported vs ~8T actual). */
static uint64_t _gpu_legacy_ent[MAX_GPU_SLOTS];   /* legacy packed entries */
static uint64_t _gpu_slab_hashes[MAX_GPU_SLOTS];  /* slab path hash count */
static uint64_t _gpu_hits[MAX_GPU_SLOTS];
static uint64_t _gpu_batches[MAX_GPU_SLOTS];
/* Per-device end-of-run timing/throughput accounting. */
static uint64_t _gpu_busy_us[MAX_GPU_SLOTS];   /* accumulated wall-clock-busy us */
static uint64_t _gpu_first_us[MAX_GPU_SLOTS];  /* wall-clock at first dispatch */
static uint64_t _gpu_last_us[MAX_GPU_SLOTS];   /* wall-clock at last completion */
static uint64_t _gpu_h2d_bytes[MAX_GPU_SLOTS]; /* accumulated H2D bytes */
static char     _gpu_lnksta_start[MAX_GPU_SLOTS][192];
static char     _gpu_lnksta_end[MAX_GPU_SLOTS][192];

/* MDXFIND_DISPATCH_TRACE=1: log every kernel-dispatch site to stderr (or a
 * file via MDXFIND_DISPATCH_TRACE_FILE). When the env var is unset, the
 * branch is one bool check + skip — no measurable overhead. */
static int  gpu_dispatch_trace_enabled = -1;   /* -1 = uninit, 0/1 = resolved */
static FILE *gpu_dispatch_trace_fp = NULL;
static lock *gpu_dispatch_trace_lock = NULL;
static uint64_t gpu_dispatch_trace_t0 = 0;     /* first-dispatch-anywhere timestamp */
static uint64_t gpu_now_us(void);              /* fwd-decl, defined below */

/* MDXFIND_PIPE_TRACE=1: emit one [pipe] line per phase per jobg slot,
 * keyed by slot address. Lets us correlate fill/queue/dispatch/return
 * times across procjob and gpujob threads. No fflush. */
static int gpu_pipe_trace_enabled = -1;        /* -1 = uninit, 0/1 = resolved */
static void gpu_pipe_trace_init(void) {
    if (gpu_pipe_trace_enabled >= 0) return;
    const char *e = getenv("MDXFIND_PIPE_TRACE");
    gpu_pipe_trace_enabled = (e && *e && *e != '0') ? 1 : 0;
}

static void gpu_dispatch_trace_init(void) {
    if (gpu_dispatch_trace_enabled >= 0) return;
    const char *e = getenv("MDXFIND_DISPATCH_TRACE");
    gpu_dispatch_trace_enabled = (e && *e && *e != '0') ? 1 : 0;
    if (gpu_dispatch_trace_enabled) {
        const char *path = getenv("MDXFIND_DISPATCH_TRACE_FILE");
        if (path && *path) {
            gpu_dispatch_trace_fp = fopen(path, "a");
            if (!gpu_dispatch_trace_fp) {
                fprintf(stderr, "WARN: MDXFIND_DISPATCH_TRACE_FILE='%s' open failed; falling back to stderr\n", path);
                gpu_dispatch_trace_fp = stderr;
            }
        } else {
            gpu_dispatch_trace_fp = stderr;
        }
        gpu_dispatch_trace_lock = new_lock(0);
        gpu_dispatch_trace_t0 = gpu_now_us();
    }
}

/* Per-dispatch wall-clock timestamp relative to the first dispatch on any
 * device. Allows post-run analysis of overlap/serialization across GPUs:
 * if dispatches across N devices have non-overlapping (start_us,
 * start_us+wall_us) windows, work is serialized; if overlap heavily, work
 * runs in parallel. Without this, the [disp] line only shows per-dispatch
 * duration, which can't distinguish "12 GPUs each idle 60% because work
 * is serialized" from "12 GPUs each idle 60% because of host bottlenecks
 * elsewhere". */
static inline uint64_t gpu_dispatch_trace_relative(uint64_t now) {
    return (now > gpu_dispatch_trace_t0) ? (now - gpu_dispatch_trace_t0) : 0;
}

/* Cheap fingerprint over packed_buf — XOR-fold of every uint32 in the
 * packed bytes (rounded down to multiple of 4). Deterministic across
 * devices for the same input, so traces from different GPUs in the same
 * run can be correlated. */
static uint32_t gpu_packed_xor(const char *buf, uint32_t size) {
    if (!buf || size == 0) return 0;
    uint32_t x = 0;
    uint32_t n = size & ~3u;
    const uint32_t *p = (const uint32_t *)buf;
    for (uint32_t i = 0; i < n / 4; i++) x ^= p[i];
    /* Tail bytes — fold in to keep checksum stable for non-aligned sizes. */
    for (uint32_t i = n; i < size; i++)
        x ^= ((uint32_t)(unsigned char)buf[i]) << ((i - n) * 8);
    return x;
}

static const char *gpu_op_name(int op) {
    /* Lightweight — only labels the most common GPU paths. Anything else
     * shows as op=NN. The full Opnames[] table is in mdxfind.c which we
     * don't pull into this TU just for tracing. */
    switch (op) {
    case JOB_MD5:                return "MD5";
    case JOB_SHA1:               return "SHA1";
    case JOB_SHA256:             return "SHA256";
    case JOB_SHA512:             return "SHA512";
    case JOB_NTLM:               return "NTLM";
    case JOB_NTLMH:              return "NTLMH";
    case JOB_MD4:                return "MD4";
    case JOB_MD5SALT:            return "MD5SALT";
    case JOB_MD5PASSSALT:        return "MD5PASSSALT";
    case JOB_MD5SALTPASS:        return "MD5SALTPASS";
    default:                     return "?";
    }
}

static uint64_t gpu_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
}

/* Inline helper to capture LnkSta via lspci. Linux only; silent on macOS
 * or where lspci is unavailable / restricted. Empty BDF -> skip. */
static void gpu_capture_lnksta(int slot, const char *bdf, char *out, size_t out_sz) {
    if (out_sz == 0) return;
    out[0] = 0;
#ifndef _WIN32
    if (!bdf || !*bdf) return;
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "lspci -vv -s %s 2>/dev/null | grep LnkSta", bdf);
    FILE *p = popen(cmd, "r");
    if (!p) return;
    if (fgets(out, (int)out_sz, p)) {
        size_t n = strlen(out);
        while (n > 0 && (out[n-1] == '\n' || out[n-1] == '\r' || out[n-1] == ' ' || out[n-1] == '\t')) {
            out[--n] = 0;
        }
        /* Trim leading whitespace */
        char *s = out;
        while (*s == ' ' || *s == '\t') s++;
        if (s != out) memmove(out, s, strlen(s) + 1);
    }
    pclose(p);
    (void)slot;
#else
    (void)slot; (void)bdf;
#endif
}
static int _max_salt_count = 0;
static int _max_salt_bytes = 0;
static int overflow_loaded = 0;
static int _gpu_batch_max = GPUBATCH_RULE_MAX; /* min across all devices */

/* GPU dispatch scheduling: NONE. Each gpujob worker thread is pinned to
 * its device and pulls from the single shared work queue (GPUWorkWaiting
 * / GPUWorkHead) with one possess()/wait_for/twist() round-trip. Slot
 * acquisition (gpujob_get_free*) blocks only on the per-pool free-list
 * waiter (GPULegacy/RulesFreeWaiting). Submission (gpujob_submit) takes
 * one lock to enqueue. No priority by filename/line, no waiter scanning,
 * no per-thread wake locks. The previous priority scheduler held a
 * single global lock (GPUSchedLock) twice per dispatch and ran a linear
 * scan of the waiter list inside it; with N=12 GPU threads × 64 procjob
 * cores all serializing through it, multi-GPU performance degraded to
 * worse-than-single-GPU on rules+packed workloads. Removed for
 * additivity. (Output ordering by filename/line is no longer guaranteed.)
 */


/* Reconstruct 13-char DES crypt string from GPU pre-FP output (l, r) and salt.
 * Applies FP permutation bit-by-bit, then base64-encodes to crypt format. */
static void des_reconstruct(uint32_t gl, uint32_t gr, const char *salt, char *out) {
    static const unsigned char DES_FP[64] = {
        40, 8,48,16,56,24,64,32,39, 7,47,15,55,23,63,31,
        38, 6,46,14,54,22,62,30,37, 5,45,13,53,21,61,29,
        36, 4,44,12,52,20,60,28,35, 3,43,11,51,19,59,27,
        34, 2,42,10,50,18,58,26,33, 1,41, 9,49,17,57,25
    };
    uint32_t il = gl, ir = gr;
    /* Apply FP to (il, ir) -> (r0, r1) */
    uint32_t r0 = 0, r1 = 0;
    for (int i = 0; i < 32; i++) {
        int b = DES_FP[i] - 1;
        uint32_t src = (b < 32) ? il : ir;
        if (src & (1u << (31 - (b % 32)))) r0 |= (1u << (31 - i));
    }
    for (int i = 0; i < 32; i++) {
        int b = DES_FP[32 + i] - 1;
        uint32_t src = (b < 32) ? il : ir;
        if (src & (1u << (31 - (b % 32)))) r1 |= (1u << (31 - i));
    }
    /* Encode: salt + 11 base64 chars */
    out[0] = salt[0]; out[1] = salt[1];
    uint32_t v;
    v = r0 >> 8;
    out[2] = phpitoa64[(v>>18)&0x3f]; out[3] = phpitoa64[(v>>12)&0x3f];
    out[4] = phpitoa64[(v>>6)&0x3f];  out[5] = phpitoa64[v&0x3f];
    v = (r0 << 16) | ((r1 >> 16) & 0xffff);
    out[6] = phpitoa64[(v>>18)&0x3f]; out[7] = phpitoa64[(v>>12)&0x3f];
    out[8] = phpitoa64[(v>>6)&0x3f];  out[9] = phpitoa64[v&0x3f];
    v = r1 << 2;
    out[10] = phpitoa64[(v>>12)&0x3f]; out[11] = phpitoa64[(v>>6)&0x3f];
    out[12] = phpitoa64[v&0x3f];
    out[13] = 0;
}

/* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): bf_encode_23 helper.
 * Encodes 23 raw bytes (bcrypt's 24-byte digest, last byte is always
 * truncated by BF_encode) into 31 base64 characters using bcrypt's
 * custom alphabet "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
 * 0123456789". Mirrors the inline b64 encoding logic from the slab
 * BCRYPT hit-replay arm at gpujob_opencl.c lines 2264-2287 (rev 1.x;
 * to be deleted in this same commit). The output buffer must be at
 * least 32 bytes (31 chars + NUL). raw must point to 23 readable
 * bytes (sourced from curin.i[0..5] reinterpreted as a 24-byte LE
 * stream; first 23 are encoded, byte 24 is the slab kernel's zero
 * padding which BF_encode discards). */
static void bf_encode_23(const unsigned char *raw, char *out) {
    static const char bf_itoa64[] =
        "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const unsigned char *sp = raw;
    char *dp = out;
    int bytes_left = 23;
    while (bytes_left > 0) {
        unsigned int c1 = *sp++;
        *dp++ = bf_itoa64[c1 >> 2];
        c1 = (c1 & 0x03) << 4;
        if (--bytes_left <= 0) { *dp++ = bf_itoa64[c1]; break; }
        unsigned int c2 = *sp++;
        c1 |= c2 >> 4;
        *dp++ = bf_itoa64[c1];
        c1 = (c2 & 0x0f) << 2;
        if (--bytes_left <= 0) { *dp++ = bf_itoa64[c1]; break; }
        unsigned int c3 = *sp++;
        c1 |= c3 >> 6;
        *dp++ = bf_itoa64[c1];
        *dp++ = bf_itoa64[c3 & 0x3f];
        bytes_left--;
    }
    *dp = 0;
}

#define GPU_MAX_RETURN 32768
#define OUTBUFSIZE (1024 * 1024)

/* gpu_pack_salts_op: caller-supplies the op so type-specific filters can
 * be applied (e.g., DESCRYPT skips saltlen != 2 to fall extended-DES
 * `_CCCCSSSS` 9-char salts back to CPU via bsd_crypt_des). For callers
 * that don't need op-specific filtering, gpu_pack_salts() (below) calls
 * this with op == -1 (no filter). */
static int gpu_pack_salts_op(struct saltentry *saltsnap, int nsalts,
                             char *salts_packed, uint32_t *soff, uint16_t *slen,
                             int *pack_map, int use_hashsalt, int op) {
    int packed = 0;
    uint32_t gsp = 0;
    for (int i = 0; i < nsalts; i++) {
        if (!Printall && *saltsnap[i].PV == 0) continue;
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): skip extended-
         * DES salts (`_CCCCSSSS` 9-char setting). Standard DES uses 2-char
         * phpitoa64 salts ONLY; the GPU kernel (gpu_descrypt_core.cl)
         * reads salt_bytes[0..1] directly. Extended-DES salts pass
         * structurally distinct password lengths AND a different bit-
         * count (12-bit standard vs 24-bit extended saltbits), so they
         * MUST CPU-fallback through bsd_crypt_des (mdxfind.c:23636-23722
         * + crypt-des.c:634-672). Q2 user decision 2026-05-08: salt-pack
         * filter at this site cleanest. */
        if (op == JOB_DESCRYPT && saltsnap[i].saltlen != 2) continue;
        /* For types that precompute MD5(salt), pack the hex hash instead of raw salt */
        char *s = (use_hashsalt && saltsnap[i].hashsalt) ? saltsnap[i].hashsalt : saltsnap[i].salt;
        int sl = (use_hashsalt && saltsnap[i].hashsalt) ? 32 : saltsnap[i].saltlen;
        soff[packed] = gsp;
        slen[packed] = sl;
        pack_map[packed] = i;
        memcpy(salts_packed + gsp, s, sl);
        gsp += sl;
        packed++;
    }
    return packed;
}

/* Backward-compat shim for callers that don't need per-op salt filtering
 * (kept to minimize edit surface in caller sites that don't yet thread
 * `op` through). For DESCRYPT and other op-aware filters, callers must
 * use gpu_pack_salts_op directly. */
static int gpu_pack_salts(struct saltentry *saltsnap, int nsalts,
                          char *salts_packed, uint32_t *soff, uint16_t *slen,
                          int *pack_map, int use_hashsalt) {
    return gpu_pack_salts_op(saltsnap, nsalts, salts_packed,
                             soff, slen, pack_map, use_hashsalt, -1);
}

static void load_overflow(int dev_idx) {
    if (!OverflowHash) return;
    int ocnt = 0;
    size_t obytes = 0;
    Word_t okey = 0;
    Word_t *OPV;
    OPV = (Word_t *)JudyLFirst(OverflowHash, &okey, NULL);
    while (OPV) {
        struct Hashchain *chain = (struct Hashchain *)(*OPV);
        /* Memo B Phase B5 sub-batch 7 (2026-05-05): zero-pad sub-128-bit
         * entries to 16 bytes in storage so the GPU 4xuint32 probe in
         * probe_compact_idx (gpu/gpu_common.cl:641,659) doesn't spill into
         * the next entry's bytes. Mirrors the HashDataBuf zero-pad fix in
         * mdxfind.c:36400-36412 (rev 1.399). For chain->len >= 16 the
         * stored size equals the real len; for sub-128-bit (e.g. MYSQL3
         * with len==8) the stored size is 16 bytes. olengths[i] keeps the
         * real len so CPU memcmp paths are unaffected. */
        while (chain) {
            int pad_len = chain->len < 16 ? 16 : chain->len;
            ocnt++;
            obytes += pad_len;
            chain = chain->next;
        }
        OPV = (Word_t *)JudyLNext(OverflowHash, &okey, NULL);
    }
    if (ocnt > 0) {
        uint64_t *okeys = (uint64_t *)malloc_lock(ocnt * sizeof(uint64_t),"load_overflow");
        unsigned char *ohashes = (unsigned char *)malloc_lock(obytes + 16,"load_overflow");
        memset(ohashes, 0, obytes + 16);
        uint32_t *ooffsets = (uint32_t *)malloc_lock(ocnt * sizeof(uint32_t),"load_overflow");
        uint16_t *olengths = (uint16_t *)malloc_lock(ocnt * sizeof(uint16_t),"load_overflow");
        int oi = 0;
        uint32_t opos = 0;
        okey = 0;
        OPV = (Word_t *)JudyLFirst(OverflowHash, &okey, NULL);
        while (OPV) {
            struct Hashchain *chain = (struct Hashchain *)(*OPV);
            while (chain) {
                int pad_len = chain->len < 16 ? 16 : chain->len;
                okeys[oi] = okey;
                ooffsets[oi] = opos;
                olengths[oi] = chain->len;  /* real length for CPU memcmp */
                memcpy(ohashes + opos, &okey, 8);
                if (chain->len > 8)
                    memcpy(ohashes + opos + 8, chain->hash, chain->len - 8);
                /* Bytes [chain->len .. pad_len) stay zero (memset above). */
                opos += pad_len;
                oi++;
                chain = chain->next;
            }
            OPV = (Word_t *)JudyLNext(OverflowHash, &okey, NULL);
        }
        gpu_opencl_set_overflow(dev_idx, okeys, ohashes, ooffsets, olengths, ocnt);
        free(okeys); free(ohashes); free(ooffsets); free(olengths);
    }
}

/* Pre-load overflow data to every GPU device at session start, before
 * procjob threads launch. Eliminates the worker-thread CPU-contention
 * gap diagnosed via GAP DEBUG (3s on fpga, longer on bigger-core hosts).
 * Idempotent — sets gpu_overflow_preloaded so the worker's lazy path
 * short-circuits. mdxfind.c calls this from main() right after the
 * GPU rule engine setup block. */
/* Memo C: per-device overflow upload thread. load_overflow operates entirely
 * on gpu_devs[di]'s context + queue, no shared writes across devices. */
static void preload_overflow_thread(void *payload) {
    int di = *(int *)payload;
    load_overflow(di);
}
void gpujob_overflow_preload_all(void) {
    if (!OverflowHash) return;
    if (gpu_overflow_preloaded) return;
    int n_dev = gpu_opencl_num_devices();
    if (n_dev <= 0) return;
    if (n_dev == 1) {
        load_overflow(0);
    } else {
        thread *t[64];
        int args[64];
        int nl = (n_dev > 64) ? 64 : n_dev;
        for (int i = 0; i < nl; i++) {
            args[i] = i;
            t[i] = launch(preload_overflow_thread, &args[i]);
        }
        for (int i = 0; i < nl; i++) join(t[i]);
    }
    gpu_overflow_preloaded = 1;
}

void gpujob(void *arg) {
    int my_slot = (int)(intptr_t)arg;
    union HashU curin;
    struct job synthetic_job;
    char *outbuf = (char *)malloc_lock(OUTBUFSIZE+1024,"gpujob");
    uint64_t hashcnt = 0, found = 0;
    char tsalt[4096];

    memset(&synthetic_job, 0, sizeof(synthetic_job));
    synthetic_job.outbuf = outbuf;

    struct saltentry *saltsnap = (struct saltentry *)malloc_lock(_max_salt_count * sizeof(struct saltentry),"saltentry");
    char *saltpool = (char *)malloc_lock(_max_salt_bytes + 16,"saltpool");
    /* Buffer must hold either raw salts OR 32-byte hex hashes per salt */
    size_t sp_size = _max_salt_bytes + 4096;
    if ((size_t)_max_salt_count * 32 + 4096 > sp_size)
        sp_size = (size_t)_max_salt_count * 32 + 4096;
    char *salts_packed = (char *)malloc_lock(sp_size,"salts_packed");
    char desbuf[16];  /* DES crypt reconstructed string (13 chars + NUL) */
    uint32_t *soff = (uint32_t *)malloc_lock(_max_salt_count * sizeof(uint32_t),"gpujob");
    uint16_t *slen = (uint16_t *)malloc_lock(_max_salt_count * sizeof(uint16_t),"gpujob");
    int *pack_map = (int *)malloc_lock(_max_salt_count * sizeof(int),"gpujob");
    int nsalts = 0;
    int nsalts_packed = 0;
    int salt_refresh = 0;
    int salt_hits_pending = 0; /* unused for now, reserved for adaptive refresh */
    int current_op = -1;
    int batch_count = 0;
    int my_overflow_loaded = 0;

    while (1) {
        possess(GPUWorkWaiting);
        wait_for(GPUWorkWaiting, NOT_TO_BE, 0);
        struct jobg *g = GPUWorkHead;
        GPUWorkHead = g->next;
        if (GPUWorkHead == NULL)
            GPUWorkTail = &GPUWorkHead;
        twist(GPUWorkWaiting, BY, -1);

        g->t_dispatched = gpu_now_us();
        if (gpu_pipe_trace_enabled == 1 && g->op != 2000 && g->t_added != 0) {
            fprintf(stderr, "[pipe] g=%p queue_us=%llu dev=%d\n",
                    (void *)g,
                    (unsigned long long)(g->t_dispatched - g->t_added),
                    my_slot);
        }

        if (g->op == 2000) {
            /* Sentinel — return to its origin pool by slot_kind. */
            g->next = NULL;
            if (g->slot_kind == JOBG_KIND_RULES) {
                possess(GPURulesFreeWaiting);
                if (GPURulesFreeTail) {
                    *GPURulesFreeTail = g;
                    GPURulesFreeTail = &(g->next);
                } else {
                    GPURulesFreeHead = g;
                    GPURulesFreeTail = &(g->next);
                }
                twist(GPURulesFreeWaiting, BY, +1);
            } else {
                possess(GPULegacyFreeWaiting);
                if (GPULegacyFreeTail) {
                    *GPULegacyFreeTail = g;
                    GPULegacyFreeTail = &(g->next);
                } else {
                    GPULegacyFreeHead = g;
                    GPULegacyFreeTail = &(g->next);
                }
                twist(GPULegacyFreeWaiting, BY, +1);
            }
            break;
        }

        /* Each device loads overflow to its own GPU memory.
         * Skipped when mdxfind.c pre-loaded at session start (the
         * gpu_overflow_preloaded flag) — that's the fast path; this
         * lazy fallback handles unexpected configurations where the
         * pre-load couldn't run. */
        if (!my_overflow_loaded && OverflowHash) {
            my_overflow_loaded = 1;
            if (!gpu_overflow_preloaded) {
                load_overflow(my_slot);
            }
        }

        int op_cat = gpu_op_category(g->op);

        /* B6 salt-axis (2026-05-06; §11 row 20): widen the salt-snapshot
         * trigger to also fire for GPU_CAT_MASK ops that have a Typesalt[]
         * (i.e., JOB_MD5SALT + JOB_MD5SALTPASS — moved to GPU_CAT_MASK in
         * §11 row 25 because they're now template-routed, not slab-routed).
         * gpu_salt_judy(g->op) returns NULL for unsalted ops; the existing
         * 32 GPU_CAT_MASK ops carry no Typesalt[] entry so this widening
         * is a no-op for them. The triple-condition matches what the
         * salt-aware template hit-replay block (§11 row 24) needs. */
        int needs_salt_snapshot =
            (op_cat == GPU_CAT_SALTED || op_cat == GPU_CAT_SALTPASS ||
             (op_cat == GPU_CAT_MASK && gpu_salt_judy(g->op) != NULL));

        /* Rebuild salt snapshot on op change, periodically, or after hits found */
        if (needs_salt_snapshot) {
            batch_count++;
            if (g->op != current_op ||
                (salt_refresh && op_cat == GPU_CAT_SALTPASS) ||
                (batch_count >= 10 && nsalts_packed > 0)) {
                salt_refresh = 0;
                if (g->op != current_op) {
                    current_op = g->op;
                    gpu_opencl_set_op(my_slot, g->op);
                }
                batch_count = 0;
                tsalt[0] = 0;
                { int use_hs = (g->op == JOB_MD5_MD5SALTMD5PASS ||
                                g->op == JOB_SHA1_MD5_MD5SALTMD5PASS ||
                                g->op == JOB_SHA1_MD5_MD5SALTMD5PASS_SALT ||
                                g->op == JOB_SHA1_MD5PEPPER_MD5SALTMD5PASS);
                  if (use_hs && Typehashsalt[g->op])
                    nsalts = build_hashsalt_snapshot(saltsnap, saltpool,
                                    Typehashsalt[g->op], tsalt, Printall);
                  else
                    nsalts = build_salt_snapshot(saltsnap, saltpool,
                                    gpu_salt_judy(g->op), tsalt, Printall);
                }
                if (nsalts > 0) {
                    int use_hs = (g->op == JOB_MD5_MD5SALTMD5PASS ||
                                  g->op == JOB_SHA1_MD5_MD5SALTMD5PASS ||
                                  g->op == JOB_SHA1_MD5_MD5SALTMD5PASS_SALT ||
                                  g->op == JOB_SHA1_MD5PEPPER_MD5SALTMD5PASS);
                    /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5):
                     * pass g->op so gpu_pack_salts_op can apply the
                     * saltlen != 2 filter for JOB_DESCRYPT (extended-DES
                     * 9-char salts CPU-fallback through bsd_crypt_des).
                     * For all other ops the op argument is ignored. */
                    nsalts_packed = gpu_pack_salts_op(saltsnap, nsalts,
                                                     salts_packed, soff, slen, pack_map,
                                                     use_hs, g->op);
                }
                else
                    nsalts_packed = 0;
                if (nsalts_packed > 0)
                    gpu_opencl_set_salts(my_slot, salts_packed, soff, slen, nsalts_packed);
                else
                    Typedone[g->op] = 1;
            }
        } else if (g->op != current_op) {
            current_op = g->op;
            gpu_opencl_set_op(my_slot, g->op);
        }

        int nhits = 0;
        uint32_t *hits = NULL;

        if (g->count == 0 && !g->packed) goto return_jobg;
        /* B6 salt-axis (2026-05-06; §11 row 21): same widening as the
         * earlier trigger. needs_salt_snapshot covers SALTED/SALTPASS
         * plus MASK-with-Typesalt — so this stale-protection branch
         * fires for JOB_MD5SALT/JOB_MD5SALTPASS too. */
        if (needs_salt_snapshot && nsalts_packed == 0) {
            /* Stale nsalts_packed — force a fresh rebuild before giving up.
             * Words in this batch were packed when salts were still active. */
            int nsalts = build_salt_snapshot(saltsnap, saltpool,
                            gpu_salt_judy(g->op), tsalt, Printall);
            if (nsalts > 0) {
                /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5):
                 * stale-rebuild path also threads g->op so the saltlen
                 * != 2 filter applies consistently. */
                nsalts_packed = gpu_pack_salts_op(saltsnap, nsalts,
                                                 salts_packed, soff, slen, pack_map,
                                                 0, g->op);
                if (nsalts_packed > 0)
                    gpu_opencl_set_salts(my_slot, salts_packed, soff, slen, nsalts_packed);
            }
            if (nsalts_packed == 0) {
                goto return_jobg;
            }
        }

        /* Mask mode: set op on first batch */
        if (op_cat == GPU_CAT_MASK && g->op != current_op) {
            current_op = g->op;
            gpu_opencl_set_op(my_slot, g->op);
        }

        synthetic_job.op = g->op;
        synthetic_job.flags = g->flags;
        synthetic_job.filename = g->filename;
        synthetic_job.doneprint = g->doneprint;
        synthetic_job.found = (unsigned int *)&found;
        synthetic_job.outlen = 0;

        /* Packed password dispatch — GPU rule path */
        if (g->packed && g->packed_count > 0) {

            /* ---------------------------------------------------------------
             * rules_engine dispatch: GPU does the (word x rule) Cartesian
             * product; host decodes hits via applyrule() replay.
             * Sub-commit B: code path is dead until sub-commit C sets
             * g->rules_engine = 1 on chokepoint slots.
             * --------------------------------------------------------------- */
            if (g->rules_engine) {
                /* Single dispatch — no chunking (kernel caps at GPU_PACKED_MAX_HITS). */
                if (my_slot < MAX_GPU_SLOTS) _gpu_batches[my_slot]++;
                gpu_dispatch_trace_init();
                uint64_t _disp_t0 = gpu_now_us();
                if (my_slot < MAX_GPU_SLOTS && _gpu_first_us[my_slot] == 0)
                    _gpu_first_us[my_slot] = _disp_t0;
                hits = gpu_opencl_dispatch_md5_rules(my_slot,
                    g->packed_buf, g->packed_pos,
                    g->word_offset, g->packed_count,
                    g->op, &nhits,
                    g->bf_chunk ? g->bf_mask_start : 0,
                    g->bf_chunk ? g->bf_offset_per_word : 0,
                    g->bf_chunk ? g->bf_num_masks : 0,
                    g->bf_chunk ? g->bf_inner_iter : 0,
                    /* Phase 1.9 A1 (2026-05-10): pass fast-path
                     * eligibility through to gpu_opencl.c, which
                     * swaps kern_template_phase0 ->
                     * kern_template_phase0_md5_bf when the flag is
                     * set, op==JOB_MD5, and the BF-fast kernel
                     * compile/lazy succeeded. */
                    (int)g->bf_fast_eligible);
                uint64_t _disp_t1 = gpu_now_us();
                if (my_slot < MAX_GPU_SLOTS) {
                    _gpu_words[my_slot] += g->packed_count;
                    if (nhits > 0) _gpu_hits[my_slot] += nhits;
                    _gpu_busy_us[my_slot]   += (_disp_t1 - _disp_t0);
                    _gpu_h2d_bytes[my_slot] += g->packed_pos;
                    _gpu_last_us[my_slot]    = _disp_t1;
                }
                /* Phase 1.4: BF adaptive servo telemetry. After clFinish in
                 * gpu_opencl_dispatch_md5_rules, _disp_t1 - _disp_t0 is the
                 * dispatch wall time, and packed_count*bf_num_masks is the
                 * exact candidate count this chunk processed. Producer reads
                 * these to update its rate-EMA and pick the next chunk_total.
                 *
                 * Phase 1.7c (2026-05-09): the FIRST dispatch on this slot
                 * includes per-op template JIT (gpu_template_resolve_kernel
                 * in gpu/gpu_opencl.c ~11019), which can be ~12 s on a cold
                 * kernel cache. That JIT-contaminated wall_us would poison
                 * the rate-EMA bootstrap. Mark the slot done on first call
                 * but skip the store; subsequent (clean) dispatches store
                 * normally. */
                if (g->bf_chunk && my_slot < MAX_GPU_SLOTS) {
                    if (!atomic_load_explicit(
                            &bf_dev_first_dispatch_done[my_slot],
                            memory_order_relaxed)) {
                        atomic_store_explicit(
                            &bf_dev_first_dispatch_done[my_slot], 1,
                            memory_order_relaxed);
                    } else {
                        /* BF Phase 1.8 (2026-05-10): each lane processes
                         * inner_iter mask values, so the actual chunk
                         * candidate count is packed_count * bf_num_masks *
                         * inner_iter. With inner_iter==0/1 this equals the
                         * pre-1.8 product (bit-identical). The servo's
                         * rate-EMA needs the true count to size subsequent
                         * chunks correctly under target_seconds. */
                        uint32_t _ii =
                            (g->bf_inner_iter == 0u) ? 1u : g->bf_inner_iter;
                        uint64_t _bf_chunk_cands =
                            (uint64_t)g->packed_count *
                            (uint64_t)g->bf_num_masks *
                            (uint64_t)_ii;
                        atomic_store_explicit(&bf_dev_chunk_total[my_slot],
                                              _bf_chunk_cands,
                                              memory_order_relaxed);
                        atomic_store_explicit(&bf_dev_wall_us[my_slot],
                                              (uint64_t)(_disp_t1 - _disp_t0),
                                              memory_order_relaxed);
                    }
                }
                if (gpu_dispatch_trace_enabled == 1) {
                    uint32_t xor_fp = gpu_packed_xor(g->packed_buf, g->packed_pos);
                    possess(gpu_dispatch_trace_lock);
                    fprintf(gpu_dispatch_trace_fp,
                            "[disp] dev=%d op=%s(%d) path=rules t_us=%llu packed_count=%u packed_pos=%u input_xor=0x%08x max_iter=%d num_masks=0 hits=%d wall_us=%llu bf_chunk=%u bf_mask_start=%llu bf_offset_per_word=%u bf_num_masks=%u bf_inner_iter=%u bf_fast=%u\n",
                            my_slot, gpu_op_name(g->op), g->op,
                            (unsigned long long)gpu_dispatch_trace_relative(_disp_t0),
                            g->packed_count, g->packed_pos, xor_fp,
                            Maxiter, nhits,
                            (unsigned long long)(_disp_t1 - _disp_t0),
                            (unsigned)g->bf_chunk,
                            (unsigned long long)g->bf_mask_start,
                            g->bf_offset_per_word,
                            g->bf_num_masks,
                            g->bf_inner_iter,
                            (unsigned)g->bf_fast_eligible);
                    fflush(gpu_dispatch_trace_fp);
                    release(gpu_dispatch_trace_lock);
                }
                /* Simulate per-(word,rule) candidate count for the
                 * "total rule-generated passwords tested" report. The
                 * kernel doesn't feed back per-lane counts (would
                 * serialize on an atomic), but per-dispatch we know
                 * exactly: packed_count words × gpu_rule_count rules
                 * = the Cartesian product the kernel just hashed. */
                /* B7.1/B7.2 mask multiplier: when multi-position append
                 * is active, each (word, rule) pair generates mask_size
                 * candidates on the GPU (mask_size = product of per-pos
                 * charset sizes = gpu_mask_total). Tothash / Totrules_gpu
                 * must reflect that or accounting under-counts the
                 * workload. mask_size == 1 when no mask is active —
                 * keeps the pre-B7 metric bit-identical. */
                /* Phase 1.5 (2026-05-10): widen to uint64 + drop the
                 * <=4G gate so masks above 2^32 (e.g., ?d^10 = 10G) are
                 * accounted for correctly. Tothash multiplier consumer
                 * downstream (line ~2290) already widened to uint64 with
                 * the BF-aware bf_num_masks substitution. */
                uint64_t b71_mask_size_acct = 1;
                if (gpu_mask_n_prepend >= 0 && gpu_mask_n_prepend <= 16
                    && gpu_mask_n_append >= 0 && gpu_mask_n_append <= 16
                    && (gpu_mask_n_prepend + gpu_mask_n_append) >= 1
                    && gpu_mask_total > 0) {
                    b71_mask_size_acct = gpu_mask_total;
                }
                /* gpu_rule_count includes the synthetic NUL no-op pass that
                 * the host always prepends to the rule program (mdxfind.c
                 * ~44644). The NUL is a convenience to make the kernel
                 * produce the implicit no-rule pass; it is not a real rule
                 * and is already accounted for by Totallines. Subtract 1
                 * so Totrules_gpu reflects only real rule applications. */
                atomic_fetch_add(&Totrules_gpu,
                    (unsigned long long)g->packed_count *
                    (unsigned long long)(gpu_rule_count - 1) *
                    (unsigned long long)b71_mask_size_acct);

                if (hits && nhits > 0) {
                    /* Precompute a table of rule-bytecode pointers the first time
                     * this thread enters the rules_engine path.  Each entry in
                     * the Rules buffer is: uint16_t length | bytecode[length].
                     * We index by orig_idx (0..Numrules-1). */
                    static __thread char  **_rule_ptr_cache  = NULL;
                    static __thread int     _rule_ptr_nrules = 0;
                    if (_rule_ptr_cache == NULL || _rule_ptr_nrules != (int)Numrules) {
                        free(_rule_ptr_cache);
                        _rule_ptr_cache = (char **)malloc((size_t)Numrules * sizeof(char *));
                        if (_rule_ptr_cache) {
                            char *rp = Rules;
                            for (int ri = 0; ri < (int)Numrules; ri++) {
                                uint16_t rlen;
                                memcpy(&rlen, rp, sizeof(uint16_t));
                                rp += sizeof(uint16_t);
                                _rule_ptr_cache[ri] = rp;
                                rp += rlen;
                            }
                        }
                        _rule_ptr_nrules = (int)Numrules;
                    }

                    /* Thread-local applyrule scratch space. */
                    static __thread struct rule_workspace _ws;
                    static __thread char _tpass[MAXLINE + 64];

                    int stored = nhits;
                    if (stored > GPU_PACKED_MAX_HITS) stored = GPU_PACKED_MAX_HITS;
                    int hexlen = gpu_hash_words(g->op) * 8;

                    /* Memo B Phase B7.1/B7.2: multi-position append mask
                     * hit decode. Kernel packs combined_ridx = rule_idx *
                     * mask_size + mask_idx into entry[1]. Host divmod's
                     * to recover both axes, then decomposes mask_idx into
                     * per-position character indices via successive
                     * divmod (last position innermost — same as
                     * mask_decode helper above and the slab path) and
                     * appends each per-position character to
                     * synthetic_job.line for checkhash().
                     *
                     * mask_size derivation MUST match the kernel's: B7.2
                     * scope is (n_prepend == 0, n_append in [1, 8]); in
                     * other configurations mask_size == 1 and the divmod
                     * reduces to ridx_decoded = combined and mask_idx = 0.
                     * That keeps the no-mask path bit-identical to pre-B7.
                     *
                     * B6 salt-axis (2026-05-06; §11 row 23+24, §12 S1c):
                     * for salted ops the kernel packs
                     *   combined_ridx = ((rule_idx * mask_size) + mask_idx)
                     *                   * num_salts_per_page + salt_local
                     * and the dispatcher post-processes slot[1] to a
                     * unified format using `nsalts_packed` (the full salt
                     * snapshot count) as the salt-axis divisor:
                     *   combined_ridx_global = ((rule_idx * mask_size) +
                     *                          mask_idx) * nsalts_packed +
                     *                          salt_idx_global.
                     * The host divmod chain here recovers (salt_idx_global,
                     * mask_idx, rule_idx) when is_salted_op is true.
                     * pack_map[salt_idx_global] -> snap_idx -> saltsnap[snap_idx]
                     * yields the salt bytes for checkhashsalt(). */
                    /* Phase 1.5 (2026-05-10): widen to uint64 + drop 4G
                     * gate. b71_mask_size is the divmod base for recovering
                     * (mask_idx_local, ridx) from combined_ridx. With the
                     * kernel emitting LOCAL mask_idx, this MUST equal the
                     * kernel-side mask_size — which is bf_num_masks for BF
                     * chunks (per-word range) and gpu_mask_total otherwise.
                     * See OCLParams.num_salts pack at gpu_opencl.c:~10420. */
                    uint64_t b71_mask_size = 1;
                    int b71_mask_active =
                        (gpu_mask_n_prepend >= 0 && gpu_mask_n_prepend <= 16
                         && gpu_mask_n_append >= 0 && gpu_mask_n_append <= 16
                         && (gpu_mask_n_prepend + gpu_mask_n_append) >= 1
                         && gpu_mask_total > 0);
                    if (b71_mask_active) {
                        b71_mask_size = (g->bf_chunk && g->bf_num_masks > 0u)
                                      ? (uint64_t)g->bf_num_masks
                                      : gpu_mask_total;
                    }

                    /* B6 salt-axis: derive is_salted_op + nsalts_for_decode
                     * for the unified-format combined_ridx decompose. For
                     * unsalted ops (32 GPU_CAT_MASK template-routed types
                     * pre-B6), this branch sets nsalts_for_decode=1 and
                     * the divmod reduces to combined / 1 == combined,
                     * combined % 1 == 0 — bit-identical to pre-B6 layout. */
                    /* B6.1 SHA1 fan-out (2026-05-06): JOB_SHA1SALTPASS
                     * joins the salted-op set for hit-replay. Same
                     * three-axis decompose (salt_local, mask_idx, ridx)
                     * via combined_ridx; SHA1SALTPASS routes through
                     * checkhashsalt at all iter values (no iter==1
                     * special case like MD5SALT — see CPU path at
                     * mdxfind.c:14404 — checkhashsalt is the only entry).
                     * Routing handled in the if (is_salted_op) branch
                     * below with the existing else { checkhashsalt(...) }
                     * arm catching all non-MD5SALT salted ops.
                     * B6.2 SHA256 fan-out (2026-05-06): JOB_SHA256SALTPASS
                     * joins. Same three-axis decompose; checkhashsalt at
                     * mdxfind.c:27633 is the only entry (no iter==1
                     * special case).
                     * B6.3 SHA224 fan-out (2026-05-06): JOB_SHA224SALTPASS
                     * joins. Same three-axis decompose; checkhashsalt is
                     * the only entry (no iter==1 special case). */
                    /* B6.4 MD5PASSSALT fan-out (2026-05-06): JOB_MD5PASSSALT
                     * joins. Same three-axis decompose; checkhashsalt at
                     * mdxfind.c:16655 is the only entry (no iter==1
                     * special case). Salt position (APPEND) is invisible
                     * to host hit-replay — the byte-order swap is purely
                     * inside template_finalize on the GPU.
                     * B6.5 SHA1PASSSALT fan-out (2026-05-06): JOB_SHA1PASSSALT
                     * joins. Same three-axis decompose; the SHA1PASSSALT
                     * hit-replay arm at line 1610 already exists (was used
                     * by the legacy slab path) and dispatches through
                     * checkhashsalt with 40-byte SHA1 hex digest length.
                     * APPEND vs PREPEND is invisible to host hit-replay
                     * — same as MD5PASSSALT. */
                    int is_salted_op =
                        (g->op == JOB_MD5SALT ||
                         g->op == JOB_MD5SALTPASS ||
                         g->op == JOB_SHA1SALTPASS ||
                         g->op == JOB_SHA256SALTPASS ||
                         g->op == JOB_SHA224SALTPASS ||
                         g->op == JOB_MD5PASSSALT ||
                         g->op == JOB_SHA1PASSSALT ||
                         /* B6.7 SHA256PASSSALT fan-out (2026-05-06):
                          * second SHA-family APPEND-shape salted variant.
                          * Hit-replay 64-byte SHA256 hex; APPEND vs
                          * PREPEND invisible to host replay. */
                         g->op == JOB_SHA256PASSSALT ||
                         /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS
                          * — first 64-bit-state salted variant on the
                          * unified template path. Hit-replay 128-byte
                          * SHA-512 hex; gpu_hash_words returns 16 (= 64
                          * digest bytes / 4 = 16 uint32 LE words),
                          * checkhashsalt called with hexlen=128. Both
                          * already wired in gpu_hash_words below — only
                          * the salted-decompose path needs this entry. */
                         g->op == JOB_SHA512SALTPASS ||
                         /* B6.10 SHA512PASSSALT fan-out (2026-05-06):
                          * SHA512PASSSALT — second 64-bit-state salted
                          * variant; APPEND-shape sibling. FINAL B6
                          * ladder step. Hit-replay 128-byte SHA-512
                          * hex; APPEND vs PREPEND invisible to host
                          * replay (the byte-order swap is purely inside
                          * template_finalize on the GPU). */
                         g->op == JOB_SHA512PASSSALT ||
                         /* B6.6 (2026-05-06): MD5SALT family variants */
                         g->op == JOB_MD5UCSALT ||
                         g->op == JOB_MD5revMD5SALT ||
                         g->op == JOB_MD5sub8_24SALT ||
                         /* B6.8 (2026-05-06): JOB_MD5_MD5SALTMD5PASS
                          * (e367) — fifth MD5SALT-family variant on
                          * the unified template path. Hit-replay 32-byte
                          * MD5 hex; the algo_mode=4 finalize re-uses the
                          * pre-computed salt-hex inside the kernel. */
                         g->op == JOB_MD5_MD5SALTMD5PASS ||
                         /* Family A (2026-05-07): JOB_HMAC_MD5 (e214) +
                          * JOB_HMAC_MD5_KPASS (e792) — sixth + seventh
                          * MD5SALT-template-kernel-sharing variants. Hit-
                          * replay 32-byte MD5 hex (4 hash words). KSALT
                          * dispatches through checkhashkey (no iter
                          * suffix in label, mdxfind.c:29386 CPU); KPASS
                          * dispatches through checkhashsalt with iter=0
                          * (mdxfind.c:29490 CPU). Both routed via the
                          * is_salted_op block at line ~1098 below. */
                         g->op == JOB_HMAC_MD5 ||
                         g->op == JOB_HMAC_MD5_KPASS ||
                         /* Family B (2026-05-07): JOB_HMAC_SHA1 (e215) +
                          * JOB_HMAC_SHA1_KPASS (e793) — share the
                          * SHA1SALTPASS GPU template kernel. Hit-replay
                          * 40-byte SHA1 hex (5 hash words). KSALT routed
                          * through checkhashkey (mdxfind.c:29301 CPU);
                          * KPASS through checkhashsalt with iter=0
                          * (mdxfind.c:29454 CPU). */
                         g->op == JOB_HMAC_SHA1 ||
                         g->op == JOB_HMAC_SHA1_KPASS ||
                         /* Family C (2026-05-07): JOB_HMAC_SHA224 (e216) +
                          * JOB_HMAC_SHA224_KPASS (e794) — share the
                          * SHA224SALTPASS GPU template kernel. Hit-replay
                          * 56-byte SHA224 hex (7 hash words). KSALT routed
                          * through checkhashkey (mdxfind.c:29326 case +
                          * HMAC_start label); KPASS through checkhashsalt
                          * with iter=0 (mdxfind.c:29479 case +
                          * HMAC_KPASS_start label). */
                         g->op == JOB_HMAC_SHA224 ||
                         g->op == JOB_HMAC_SHA224_KPASS ||
                         /* Family D (2026-05-08): JOB_HMAC_SHA256 (e217) +
                          * JOB_HMAC_SHA256_KPASS (e795) — share the
                          * SHA256SALTPASS GPU template kernel. Hit-replay
                          * 64-byte SHA256 hex (8 hash words). KSALT routed
                          * through checkhashkey (mdxfind.c:29581 case +
                          * HMAC_start label, hmac_len=64); KPASS through
                          * checkhashsalt with iter=0 (mdxfind.c:29734 case
                          * + HMAC_KPASS_start label, hmac_len=64). Final
                          * HMAC family in the ladder. */
                         g->op == JOB_HMAC_SHA256 ||
                         g->op == JOB_HMAC_SHA256_KPASS ||
                         /* Family E HMAC-SHA384 carrier (2026-05-08):
                          * JOB_HMAC_SHA384 (e543) + JOB_HMAC_SHA384_KPASS
                          * (e796) — share the SHA384SALTPASS-shaped carrier
                          * GPU template kernel. Hit-replay 96-byte SHA384
                          * hex (12 hash words). KSALT routed through
                          * checkhashkey (mdxfind.c:29369 case + HMAC_start
                          * label, hmac_len=96); KPASS through checkhashsalt
                          * with iter=0 (mdxfind.c:29522 case + HMAC_KPASS_-
                          * start label, hmac_len=96). */
                         g->op == JOB_HMAC_SHA384 ||
                         g->op == JOB_HMAC_SHA384_KPASS ||
                         /* Family F (2026-05-08): JOB_HMAC_SHA512 (e218) +
                          * JOB_HMAC_SHA512_KPASS (e797) — share the
                          * SHA512SALTPASS GPU template kernel. Hit-replay
                          * 128-byte SHA512 hex (16 hash words). KSALT
                          * routed through checkhashkey (mdxfind.c:29400
                          * case + HMAC_start label, hmac_len=128); KPASS
                          * through checkhashsalt with iter=0 (mdxfind.c:
                          * 29553 case + HMAC_KPASS_start label,
                          * hmac_len=128). */
                         g->op == JOB_HMAC_SHA512 ||
                         g->op == JOB_HMAC_SHA512_KPASS ||
                         /* Family G HMAC-RIPEMD-160 carrier (2026-05-08):
                          * JOB_HMAC_RMD160 (e211) + JOB_HMAC_RMD160_KPASS
                          * (e798) — share the RIPEMD160SALTPASS-shaped
                          * carrier GPU template kernel. Hit-replay 40-byte
                          * RMD160 hex (5 hash words; gpu_hash_words returns
                          * 5 for both ops). KSALT routed through
                          * checkhashkey (mdxfind.c:29391 case + HMAC_start
                          * label, hmac_len=40); KPASS through checkhashsalt
                          * with iter=0 (mdxfind.c:29584 case + HMAC_KPASS_-
                          * start label, hmac_len=40). */
                         g->op == JOB_HMAC_RMD160 ||
                         g->op == JOB_HMAC_RMD160_KPASS ||
                         /* Family H HMAC-RIPEMD-320 carrier (2026-05-08):
                          * JOB_HMAC_RMD320 (e213) + JOB_HMAC_RMD320_KPASS
                          * (e799) — share the RIPEMD320SALTPASS-shaped
                          * carrier GPU template kernel. Hit-replay 80-byte
                          * RMD320 hex (10 hash words; gpu_hash_words returns
                          * 10 for both ops, already wired in gpu_hash_words
                          * at line 2925 — no change needed). KSALT routed
                          * through checkhashkey (mdxfind.c:29428 case +
                          * HMAC_start label, hmac_len=80); KPASS through
                          * checkhashsalt with iter=0 (mdxfind.c:29616 case +
                          * HMAC_KPASS_start label, hmac_len=80). */
                         g->op == JOB_HMAC_RMD320 ||
                         g->op == JOB_HMAC_RMD320_KPASS ||
                         /* Family I HMAC-BLAKE2S carrier (2026-05-08):
                          * JOB_HMAC_BLAKE2S (e828) — single algo_mode (5);
                          * no KPASS sibling op exists. Routes via the hand-
                          * written Path A carrier GPU template kernel
                          * (gpu_hmac_blake2s_core.cl). Hit-replay 64-byte
                          * BLAKE2S hex (8 hash words; gpu_hash_words returns
                          * 8 for JOB_HMAC_BLAKE2S, already wired at line
                          * 2966 — no change needed). The op is KPASS-shape
                          * algorithmically (key=pass, msg=salt) but is named
                          * JOB_HMAC_BLAKE2S without a -KPASS suffix; routed
                          * through checkhashsalt with iter=0 (mdxfind.c:30391
                          * — emits "HMAC-BLAKE2S %s:%s:%s\n" with no xNN
                          * suffix). */
                         g->op == JOB_HMAC_BLAKE2S ||
                         /* Family J HMAC-STREEBOG-256 carrier (2026-05-08):
                          * JOB_HMAC_STREEBOG256_KSALT (e838) + JOB_HMAC_-
                          * STREEBOG256_KPASS (e837) — share the hand-written
                          * Path A carrier GPU template kernel
                          * (gpu_hmac_streebog256_core.cl) via algo_mode=5/6.
                          * Hit-replay 64-byte STREEBOG-256 hex (8 hash words;
                          * gpu_hash_words returns 8 for both ops, already
                          * wired in gpu_hash_words at line 3021). BOTH ops
                          * route through checkhashsalt with iter=0 (CPU
                          * semantics at mdxfind.c:30810 [KPASS] +
                          * mdxfind.c:30868 [KSALT] both use checkhashsalt;
                          * unlike Families A-H KSALT siblings which use
                          * checkhashkey, Streebog256 HMAC uses checkhashsalt
                          * for BOTH modes). */
                         g->op == JOB_HMAC_STREEBOG256_KSALT ||
                         g->op == JOB_HMAC_STREEBOG256_KPASS ||
                         /* Family K HMAC-STREEBOG-512 carrier (2026-05-08):
                          * JOB_HMAC_STREEBOG512_KSALT (e840) + JOB_HMAC_-
                          * STREEBOG512_KPASS (e839) - share the hand-written
                          * Path A carrier GPU template kernel
                          * (gpu_hmac_streebog512_core.cl) via algo_mode=5/6.
                          * Hit-replay 128-byte STREEBOG-512 hex (16 hash
                          * words; gpu_hash_words returns 16 for both ops,
                          * already wired in gpu_hash_words at line 3057).
                          * BOTH ops route through checkhashsalt with iter=0
                          * (CPU semantics at mdxfind.c:30963 [KPASS] +
                          * mdxfind.c:31020 [KSALT] both use checkhashsalt;
                          * mirrors Family J STREEBOG-256 pattern). Final
                          * HMAC family in the ladder. */
                         g->op == JOB_HMAC_STREEBOG512_KSALT ||
                         g->op == JOB_HMAC_STREEBOG512_KPASS ||
                         /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3
                          * (e455) — routes through the hand-written
                          * Path A salted-template kernel
                          * (gpu_phpbb3_core.cl). Hit-replay 32-byte
                          * MD5 hex (4 hash words; gpu_hash_words
                          * default 4). Hit-replay arm uses checkhashbb
                          * (NOT checkhashkey/checkhashsalt — PHPBB3 has
                          * its own bb-specific output format). The
                          * salt source is Typesalt (the 12-byte
                          * "$H$<cost><8>" prefix); gpu_salt_judy(JOB_-
                          * PHPBB3) resolves Typesalt[JOB_PHPBB3] via
                          * the default arm (TYPEOPT_NEEDSALT). max_-
                          * iter forced to 1 host-side; the algorithm's
                          * internal iter count (decoded from salt[3])
                          * runs INSIDE template_finalize. */
                         g->op == JOB_PHPBB3 ||
                         /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT
                          * (e511) -- routes through the hand-written
                          * Path A salted-template kernel
                          * (gpu_md5crypt_core.cl). Hit-replay 16-byte
                          * binary MD5 digest (4 hash words; gpu_hash_-
                          * words default 4) probed via hybrid_check;
                          * NEW hit-replay arm reconstructs "$1$<salt>$
                          * <22-char-phpitoa64>" via md5crypt_b64encode
                          * (mirrors existing slab arm at line 1723).
                          * The salt source is Typesalt (variable-length
                          * "$1$<salt>$" prefix); gpu_salt_judy(JOB_-
                          * MD5CRYPT) resolves Typesalt[JOB_MD5CRYPT] via
                          * the default arm (TYPEOPT_NEEDSALT). max_iter
                          * forced to 1 host-side; the algorithm's FIXED
                          * 1000-iter loop runs INSIDE template_finalize.
                          * Phase 1 of the Unix-crypt ladder. */
                         g->op == JOB_MD5CRYPT ||
                         /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT
                          * (e512) -- routes through the hand-written Path A
                          * salted-template kernel (gpu_shacrypt_core.cl at
                          * HASH_WORDS=8). Hit-replay 32-byte binary SHA-256
                          * digest (8 hash words; gpu_hash_words returns 8 for
                          * JOB_SHA256CRYPT) probed via hybrid_check; NEW
                          * hit-replay arm reconstructs "$5$[rounds=N$]<salt>$
                          * <43-char-base64>" via sha256crypt_b64encode (NEW
                          * helper in this commit; ports the per-byte permu-
                          * tation from mdxfind.c:12753-12980). The salt
                          * source is Typesalt (variable-length
                          * "$5$[rounds=N$]<salt>$" prefix); gpu_salt_judy
                          * (JOB_SHA256CRYPT) resolves Typesalt[JOB_-
                          * SHA256CRYPT] via the default arm (TYPEOPT_-
                          * NEEDSALT). max_iter forced to 1 host-side; the
                          * algorithm's variable-rounds loop (default 5000;
                          * configurable via "rounds=N$" salt prefix) runs
                          * INSIDE template_finalize. Phase 2 of the Unix-
                          * crypt ladder. */
                         g->op == JOB_SHA256CRYPT ||
                         /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT
                          * (e513) -- routes through the hand-written Path A
                          * salted-template kernel (gpu_shacrypt_core.cl at
                          * HASH_WORDS=16). Hit-replay 64-byte binary SHA-512
                          * digest (16 hash words; gpu_hash_words returns 16
                          * for JOB_SHA512CRYPT) probed via hybrid_check;
                          * NEW hit-replay arm reconstructs "$6$[rounds=N$]
                          * <salt>$<86-char-base64>" via sha512crypt_b64-
                          * encode (NEW helper in this commit; ports the
                          * per-byte permutation from mdxfind.c:12361-12780,
                          * which matches the compact-table loader's
                          * permutation at mdxfind.c:37036-37041). The salt
                          * source is Typesalt (variable-length
                          * "$6$[rounds=N$]<salt>$" prefix); gpu_salt_judy
                          * (JOB_SHA512CRYPT) resolves Typesalt[JOB_-
                          * SHA512CRYPT] via the default arm (TYPEOPT_-
                          * NEEDSALT). max_iter forced to 1 host-side; the
                          * algorithm's variable-rounds loop (default 5000;
                          * configurable via "rounds=N$" salt prefix) runs
                          * INSIDE template_finalize. Phase 3 of the Unix-
                          * crypt ladder. */
                         g->op == JOB_SHA512CRYPT ||
                         /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_-
                          * SHA512CRYPTMD5 (e510) -- REUSES Phase 3 SHA512-
                          * CRYPT's compiled kernel (gpu_shacrypt_core.cl
                          * at HASH_WORDS=16). Hit-replay 64-byte binary
                          * SHA-512 digest probed via hybrid_check -- the
                          * b64 prefix-scan + reconstruction path is
                          * IDENTICAL to JOB_SHA512CRYPT (same "$6$
                          * [rounds=N$]<salt>$" prefix shape, same 86-char
                          * b64 tail). The salt source is Typesalt[JOB_-
                          * SHA512CRYPTMD5], a sibling Judy that mdxfind.c:
                          * 47077-47087 populates with the SAME line as
                          * Typesalt[JOB_SHA512CRYPT]. The MD5-preprocess
                          * is HOST-side (mdxfind.c:12256-12258); the GPU
                          * runs the IDENTICAL SHA-512 chain. is_salted_op
                          * is true so the 3-axis combined_ridx decompose
                          * uses nsalts_packed. Phase 4 of the Unix-crypt
                          * ladder. */
                         g->op == JOB_SHA512CRYPTMD5 ||
                         /* DESCRYPT carrier (2026-05-08, Unix-crypt
                          * Phase 5): JOB_DESCRYPT (e500) -- routes
                          * through the hand-written Path A salted-
                          * template kernel (gpu_descrypt_core.cl) at
                          * HASH_WORDS=4. Hit-replay 13-char crypt(3)
                          * hash via des_reconstruct (existing helper at
                          * gpujob_opencl.c:448-481, ports the inverse
                          * FP permutation byte-by-byte from the slab
                          * arm). The salt source is Typesalt[JOB_-
                          * DESCRYPT] (TYPEOPT_NEEDSALT default); gpu_-
                          * salt_judy(JOB_DESCRYPT) resolves Typesalt at
                          * the GPU pack site. The salt-pack filter at
                          * gpu_pack_salts skips saltlen != 2 entries
                          * (extended-DES `_CCCCSSSS` 9-char salts CPU-
                          * fallback through bsd_crypt_des). max_iter
                          * forced to 1 host-side; 25-iter Feistel chain
                          * (FIXED count) runs INSIDE template_finalize.
                          * is_salted_op is true so the 3-axis combined_-
                          * ridx decompose uses nsalts_packed. Phase 5
                          * of the Unix-crypt ladder (FINAL phase). */
                         g->op == JOB_DESCRYPT ||
                         /* BCRYPT carrier (2026-05-08, Unix-crypt
                          * Phase 6): JOB_BCRYPT (e450) -- routes through
                          * the hand-written Path A salted-template
                          * kernel (gpu_bcrypt_core.cl) at HASH_WORDS=6.
                          * Hit-replay 60-char "$2[abkxy]$NN$<22-b64-
                          * salt><31-b64-hash>" crypt(3) hash via bf_-
                          * encode_23 (NEW helper at gpujob_opencl.c
                          * ~485-510, ports b64 encoding from slab arm
                          * lines 2264-2287). The salt source is Type-
                          * salt[JOB_BCRYPT] (TYPEOPT_NEEDSALT default);
                          * gpu_salt_judy(JOB_BCRYPT) resolves Typesalt
                          * at the GPU pack site. max_iter forced to 1
                          * host-side; 2^cost Eksblowfish chain (cost
                          * parsed per-salt-string at kernel entry; SIMT
                          * divergence accepted) runs INSIDE template_-
                          * finalize. is_salted_op is true so the 3-axis
                          * combined_ridx decompose uses nsalts_packed.
                          * Compound siblings (BCRYPTMD5/BCRYPTSHA1/
                          * BCRYPTSHA512) are CPU-only via gpu_op_-
                          * category default fall-through (NOT in this
                          * is_salted_op chain). Phase 6 of the slab-
                          * retirement ladder (final major slab kernel). */
                         g->op == JOB_BCRYPT);
                    uint32_t nsalts_for_decode =
                        is_salted_op ? (uint32_t)nsalts_packed : 1u;
                    if (nsalts_for_decode == 0u) nsalts_for_decode = 1u;

                    for (int h = 0; h < stored; h++) {
                        uint32_t *entry = hits + h * GPU_HIT_STRIDE;
                        uint32_t widx   = entry[0];
                        uint32_t combined_ridx = entry[1];
                        /* B6: three-axis decompose. salt_idx_global is
                         * the index into pack_map[] (== into the host's
                         * salt snapshot). For unsalted: nsalts_for_decode
                         * == 1, salt_idx_global == 0, the rest collapses
                         * to the pre-B6 layout. */
                        uint32_t salt_idx_global = combined_ridx % nsalts_for_decode;
                        uint32_t tmp             = combined_ridx / nsalts_for_decode;
                        /* Phase 1.5 (2026-05-10): b71_mask_size is uint64
                         * but combined_ridx (and thus tmp) is uint32, so
                         * the divmod result fits uint32. Cast for clarity. */
                        uint32_t mask_idx        = (uint32_t)((uint64_t)tmp % b71_mask_size);
                        uint32_t ridx            = (uint32_t)((uint64_t)tmp / b71_mask_size);
                        /* BF Phase 1.8 (2026-05-10): for BF chunks with
                         * inner_iter > 1 the kernel encoded iter_idx in
                         * the rule slot (combined_ridx =
                         * (rule_idx + iter_idx * n_rules) * mask_size +
                         * mask_idx_local). Since BF chunks have n_rules=1,
                         * the divmod above returned ridx=iter_idx. Swap
                         * them: iter_idx claims the bits, ridx resets to 0.
                         * For inner_iter==0/1, iter_idx is 0 (or harmless)
                         * and ridx stays 0 — bit-identical to pre-1.8. */
                        uint32_t iter_idx_local = 0u;
                        if (g->bf_chunk && g->bf_inner_iter > 1u) {
                            iter_idx_local = ridx;
                            ridx           = 0u;
                        }
                        int iter_num    = entry[2];

                        if (widx >= g->packed_count) continue;
                        if ((int)ridx >= gpu_rule_count) continue;
                        if (b71_mask_active && mask_idx >= b71_mask_size) continue;
                        if (is_salted_op && (int)salt_idx_global >= nsalts_packed) continue;

                        /* B6 salt-axis: resolve salt bytes for checkhashsalt. */
                        char *salt_bytes = NULL;
                        int   salt_len_b = 0;
                        struct saltentry *salt_snap_entry = NULL;
                        if (is_salted_op) {
                            int snap_idx = pack_map[salt_idx_global];
                            if (snap_idx < 0) continue;  /* defensive */
                            salt_snap_entry = &saltsnap[snap_idx];
                            salt_bytes      = salt_snap_entry->salt;
                            salt_len_b      = salt_snap_entry->saltlen;
                        }

                        /* Decode the candidate hash from the hit entry. */
                        { int hw = gpu_hash_words(g->op);
                          for (int w = 0; w < hw; w++)
                              curin.i[w] = entry[3 + w];
                        }

                        /* Recover original word from packed_buf. */
                        uint32_t pos = g->word_offset[widx];
                        if (pos >= g->packed_pos) continue;
                        uint8_t plen = (uint8_t)g->packed_buf[pos];
                        if (pos + 1 + plen > g->packed_pos) continue;
                        char *pword = g->packed_buf + pos + 1;

                        /* Map GPU rule index -> original Rules[] index.
                         * Sentinel orig_idx == -1 means this hit came from the
                         * synthetic `:` no-rule pass (auto-injected at session
                         * start so the GPU produces md5(word) for free); skip
                         * applyrule replay and use the original word directly. */
                        int orig_idx = gpu_rule_origin[ridx];
                        int out_len;
                        if (orig_idx == -1) {
                            /* Synthetic `:` — plaintext IS the original word. */
                            memcpy(synthetic_job.line, pword, plen);
                            synthetic_job.line[plen] = 0;
                            out_len = (int)plen;
                            synthetic_job.Ruleindex = 0;  /* no rule applied */
                        } else {
                            if (orig_idx < 0 || orig_idx >= (int)Numrules) continue;

                            /* Replay applyrule() to reconstruct the transformed plaintext. */
                            memcpy(synthetic_job.line, pword, plen);
                            synthetic_job.line[plen] = 0;

                            char *rule_bc = (_rule_ptr_cache && _rule_ptr_nrules == (int)Numrules)
                                            ? _rule_ptr_cache[orig_idx]
                                            : NULL;
                            if (!rule_bc) continue;

                            int new_len = applyrule(synthetic_job.line, _tpass, (int)plen,
                                                    rule_bc, &_ws);
                            if (new_len == -2) {
                                /* Auto-skip: output equals input. */
                                out_len = (int)plen;
                                /* synthetic_job.line already holds the original word. */
                            } else if (new_len < 0) {
                                /* Rule errored — skip this hit. */
                                continue;
                            } else {
                                memcpy(synthetic_job.line, _tpass, new_len);
                                synthetic_job.line[new_len] = 0;
                                out_len = new_len;
                            }
                            synthetic_job.Ruleindex = orig_idx;
                        }

                        /* B7.1-B7.5: prepend+append the mask characters to
                         * the candidate plaintext. The kernel did the same
                         * byte modification after the rule walker, so
                         * checkhash() sees the EXACT plaintext that hashed
                         * to the matched digest. mask_idx came from the
                         * combined_ridx divmod above.
                         *
                         * Slab convention (matches gpu_template.cl and
                         * gpu_kernels.cl md5_mask_batch):
                         *   mask_idx = prepend_idx * append_combos + append_idx
                         *   append_combos = product(sizes[npre..npre+napp))
                         *
                         * Decomposition: within each section iterate from
                         * highest index to lowest (last position innermost).
                         * mask_charsets layout in gpu_mask_desc: bytes
                         * [0..n_total) are the sizes header; bytes
                         * [n_total + i*256..n_total+(i+1)*256) are tables[i]
                         * for position i (prepend [0..npre), append
                         * [npre..npre+napp)).
                         *
                         * Output assembly: [prepend][rule_output][append].
                         * Use memmove to shift the rule output right by
                         * npre bytes. */
                        if (b71_mask_active) {
                            int npre = gpu_mask_n_prepend;
                            int napp = gpu_mask_n_append;
                            if (npre > 16) npre = 16;
                            if (napp > 16) napp = 16;
                            int n_total = npre + napp;
                            /* Compute append_combos from sizes[npre..npre+napp).
                             * Phase 1.5 (2026-05-10): widened to uint64 to mirror
                             * kernel; ?d^10 = 1e10 > 2^32 wraps a uint32. */
                            uint64_t append_combos = 1u;
                            for (int j = 0; j < napp; j++) {
                                int sz = gpu_mask_sizes[npre + j];
                                if (sz <= 0) sz = 1;
                                append_combos *= (uint64_t)sz;
                            }
                            if (append_combos == 0u) append_combos = 1u;
                            /* BF chunk-as-job Phase 1.5 (2026-05-10): kernel
                             * emits LOCAL mask_idx in [0, bf_num_masks) for
                             * BF chunks (or [0, gpu_mask_total) for non-BF,
                             * which equals absolute when bf_chunk==0). Host
                             * reconstructs absolute via:
                             *   mask_idx_abs = bf_mask_start
                             *                + widx * bf_offset_per_word
                             *                + mask_idx_local
                             * which supports keyspaces > 4G because
                             * bf_mask_start is uint64 and accumulates across
                             * chunks. For non-BF (bf_chunk==0), bf_mask_start
                             * and bf_offset_per_word are both 0; the formula
                             * collapses to mask_idx_abs = mask_idx (host
                             * divmod above already recovered the absolute
                             * value when bf_num_masks==0 and b71_mask_size
                             * == gpu_mask_total). Restores the host re-add
                             * that Tranche 3 removed; the kernel's
                             * compensating absolute shift was wrong above
                             * 2^32 because of uint truncation. */
                            uint64_t mask_idx_abs = (uint64_t)mask_idx;
                            if (g->bf_chunk) {
                                /* BF Phase 1.8: add iter stride. For
                                 * inner_iter==1 (servo default or salted
                                 * guard) iter_idx_local is 0 and this
                                 * term is 0 — bit-identical to pre-1.8.
                                 * For inner_iter>1 the kernel walked
                                 * mask_idx_local through [0..mask_size)
                                 * inner_iter times, each shift adding
                                 * mask_size (== bf_num_masks) to the
                                 * absolute mask position. */
                                mask_idx_abs = (uint64_t)g->bf_mask_start
                                             + (uint64_t)widx
                                                 * (uint64_t)g->bf_offset_per_word
                                             + (uint64_t)iter_idx_local
                                                 * (uint64_t)g->bf_num_masks
                                             + (uint64_t)mask_idx;
                            }
                            uint64_t prepend_idx = mask_idx_abs / (uint64_t)append_combos;
                            uint64_t append_idx  = mask_idx_abs % (uint64_t)append_combos;

                            char prepend_chars[16];
                            char append_chars[16];
                            /* Decode prepend chars (positions [0..npre) in
                             * mask_charsets layout). */
                            {
                                uint64_t remaining = prepend_idx;
                                for (int k = 0; k < npre; k++) {
                                    int i = npre - 1 - k;
                                    int sz = gpu_mask_sizes[i];
                                    if (sz <= 0) sz = 1;
                                    int pidx = (int)(remaining % (uint64_t)sz);
                                    remaining /= (uint64_t)sz;
                                    prepend_chars[i] = (char)gpu_mask_desc[
                                        (size_t)n_total + (size_t)i * 256 + pidx];
                                }
                            }
                            /* Decode append chars (positions [npre..npre+napp)
                             * in mask_charsets layout). */
                            {
                                uint64_t remaining = append_idx;
                                for (int k = 0; k < napp; k++) {
                                    int i = napp - 1 - k;
                                    int row = npre + i;
                                    int sz = gpu_mask_sizes[row];
                                    if (sz <= 0) sz = 1;
                                    int pidx = (int)(remaining % (uint64_t)sz);
                                    remaining /= (uint64_t)sz;
                                    append_chars[i] = (char)gpu_mask_desc[
                                        (size_t)n_total + (size_t)row * 256 + pidx];
                                }
                            }
                            /* Assemble [prepend][rule_output][append] in
                             * synthetic_job.line. Shift the existing rule
                             * output right by npre via memmove (handles
                             * overlap), then write prepend at the front
                             * and append at the end. */
                            if (out_len + npre + napp <= (int)(MAXLINE + 60)) {
                                if (npre > 0) {
                                    memmove(synthetic_job.line + npre,
                                            synthetic_job.line,
                                            (size_t)out_len);
                                    for (int i = 0; i < npre; i++) {
                                        synthetic_job.line[i] = prepend_chars[i];
                                    }
                                }
                                for (int i = 0; i < napp; i++) {
                                    synthetic_job.line[npre + out_len + i] =
                                        append_chars[i];
                                }
                                out_len += npre + napp;
                                synthetic_job.line[out_len] = 0;
                            }
                        }
                        synthetic_job.clen       = out_len;
                        synthetic_job.pass       = synthetic_job.line;
                        /* B5 sub-batch 2 debug: dump GPU output bytes when
                         * MDXFIND_GPU_DEBUG_DIGEST is set. Helps diagnose
                         * byte-exact mismatches in new template kernels. */
                        if (getenv("MDXFIND_GPU_DEBUG_DIGEST")) {
                            static __thread int _logged = 0;
                            if (_logged < 5) {
                                fprintf(stderr, "[gpu-dbg] op=%d hexlen=%d widx=%u ridx=%u mask_idx=%u plain=\"%.*s\" digest=",
                                        g->op, hexlen, widx, ridx, mask_idx,
                                        out_len, synthetic_job.line);
                                int dbytes = hexlen / 2;
                                if (dbytes > 64) dbytes = 64;
                                for (int b = 0; b < dbytes; b++)
                                    fprintf(stderr, "%02x", curin.h[b]);
                                fprintf(stderr, "\n");
                                _logged++;
                            }
                        }
                        /* B6 salt-axis (2026-05-06; §11 row 24): salted
                         * ops route through checkhashkey (iter==1) or
                         * checkhashsalt (iter > 1) — matches CPU path
                         * MD5SALT at mdxfind.c:22198 / 22207. The
                         * iter==1 CPU branch uses checkhashkey which
                         * outputs the no-iter label `MD5SALT %s:%s:%s\n`
                         * (no `x01` suffix); the iter > 1 branch uses
                         * checkhashsalt which outputs `MD5SALTx%02d`.
                         * For JOB_MD5SALTPASS the CPU path at
                         * mdxfind.c:15813 always uses checkhashsalt
                         * (different convention — its iter==1 output is
                         * `MD5SALTPASSx01`); but mdxfind.c:15813 passes
                         * `x` as the loop counter starting from 1, and
                         * the format gate at line 8886 emits `x01` for
                         * x>0. Both algos here pass iter_num==1 from
                         * the kernel, so the per-op CPU convention is
                         * what we need to mirror. */
                        if (is_salted_op) {
                            int hit = 0;
                            if (g->op == JOB_MD5SALT && iter_num == 1) {
                                /* iter==1 CPU path uses checkhashkey
                                 * (no iter suffix in label). */
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            /* Family A (2026-05-07): HMAC-MD5 routing.
                             *   KSALT (e214): mdxfind.c:29386 uses
                             *     checkhashkey(curin, hmac_len, tsalt, job)
                             *     — emits `HMACMD5 %s:%s:%s\n` with no
                             *     iter suffix.
                             *   KPASS (e792): mdxfind.c:29490 uses
                             *     checkhashsalt(curin, hmac_len, s1,
                             *     saltlen, 0, job) — emits `HMACMD5KPASS
                             *     %s:%s:%s\n` (iter=0 sentinel suppresses
                             *     the xNN suffix in the format gate).
                             * max_iter is forced to 1 host-side, so iter_num
                             * is always 1 here — but match the CPU label
                             * convention exactly: KSALT -> checkhashkey;
                             * KPASS -> checkhashsalt with iter=0. */
                            } else if (g->op == JOB_HMAC_MD5) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_MD5_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family B (2026-05-07): HMAC-SHA1 routing.
                             *   KSALT (e215): mdxfind.c:29301 uses
                             *     checkhashkey(curin, hexlen, tsalt, job)
                             *     — emits "HMACSHA1 %s:%s:%s\n" with no
                             *     iter suffix.
                             *   KPASS (e793): mdxfind.c:29454 uses
                             *     checkhashsalt(curin, hexlen, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMACSHA1KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirror
                             * Family A label-convention exactly. */
                            } else if (g->op == JOB_HMAC_SHA1) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_SHA1_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family C (2026-05-07): HMAC-SHA224 routing.
                             *   KSALT (e216): mdxfind.c:29326 case +
                             *     HMAC_start label uses
                             *     checkhashkey(curin, hexlen=56, tsalt, job)
                             *     — emits "HMAC-SHA224 %s:%s:%s\n" with
                             *     no iter suffix.
                             *   KPASS (e794): mdxfind.c:29479 case +
                             *     HMAC_KPASS_start label uses
                             *     checkhashsalt(curin, hexlen=56, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMAC-SHA224-KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirrors
                             * Families A and B label-convention. */
                            } else if (g->op == JOB_HMAC_SHA224) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_SHA224_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family D (2026-05-08): HMAC-SHA256 routing.
                             *   KSALT (e217): mdxfind.c:29581 case +
                             *     HMAC_start label uses
                             *     checkhashkey(curin, hexlen=64, tsalt, job)
                             *     — emits "HMAC-SHA256 %s:%s:%s\n" with
                             *     no iter suffix.
                             *   KPASS (e795): mdxfind.c:29734 case +
                             *     HMAC_KPASS_start label uses
                             *     checkhashsalt(curin, hexlen=64, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMAC-SHA256-KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirrors
                             * Families A/B/C label-convention. Final HMAC
                             * family in the ladder; HMAC ladder COMPLETE
                             * 21/21 algos shipped. */
                            } else if (g->op == JOB_HMAC_SHA256) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_SHA256_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family E HMAC-SHA384 carrier (2026-05-08):
                             *   KSALT (e543): mdxfind.c:29369 case +
                             *     HMAC_start label uses
                             *     checkhashkey(curin, hexlen=96, tsalt, job)
                             *     — emits "HMAC-SHA384 %s:%s:%s\n" with
                             *     no iter suffix.
                             *   KPASS (e796): mdxfind.c:29522 case +
                             *     HMAC_KPASS_start label uses
                             *     checkhashsalt(curin, hexlen=96, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMAC-SHA384-KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirrors
                             * Families A/B/C label-convention. */
                            } else if (g->op == JOB_HMAC_SHA384) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_SHA384_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family F (2026-05-08): HMAC-SHA512 routing.
                             *   KSALT (e218): mdxfind.c:29400 case +
                             *     HMAC_start label uses
                             *     checkhashkey(curin, hexlen=128, tsalt, job)
                             *     — emits "HMAC-SHA512 %s:%s:%s\n" with
                             *     no iter suffix.
                             *   KPASS (e797): mdxfind.c:29553 case +
                             *     HMAC_KPASS_start label uses
                             *     checkhashsalt(curin, hexlen=128, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMAC-SHA512-KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirrors
                             * Families A/B/C/E label-convention. */
                            } else if (g->op == JOB_HMAC_SHA512) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_SHA512_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family G HMAC-RIPEMD-160 carrier (2026-05-08):
                             *   KSALT (e211): mdxfind.c:29391 case +
                             *     HMAC_start label uses
                             *     checkhashkey(curin, hexlen=40, tsalt, job)
                             *     — emits "HMAC-RMD160 %s:%s:%s\n" with
                             *     no iter suffix.
                             *   KPASS (e798): mdxfind.c:29584 case +
                             *     HMAC_KPASS_start label uses
                             *     checkhashsalt(curin, hexlen=40, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMAC-RMD160-KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirrors
                             * Families A/B/C/E/F label-convention. */
                            } else if (g->op == JOB_HMAC_RMD160) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_RMD160_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family H HMAC-RIPEMD-320 carrier (2026-05-08):
                             *   KSALT (e213): mdxfind.c:29428 case +
                             *     HMAC_start label uses
                             *     checkhashkey(curin, hexlen=80, tsalt, job)
                             *     — emits "HMAC-RMD320 %s:%s:%s\n" with
                             *     no iter suffix.
                             *   KPASS (e799): mdxfind.c:29616 case +
                             *     HMAC_KPASS_start label uses
                             *     checkhashsalt(curin, hexlen=80, s1,
                             *     saltlen, 0, job) — emits
                             *     "HMAC-RMD320-KPASS %s:%s:%s\n" (iter=0
                             *     sentinel suppresses xNN suffix).
                             * max_iter forced to 1 host-side. Mirrors
                             * Families A/B/C/E/F/G label-convention. */
                            } else if (g->op == JOB_HMAC_RMD320) {
                                hit = checkhashkey(&curin, hexlen,
                                                   salt_bytes,
                                                   &synthetic_job);
                            } else if (g->op == JOB_HMAC_RMD320_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family I HMAC-BLAKE2S carrier (2026-05-08):
                             *   JOB_HMAC_BLAKE2S (e828): mdxfind.c:30391 calls
                             *     checkhashsalt(curin, hexlen=64, s1, saltlen,
                             *     0, job) — emits "HMAC-BLAKE2S %s:%s:%s\n"
                             *     with no iter suffix (iter=0 sentinel).
                             * Single algo_mode (5); no KPASS sibling op. The
                             * algorithm is KPASS-shape (key=pass, msg=salt)
                             * but named without -KPASS suffix because mdxfind
                             * never created a KSALT sibling. max_iter forced
                             * to 1 host-side. Mirrors Families G/H KPASS arm
                             * label-convention. */
                            } else if (g->op == JOB_HMAC_BLAKE2S) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family J HMAC-STREEBOG-256 carrier (2026-05-08):
                             *   JOB_HMAC_STREEBOG256_KSALT (e838): mdxfind.c:
                             *     30868 calls checkhashsalt(curin, hexlen=64,
                             *     s1, saltlen, 0, job) — emits "HMAC-STREEBOG-
                             *     256-KSALT %s:%s:%s\n" with no iter suffix.
                             *   JOB_HMAC_STREEBOG256_KPASS (e837): mdxfind.c:
                             *     30810 calls checkhashsalt(curin, hexlen=64,
                             *     s1, saltlen, 0, job) — emits "HMAC-STREEBOG-
                             *     256-KPASS %s:%s:%s\n" with no iter suffix.
                             * BOTH ops use checkhashsalt (unlike Families A-H
                             * KSALT siblings which use checkhashkey). hexlen=
                             * 64 = 64 hex chars / 32 bytes (8 LE uint32 =
                             * HASH_WORDS=8). max_iter forced to 1 host-side. */
                            } else if (g->op == JOB_HMAC_STREEBOG256_KSALT ||
                                       g->op == JOB_HMAC_STREEBOG256_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* Family K HMAC-STREEBOG-512 carrier (2026-05-08):
                             *   JOB_HMAC_STREEBOG512_KSALT (e840): mdxfind.c:
                             *     31020 calls checkhashsalt(curin, hexlen=128,
                             *     s1, saltlen, 0, job) - emits "HMAC-STREEBOG-
                             *     512-KSALT %s:%s:%s\n" with no iter suffix.
                             *   JOB_HMAC_STREEBOG512_KPASS (e839): mdxfind.c:
                             *     30963 calls checkhashsalt(curin, hexlen=128,
                             *     s1, saltlen, 0, job) - emits "HMAC-STREEBOG-
                             *     512-KPASS %s:%s:%s\n" with no iter suffix.
                             * BOTH ops use checkhashsalt (mirrors Family J
                             * STREEBOG-256 pattern; unlike Families A-H
                             * KSALT siblings which use checkhashkey).
                             * hexlen=128 = 128 hex chars / 64 bytes (16 LE
                             * uint32 = HASH_WORDS=16). max_iter forced to
                             * 1 host-side. Final HMAC family in the ladder. */
                            } else if (g->op == JOB_HMAC_STREEBOG512_KSALT ||
                                       g->op == JOB_HMAC_STREEBOG512_KPASS) {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    0,
                                                    &synthetic_job);
                            /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455)
                             * routing. CPU semantics at mdxfind.c:13620 calls
                             * checkhashbb(curin, 32, s1, job) where s1 is the
                             * full 12-byte "$H$<cost><8>" salt prefix from
                             * saltsnap[si].salt. The hit-replay arm mirrors
                             * the slab-path arm at line 1682 (the slab
                             * already used checkhashbb). Hexlen=32 = 32 hex
                             * chars / 16 bytes (4 LE uint32 = HASH_WORDS=4
                             * = MD5 width). max_iter forced to 1 host-side;
                             * the algorithm's internal iter count is decoded
                             * from salt_bytes[3] inside template_finalize.
                             * NOT routed through checkhashkey/checkhashsalt
                             * because PHPBB3 has its own bb-specific output
                             * format (phpitoa64-encoded 22-char hash + the
                             * salt prefix), distinct from the HMAC families'
                             * hex-encoded outputs. */
                            } else if (g->op == JOB_PHPBB3) {
                                hit = checkhashbb(&curin, hexlen,
                                                  salt_bytes,
                                                  &synthetic_job);
                            /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511)
                             * routing. CPU semantics at mdxfind.c:13071 calls
                             * hybrid_check(curin.h, 16, &match_len, &match_-
                             * flags) on the 16-byte binary MD5 digest, then
                             * reconstructs "$1$<salt>$<22-char-phpitoa64>"
                             * via the md5crypt_b64encode helper. We mirror
                             * the slab-path arm at line 1723 byte-for-byte
                             * (probe + reconstruct + prfound). NOT routed
                             * through checkhashbb / checkhashkey / check-
                             * hashsalt because MD5CRYPT has its own bespoke
                             * output format (phpitoa64-encoded 22-char hash
                             * with custom byte permutation distinct from
                             * PHPBB3's). hit is set to 0 here -- the arm
                             * does its own PV_DEC + prfound inline (mirroring
                             * the slab arm); the outer salt-snapshot PV_DEC
                             * at line 1527 is gated by hit && salt_snap_-
                             * entry, so leaving hit=0 keeps PV accounting
                             * single-path. Phase 1 of the Unix-crypt ladder. */
                            } else if (g->op == JOB_MD5CRYPT) {
                                /* hit stays 0: this arm does its own
                                 * PV_DEC + prfound inline (mirroring slab
                                 * arm at line 1723); the outer salt-snap
                                 * PV_DEC at line ~1546 is gated by
                                 * `hit && salt_snap_entry`, so leaving
                                 * hit=0 keeps PV accounting single-path. */
                                hit = 0;
                                if (salt_snap_entry) {
                                    int match_len; unsigned short *match_flags;
                                    int hf = hybrid_check(curin.h, 16,
                                                          &match_len, &match_flags);
                                    if (hf && *match_flags != (unsigned short)g->op) {
                                        *match_flags = g->op;
                                        PV_DEC(salt_snap_entry->PV);
                                        char *sp = salt_snap_entry->salt;
                                        int splen = salt_snap_entry->saltlen;
                                        char mdbuf[128];
                                        memcpy(mdbuf, sp, splen);
                                        md5crypt_b64encode(curin.h, mdbuf + splen);
                                        prfound(&synthetic_job, mdbuf);
                                    }
                                }
                            /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT
                             * (e512) routing. CPU semantics at mdxfind.c:12290
                             * computes 32-byte SHA-256 digest in curin.h, then
                             * reconstructs "$5$[rounds=N$]<salt>$<43-base64>"
                             * via the cryptlen=32 b64 byte-permutation table at
                             * mdxfind.c:12753-12980 (now wrapped in
                             * sha256crypt_b64encode helper above). We mirror
                             * the MD5CRYPT arm structure: hybrid_check on the
                             * 32-byte binary digest, gate on match-flag races,
                             * PV_DEC + prfound with salt-prefix-prepended b64
                             * output. NOT routed through checkhashbb /
                             * checkhashkey / checkhashsalt because SHA256CRYPT
                             * has its own bespoke output format (43-char base64
                             * with the 32-byte permutation distinct from MD5-
                             * CRYPT's 22-char or PHPBB3's 22-char encodings).
                             * hit stays 0 (this arm does its own PV_DEC +
                             * prfound inline; outer salt-snap PV_DEC gate is
                             * `hit && salt_snap_entry`, leaving hit=0 keeps
                             * PV accounting single-path). Uses salt_snap_entry
                             * + salt_bytes per feedback_rules_engine_hit_-
                             * replay_vars.md (rules-engine context, NOT slab
                             * `sidx`). Phase 2 of the Unix-crypt ladder. */
                            } else if (g->op == JOB_SHA256CRYPT) {
                                hit = 0;
                                if (salt_snap_entry) {
                                    int match_len; unsigned short *match_flags;
                                    int hf = hybrid_check(curin.h, 32,
                                                          &match_len, &match_flags);
                                    if (hf && *match_flags != (unsigned short)g->op) {
                                        *match_flags = g->op;
                                        PV_DEC(salt_snap_entry->PV);
                                        char *sp = salt_snap_entry->salt;
                                        int splen = salt_snap_entry->saltlen;
                                        /* Typesalt[JOB_SHA256CRYPT] holds the
                                         * FULL hash line "$5$[rounds=N$]<salt>$
                                         * <43-b64>" (loader at mdxfind.c:47013-
                                         * 47041 inserts the line verbatim with
                                         * no b64-tail strip; sibling consumers
                                         * in crypt_round at mdxfind.c:12217-
                                         * 12290 rely on the b64 tail for early-
                                         * out comparisons, so the schema must
                                         * not change).  Derive the salt-prefix
                                         * length by scanning back to the last
                                         * '$':  the b64 alphabet phpitoa64
                                         * (./0-9A-Za-z, mdxfind.c:2811) excludes
                                         * '$', so the final '$' reliably
                                         * terminates "$5$[rounds=N$]<salt>$".
                                         * Without this scan we'd write the b64
                                         * AFTER the existing tail, producing
                                         * "<full-line><freshly-computed-b64>:
                                         * <password>" instead of the expected
                                         * "<full-line>:<password>".
                                         * Output buffer: salt prefix (up to
                                         * ~30 bytes incl rounds=N$) + 43-char
                                         * b64 + NUL.  128 is comfortably
                                         * oversized. */
                                        char mdbuf[128];
                                        int prefix_len = splen;
                                        while (prefix_len > 0 && sp[prefix_len - 1] != '$')
                                            prefix_len--;
                                        memcpy(mdbuf, sp, prefix_len);
                                        sha256crypt_b64encode(curin.h, mdbuf + prefix_len);
                                        prfound(&synthetic_job, mdbuf);
                                    }
                                }
                            /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT
                             * (e513) routing. CPU semantics at mdxfind.c:12290
                             * (cryptlen=64 branch) computes 64-byte SHA-512
                             * digest in curin.h, then reconstructs
                             * "$6$[rounds=N$]<salt>$<86-base64>" via the
                             * cryptlen=64 b64 byte-permutation table at
                             * mdxfind.c:12361-12780 (now wrapped in
                             * sha512crypt_b64encode helper above). We mirror
                             * the SHA256CRYPT arm structure verbatim, swap-
                             * ping the digest length 32 -> 64 and the b64
                             * encoder. mdbuf[128] still suffices: 30-byte
                             * prefix cap + 86 b64 chars + 1 NUL = 117 bytes.
                             * NOT routed through checkhashbb / checkhashkey /
                             * checkhashsalt because SHA512CRYPT has its own
                             * bespoke output format (86-char base64 with the
                             * 64-byte permutation distinct from SHA256CRYPT's
                             * 43-char or MD5CRYPT's 22-char encodings).
                             * hit stays 0 (this arm does its own PV_DEC +
                             * prfound inline; outer salt-snap PV_DEC gate is
                             * `hit && salt_snap_entry`, leaving hit=0 keeps
                             * PV accounting single-path). Uses salt_snap_entry
                             * + salt_bytes per feedback_rules_engine_hit_-
                             * replay_vars.md (rules-engine context, NOT slab
                             * `sidx`). The last-`$` prefix-len scan is
                             * length-agnostic: phpitoa64 alphabet excludes
                             * '$', so the final '$' reliably terminates
                             * "$6$[rounds=N$]<salt>$" regardless of whether
                             * the trailing b64 is 43 chars (SHA256CRYPT) or
                             * 86 chars (SHA512CRYPT). Phase 3 of the Unix-
                             * crypt ladder. */
                            /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_-
                             * SHA512CRYPTMD5 (e510) routing. CPU semantics
                             * at mdxfind.c:12199-12212 + crypt_round at
                             * 12290 (cryptlen=64 branch) computes 64-byte
                             * SHA-512 digest in curin.h on a password
                             * pre-processed via MD5 (the 32-char hex of
                             * the original password). The MD5-preprocess
                             * happens HOST-side at mdxfind.c:12256-12258
                             * BEFORE gpu_try_pack -- the GPU runs the
                             * IDENTICAL SHA-512 crypt chain as JOB_SHA512-
                             * CRYPT. Hit-format and reconstruction shape
                             * are STRUCTURALLY IDENTICAL to SHA512CRYPT
                             * (same "$6$[rounds=N$]<salt>$<86-b64>" form;
                             * Typesalt[JOB_SHA512CRYPTMD5] carries the
                             * SAME line as Typesalt[JOB_SHA512CRYPT] per
                             * mdxfind.c:47077-47087). We share the SHA512-
                             * CRYPT arm verbatim by widening the gate to
                             * accept both ops. *match_flags = g->op
                             * preserves per-op deduplication (a $6$ line
                             * matched as JOB_SHA512CRYPT is not re-emitted
                             * as JOB_SHA512CRYPTMD5 within the same hit
                             * record, and vice versa). Phase 4 of the
                             * Unix-crypt ladder (final phase). */
                            } else if (g->op == JOB_SHA512CRYPT ||
                                       g->op == JOB_SHA512CRYPTMD5) {
                                hit = 0;
                                if (salt_snap_entry) {
                                    int match_len; unsigned short *match_flags;
                                    int hf = hybrid_check(curin.h, 64,
                                                          &match_len, &match_flags);
                                    if (hf && *match_flags != (unsigned short)g->op) {
                                        *match_flags = g->op;
                                        PV_DEC(salt_snap_entry->PV);
                                        char *sp = salt_snap_entry->salt;
                                        int splen = salt_snap_entry->saltlen;
                                        /* Typesalt[op] (where op is either
                                         * JOB_SHA512CRYPT or JOB_SHA512-
                                         * CRYPTMD5) holds the FULL hash
                                         * line "$6$[rounds=N$]<salt>$<86-
                                         * b64>" (loader at mdxfind.c:
                                         * 47049-47062 + 47077-47087
                                         * inserts the same line into BOTH
                                         * Judy arrays). Output buffer:
                                         * salt prefix (up to ~30 bytes
                                         * incl rounds=N$) + 86-char b64 +
                                         * NUL = up to 117 bytes. mdbuf
                                         * [128] is comfortably oversized. */
                                        char mdbuf[128];
                                        int prefix_len = splen;
                                        while (prefix_len > 0 && sp[prefix_len - 1] != '$')
                                            prefix_len--;
                                        memcpy(mdbuf, sp, prefix_len);
                                        sha512crypt_b64encode(curin.h, mdbuf + prefix_len);
                                        prfound(&synthetic_job, mdbuf);
                                    }
                                }
                            /* DESCRYPT carrier (2026-05-08, Unix-crypt
                             * Phase 5): JOB_DESCRYPT (e500) routing. The
                             * GPU emits pre-FP (l, r) in curin.i[0..1]
                             * (h[2..3] are zero-padded by the kernel to
                             * match the host compact-table layout 4 il +
                             * 4 ir + 8 zero pad). Hit-replay reconstructs
                             * the 13-char crypt(3) hash via the existing
                             * des_reconstruct helper at gpujob_opencl.c:
                             * 448-481 (applies inverse FP permutation
                             * byte-by-byte, then phpitoa64-encodes the 8
                             * post-FP bytes into 11 base64 chars + 2-byte
                             * salt prefix = 13 chars total). Probes
                             * JudyJ[JOB_DESCRYPT] for the 13-char string
                             * and uses CAS dedup (atomic compare-and-swap
                             * 0 -> 1 on the Judy value), mirroring the
                             * existing slab arm at gpujob_opencl.c:2113-
                             * 2130 byte-for-byte. NOT routed through
                             * checkhashbb / checkhashkey / checkhashsalt
                             * because DESCRYPT has its own bespoke output
                             * format (13-char salt+11-base64 crypt(3)
                             * hash, distinct from MD5CRYPT's 22-char b64
                             * or PHPBB3's 22-char b64). hit stays 0 (this
                             * arm does its own PV_DEC + prfound inline;
                             * outer salt-snap PV_DEC gate is `hit &&
                             * salt_snap_entry`, leaving hit=0 keeps PV
                             * accounting single-path). Uses salt_snap_-
                             * entry + salt_bytes per feedback_rules_-
                             * engine_hit_replay_vars.md (rules-engine
                             * context, NOT slab `sidx`). Display password
                             * CLAMPED TO 8 BYTES per CPU parity (mirrors
                             * mdxfind.c:23676-23677 `i = min(len, 8)` for
                             * non-extended salts; Q1 user decision 2026-
                             * 05-08). The post-rule plaintext lives in
                             * synthetic_job.line (already populated above
                             * by either the synthetic-`:` arm or the
                             * applyrule replay arm + optional mask
                             * prepend/append); we re-clamp the LENGTH for
                             * display while preserving the underlying
                             * buffer (the kernel only saw the first 8
                             * bytes anyway via the host-side rules-engine
                             * pack-site clamp at mdxfind.c:11021-11026 +
                             * the kernel-side `if (plen > 8) plen = 8;`
                             * defensive cap in template_finalize). Phase
                             * 5 of the Unix-crypt ladder (FINAL phase). */
                            } else if (g->op == JOB_DESCRYPT) {
                                hit = 0;
                                if (salt_snap_entry) {
                                    char desbuf[64];
                                    des_reconstruct(curin.i[0], curin.i[1],
                                                    salt_bytes, desbuf);
                                    Word_t *HPV;
                                    JSLG(HPV, JudyJ[JOB_DESCRYPT],
                                         (unsigned char *)desbuf);
                                    if (HPV && __sync_bool_compare_and_swap(HPV, 0, 1)) {
                                        PV_DEC(salt_snap_entry->PV);
                                        /* Q1 (2026-05-08): clamp display
                                         * password to 8 bytes for CPU
                                         * parity. The post-rule + post-
                                         * mask candidate already lives in
                                         * synthetic_job.line at length
                                         * out_len; re-emit it truncated
                                         * to 8 bytes via the synthetic_-
                                         * job.clen field. The line buffer
                                         * itself stays intact (no mutation
                                         * of bytes 9+) but prfound +
                                         * downstream printers honor
                                         * synthetic_job.clen. */
                                        int cplen = (out_len > 8) ? 8 : out_len;
                                        synthetic_job.line[cplen] = 0;
                                        synthetic_job.clen = cplen;
                                        prfound(&synthetic_job, desbuf);
                                    }
                                }
                            /* BCRYPT carrier (2026-05-08, Unix-crypt
                             * Phase 6): JOB_BCRYPT (e450) routing. The
                             * GPU emits 6 LE uint32 words = 24 bytes in
                             * curin.i[0..5] (raw byte stream after the
                             * kernel's BE->LE swap at gpu_bcrypt_core.cl
                             * lines 821-824). Hit-replay reconstructs
                             * the 60-char "$2[abkxy]$NN$<22-b64-salt>
                             * <31-b64-hash>" crypt(3) hash via bf_encode_23
                             * (NEW helper at gpujob_opencl.c ~485-510,
                             * ports the b64 encoding from the slab arm
                             * lines 2264-2287 which is being deleted in
                             * this same commit). The salt prefix
                             * (28-char "$2[abkxy]$NN$<22-b64-salt>" or
                             * 29-char $2k variant) lives in salt_bytes
                             * + salt_len_b -- mdxfind.c:40402-40436
                             * stores the FULL 60-char hash line into
                             * Typesalt[JOB_BCRYPT] with saltlen = total
                             * line length, but the salt prefix portion
                             * is what matters for reconstruction (the
                             * trailing 31-char hash portion gets over-
                             * written by the GPU's computed hash via
                             * bf_encode_23). Probes JudyJ[JOB_BCRYPT]
                             * for the full 60-char string and uses CAS
                             * dedup (atomic compare-and-swap 0 -> 1 on
                             * the Judy value), mirroring the existing
                             * slab arm byte-for-byte. NOT routed through
                             * checkhashbb / checkhashkey / checkhashsalt
                             * because BCRYPT has its own bespoke output
                             * format (60-char crypt(3) hash, distinct
                             * from MD5CRYPT's $1$/PHPBB3's $H$ shapes).
                             * hit stays 0 (this arm does its own PV_DEC
                             * + prfound inline; outer salt-snap PV_DEC
                             * gate is `hit && salt_snap_entry`, leaving
                             * hit=0 keeps PV accounting single-path).
                             * Uses salt_snap_entry + salt_bytes +
                             * salt_len_b per feedback_rules_engine_hit_-
                             * replay_vars.md (rules-engine context, NOT
                             * slab `sidx` like the deleted slab arm).
                             * Display password is FULL post-rule
                             * plaintext per Q1 user decision 2026-05-08
                             * (DIFFERENT from DESCRYPT's 8-byte clamp;
                             * BCRYPT does NOT clamp display -- the 72-
                             * byte truncation is INSIDE BF_set_key, not
                             * at display). The post-rule plaintext lives
                             * in synthetic_job.line at length out_len
                             * (already populated above by either the
                             * synthetic-`:` arm or the applyrule replay
                             * arm + optional mask prepend/append); we
                             * re-emit it AS-IS with synthetic_job.clen
                             * = out_len. Phase 6 of the slab-retirement
                             * ladder (final major slab kernel). */
                            } else if (g->op == JOB_BCRYPT) {
                                hit = 0;
                                if (salt_snap_entry) {
                                    /* GPU emits 6 LE uint32 words = 24
                                     * bytes in curin.i[0..5]; cast to
                                     * uchar for byte-stream encoding.
                                     * bf_encode_23 reads first 23 bytes
                                     * (24th is zero pad from BE->LE swap
                                     * tail; BF_encode discards it). */
                                    unsigned char *raw = (unsigned char *)&curin.i[0];
                                    char hashb64[32];
                                    bf_encode_23(raw, hashb64);
                                    /* Build full 60-char hash: salt prefix
                                     * (28 or 29 chars) + 31-char b64 hash.
                                     * fullhash[80] is comfortably oversized
                                     * for the 60-char standard / 59-char
                                     * $2k variant. */
                                    char fullhash[80];
                                    int splen = salt_len_b;
                                    /* The salt prefix in Typesalt[JOB_-
                                     * BCRYPT] is the full 60-char hash
                                     * line per mdxfind.c BCRYPT loader;
                                     * but for reconstruction we need only
                                     * the first 28 (or 29 for $2k$)
                                     * characters -- the part BEFORE the
                                     * 31-char b64 hash. The slab arm at
                                     * line 2289-2293 (being deleted)
                                     * memcpy'd `splen` bytes verbatim
                                     * then overwrote the trailing 31
                                     * with the computed hash; we mirror
                                     * that pattern exactly. salt_len_b
                                     * is the saltlen field from the
                                     * snapshot which carries the full
                                     * line length (60 or 59 chars); we
                                     * truncate to splen-31 for the
                                     * prefix portion. */
                                    int prefix_len = (splen > 31) ? splen - 31 : splen;
                                    if (prefix_len < 0) prefix_len = 0;
                                    if (prefix_len > 64) prefix_len = 64;
                                    memcpy(fullhash, salt_bytes, prefix_len);
                                    memcpy(fullhash + prefix_len, hashb64, 31);
                                    fullhash[prefix_len + 31] = 0;
                                    Word_t *HPV;
                                    JSLG(HPV, JudyJ[JOB_BCRYPT],
                                         (unsigned char *)fullhash);
                                    if (HPV && __sync_bool_compare_and_swap(HPV, 0, 1)) {
                                        PV_DEC(salt_snap_entry->PV);
                                        /* Q1 (2026-05-08): NO display
                                         * clamp. Render full post-rule
                                         * plaintext (CPU does NOT clamp
                                         * display for BCRYPT -- the 72-
                                         * byte truncation is INSIDE
                                         * BF_set_key, not at display).
                                         * DIFFERENT from DESCRYPT's
                                         * 8-byte clamp. The post-rule
                                         * plaintext at synthetic_job.line
                                         * is already at length out_len;
                                         * just set clen accordingly. */
                                        synthetic_job.line[out_len] = 0;
                                        synthetic_job.clen = out_len;
                                        prfound(&synthetic_job, fullhash);
                                    }
                                }
                            } else {
                                hit = checkhashsalt(&curin, hexlen,
                                                    salt_bytes, salt_len_b,
                                                    iter_num,
                                                    &synthetic_job);
                            }
                            if (hit && salt_snap_entry) {
                                PV_DEC(salt_snap_entry->PV);
                            }
                        } else {
                            checkhash(&curin, hexlen, iter_num, &synthetic_job);
                        }
                    }
                }

                /* Flush output buffer */
                if (synthetic_job.outlen > 0) {
                    fwrite(outbuf, synthetic_job.outlen, 1, stdout);
                    fflush(stdout);
                    synthetic_job.outlen = 0;
                }

                /* Salt fan-out: rules-engine path evaluates each (word, rule, mask)
                 * candidate against ALL nsalts_packed salts; matches slab semantics
                 * at gpujob_opencl.c slab arm (g->count × nsalts_packed × Maxiter).
                 * Pre-2026-05-09 this site omitted nsalts_packed and under-counted
                 * Tothash by the salt fan-out factor (~590K× for e31 MD5SALT real
                 * workloads). Clamp ≥1 for unsalted ops sharing this path. */
                uint32_t _nsalt_acct = (nsalts_packed > 0) ? (uint32_t)nsalts_packed : 1u;
                /* BF chunk-as-job (2026-05-10): for BF chunks the kernel iterates
                 * bf_num_masks per word, NOT the full gpu_mask_total. Using
                 * b71_mask_size_acct (= gpu_mask_total) over-counts Tothash by
                 * (gpu_mask_total / bf_num_masks). E.g., 8-digit BF with mspw=131072
                 * inflates by 763×. Substitute the per-word range when bf_chunk.
                 *
                 * BF Phase 1.8 (2026-05-10) — LOAD-BEARING: with inner_iter > 1,
                 * each lane processes inner_iter mask values, so the per-word
                 * range is bf_num_masks * inner_iter. Without this multiplier,
                 * Tothash undercounts by inner_iter× (Tranche 3 lesson 2 redux:
                 * a single missed multiplier in accounting silently corrupts
                 * the reported total hashes-tested metric). When inner_iter
                 * is 0 or 1, the multiplier degenerates to 1 — bit-identical
                 * to the Phase 1 accounting. */
                uint32_t _ii_acct = (g->bf_chunk && g->bf_inner_iter > 0u)
                                  ? g->bf_inner_iter : 1u;
                uint64_t _mask_size_for_acct = (g->bf_chunk && g->bf_num_masks > 0u)
                                             ? (uint64_t)g->bf_num_masks *
                                                 (uint64_t)_ii_acct
                                             : (uint64_t)b71_mask_size_acct;
                uint64_t _per_dispatch_hashes =
                    (uint64_t)g->packed_count *
                    (uint64_t)(gpu_rule_count > 0 ? gpu_rule_count : 1) *
                    _mask_size_for_acct *
                    (uint64_t)_nsalt_acct *
                    (uint64_t)Maxiter;
                hashcnt += _per_dispatch_hashes;
                if (my_slot < MAX_GPU_SLOTS) {
                    _gpu_rules_hashes[my_slot] += _per_dispatch_hashes;
                }
                if (hashcnt > 10000000 || found > 0) {
                    atomic_fetch_add(&Tothash, hashcnt);
                    atomic_fetch_add(&Totfound, found);
                    hashcnt = 0;
                    found = 0;
                }
                goto return_jobg;
            } /* end rules_engine */

            /* ---------------------------------------------------------------
             * B7.9 (2026-05-07): the prior "Existing packed dispatch — GPU
             * pre-expanded (word,rule) pairs" branch was retired with the
             * chokepoint pack at mdxfind.c. Every packed slot in the pool
             * is now flagged g->rules_engine = 1 (the rules-engine pack at
             * mdxfind.c:10613/10660 is the sole producer of g->packed=1
             * slots). The `if (g->rules_engine)` arm above handles every
             * dispatch; this `else` is structurally unreachable.
             *
             * Defensive abort: if a slot ever arrives with g->packed=1 and
             * g->rules_engine=0 the workload would silently lose cracks.
             * Loud-fail instead so the caller-site bug surfaces immediately.
             *
             * RCS history retains the prior packed-dispatch flow (chunked
             * 16M-word loop into gpu_opencl_dispatch_packed, NTLM/NTLMH
             * iconv hit-replay, packed_count*Maxiter hashcnt accounting). */
            fprintf(stderr,
                "BUG: gpujob_opencl_worker received packed slot with "
                "rules_engine=0 (op=%s(%d) packed_count=%u) — chokepoint "
                "pack was retired in B7.9; only rules-engine slots may "
                "arrive here. Aborting to surface caller bug.\n",
                gpu_op_name(g->op), g->op, g->packed_count);
            abort();
        }

        /* BF Phase 3b Tranche B (2026-05-10): slab dispatcher arm retired.
         * The slab arm (formerly ~445 LOC at rev 1.134) dispatched g->packed=0
         * jobs via gpu_opencl_dispatch_batch + per-position hit-replay decompose
         * for the legacy GPU_CAT_SALTED / SALTPASS / ITER / MASK paths. Sole
         * producer was gpu_try_pack in mdxfind.c, retired in same commit; sole
         * downstream was gpu_opencl_dispatch_batch in gpu/gpu_opencl.c, also
         * retired in same commit. All GPU traffic now flows through the rules-
         * engine path above (gpu_opencl_dispatch_md5_rules); the defensive
         * abort() at line 2429 (B7.9) already proved no packed=1/rules_engine=0
         * slots reach the worker. The slab arm is structurally unreachable in
         * the post-Tranche-B world. RCS history retains the prior implemen-
         * tation (gpu/gpujob_opencl.c rev 1.134). The bf_encode_23 helper at
         * gpu/gpujob_opencl.c:497-510 was kept (used by the rules-engine arm
         * at line ~2287 for BCRYPT hit-replay). Helper functions referenced
         * only from the deleted slab arm (gpu_opencl_set_mask_resume / set_
         * salt_resume / has_resume / last_mask_start, gpu/gpu_opencl.c:3913-
         * 3930) are now unreachable; their deletion is deferred to a follow-
         * up Tranche C to keep this commit focused on the dispatch-path
         * deletion (Tranche B). */

return_jobg:
        g->t_return_start = gpu_now_us();
        if (gpu_pipe_trace_enabled == 1 && g->t_dispatched != 0) {
            fprintf(stderr, "[pipe] g=%p disp_us=%llu dev=%d\n",
                    (void *)g,
                    (unsigned long long)(g->t_return_start - g->t_dispatched),
                    my_slot);
        }
        g->next = NULL;
        g->count = 0;
        g->passbuf_pos = 0;
        g->word_stride = 0;
        g->packed = 0;
        g->packed_count = 0;
        g->packed_pos = 0;
        g->rules_engine = 0;
        /* Tranche 1 BF chunk-as-job: keep additive plumbing inert. */
        g->bf_chunk = 0;
        g->bf_mask_start = 0;
        g->bf_offset_per_word = 0;
        g->bf_num_masks = 0;
        g->bf_inner_iter = 0;
        /* Phase 1.9 A1 (2026-05-10): clear fast-path eligibility on
         * slot return so the next acquirer must affirmatively re-arm. */
        g->bf_fast_eligible = 0;
        if (g->slot_kind == JOBG_KIND_RULES) {
            possess(GPURulesFreeWaiting);
            if (GPURulesFreeTail) {
                *GPURulesFreeTail = g;
                GPURulesFreeTail = &(g->next);
            } else {
                GPURulesFreeHead = g;
                GPURulesFreeTail = &(g->next);
            }
            twist(GPURulesFreeWaiting, BY, +1);
        } else {
            possess(GPULegacyFreeWaiting);
            if (GPULegacyFreeTail) {
                *GPULegacyFreeTail = g;
                GPULegacyFreeTail = &(g->next);
            } else {
                GPULegacyFreeHead = g;
                GPULegacyFreeTail = &(g->next);
            }
            twist(GPULegacyFreeWaiting, BY, +1);
        }
        if (gpu_pipe_trace_enabled == 1) {
            fprintf(stderr, "[pipe] g=%p ret_us=%llu dev=%d\n",
                    (void *)g,
                    (unsigned long long)(gpu_now_us() - g->t_return_start),
                    my_slot);
        }
    }

    if (hashcnt || found) {
        atomic_fetch_add(&Tothash, hashcnt);
        atomic_fetch_add(&Totfound, found);
    }

    free(outbuf);
    free(saltsnap);
    free(saltpool);
    free(salts_packed);
    free(soff);
    free(slen);
    free(pack_map);
}

int gpujob_init(int num_jobg) {
    if (!gpu_opencl_available()) return -1;

    /* Resolve pipeline-trace env var once. */
    gpu_pipe_trace_init();

    _max_salt_count = 0;
    _max_salt_bytes = 0;
    for (int sti = 0; sti < 2000; sti++) {
        if (Typesaltcnt[sti] > _max_salt_count)
            _max_salt_count = Typesaltcnt[sti];
        if (Typesaltbytes[sti] > _max_salt_bytes)
            _max_salt_bytes = Typesaltbytes[sti];
    }
    if (_max_salt_count < 1024) _max_salt_count = 1024;
    if (_max_salt_bytes < 8192) _max_salt_bytes = 8192;

    for (int di = 0; di < gpu_opencl_num_devices(); di++)
        gpu_opencl_set_max_iter(di, Maxiter);

    /* One gpujob thread per GPU device; compute min batch limit */
    _gpujob_count = gpu_opencl_num_devices();
    if (_gpujob_count < 1) _gpujob_count = 1;

    _gpu_batch_max = GPUBATCH_RULE_MAX;
    for (int i = 0; i < _gpujob_count; i++) {
        int mb = gpu_opencl_max_batch(i);
        if (mb < _gpu_batch_max) _gpu_batch_max = mb;
    }

    /* Hard-stop: confirm the rules-engine per-slot sizing can accommodate
     * the rule count before any worker can attempt to pack. The rules-engine
     * lane formula at mdxfind.c:9754 clamps required entries to
     * GPU_RULES_MAX_WORDS_PER_BATCH, so this is structurally unreachable
     * today — but keeps the invariant honest if the lane target ever
     * changes.
     *
     * Synthetic-only path (gpu_rule_count == 1, Numrules == 0): the lane
     * heuristic doesn't apply — per-batch is a flat 16K md5(word) probes
     * with no Cartesian product. Skip the gate; throughput comes from
     * dispatch rate, not per-dispatch lane count. */
    if (gpu_rule_count > 1) {
        uint32_t need = 1000000u / (uint32_t)gpu_rule_count;
        if (need < 64) need = 64;
        if (need > (uint32_t)GPU_RULES_MAX_WORDS_PER_BATCH) {
            /* Sub-optimal lane utilization but not a crash condition. Warn
             * once and proceed with the compile-time max as the effective
             * words-per-batch. User can recompile with a bigger
             * GPU_RULES_MAX_WORDS_PER_BATCH for full perf if they care.
             * Pre-2026-05-10 this was a fatal -1 that knocked small-rule-count
             * (-r with 2..61 GPU-eligible rules) workloads to CPU-only. See
             * project_gpu_rules_engine_small_count_gate.md. */
            fprintf(stderr,
                    "WARNING: GPU rule engine running below lane target "
                    "(gpu_rule_count=%d, words/batch=%u). "
                    "Recompile with larger GPU_RULES_MAX_WORDS_PER_BATCH "
                    "for full performance.\n",
                    gpu_rule_count, (uint32_t)GPU_RULES_MAX_WORDS_PER_BATCH);
            need = (uint32_t)GPU_RULES_MAX_WORDS_PER_BATCH;
            (void)need;  /* clamp not consumed below; documented intent only */
        }
    }

    /* Pool sizing: legacy pool always full size; rules pool only when
     * the rules engine is active. Both get the same slot count so
     * neither path can starve due to the other. */
    int n_legacy = num_jobg;
    int n_rules  = (gpu_rule_count > 0) ? num_jobg : 0;
    _num_legacy_jobg = n_legacy;
    _num_rules_jobg  = n_rules;

    GPUWorkWaiting = new_lock(0);
    GPULegacyFreeWaiting = new_lock(n_legacy);
    GPURulesFreeWaiting  = new_lock(n_rules);
    GPUWorkTail = &GPUWorkHead;

    for (int i = 0; i < n_legacy; i++) {
        struct jobg *g = (struct jobg *)malloc_lock(sizeof(struct jobg), "jobg");
        g->packed_buf = NULL;
        g->word_offset = NULL;
        g->packed = 0;
        g->packed_count = 0;
        g->packed_pos = 0;
        g->rules_engine = 0;
        /* Tranche 1 BF chunk-as-job: keep additive plumbing inert. */
        g->bf_chunk = 0;
        g->bf_mask_start = 0;
        g->bf_offset_per_word = 0;
        g->bf_num_masks = 0;
        g->bf_inner_iter = 0;
        /* Phase 1.9 A1 (2026-05-10): init fast-path eligibility on
         * legacy-pool slot creation. */
        g->bf_fast_eligible = 0;
        g->slot_kind = JOBG_KIND_LEGACY;
        g->packed_buf_size = (size_t)GPUBATCH_PACKED_SIZE;
        g->word_offset_entries = (uint32_t)(GPUBATCH_PACKED_SIZE / 2);
        if (GPULegacyFreeTail) {
            *GPULegacyFreeTail = g;
            GPULegacyFreeTail = &(g->next);
        } else {
            GPULegacyFreeHead = g;
            GPULegacyFreeTail = &(g->next);
        }
    }

    /* Phase D: slot pool allocation timing. Rules-engine pool only —
     * legacy 160 x 128 MB slot allocation is intentionally NOT timed
     * here (per Shooter design brief). The rules-engine slots' actual
     * packed_buf allocations happen lazily on first use via
     * malloc_pinned (mdxfind.c:9909/9944) — those go through the
     * pin tracker. This phase just times jobg-struct allocation. */
    /* slot pool BEGIN stderr emit retired 2026-05-09 — zero-time progress noise. */
    for (int i = 0; i < n_rules; i++) {
        struct jobg *g = (struct jobg *)malloc_lock(sizeof(struct jobg), "jobg");
        g->packed_buf = NULL;
        g->word_offset = NULL;
        g->packed = 0;
        g->packed_count = 0;
        g->packed_pos = 0;
        g->rules_engine = 0;
        /* Tranche 1 BF chunk-as-job: keep additive plumbing inert. */
        g->bf_chunk = 0;
        g->bf_mask_start = 0;
        g->bf_offset_per_word = 0;
        g->bf_num_masks = 0;
        g->bf_inner_iter = 0;
        /* Phase 1.9 A1 (2026-05-10): init fast-path eligibility on
         * rules-pool slot creation. */
        g->bf_fast_eligible = 0;
        g->slot_kind = JOBG_KIND_RULES;
        g->packed_buf_size = (size_t)GPUBATCH_RULES_PACKED_SIZE;
        g->word_offset_entries = (uint32_t)GPU_RULES_MAX_WORDS_PER_BATCH;
        if (GPURulesFreeTail) {
            *GPURulesFreeTail = g;
            GPURulesFreeTail = &(g->next);
        } else {
            GPURulesFreeHead = g;
            GPURulesFreeTail = &(g->next);
        }
    }
    /* slot pool DONE stderr emit retired 2026-05-09 — paired with BEGIN above. */

    /* Capture per-device PCIe LnkSta at startup (Linux only; silent on macOS). */
    for (int i = 0; i < _gpujob_count && i < MAX_GPU_SLOTS; i++) {
        char bdf[32] = "";
        gpu_opencl_device_bdf(i, bdf, sizeof(bdf));
        gpu_capture_lnksta(i, bdf, _gpu_lnksta_start[i], sizeof(_gpu_lnksta_start[i]));
    }

    /* Spawn one worker per non-disabled device. Disabled devices
     * (compact table didn't fit; see gpu_opencl_device_disabled doc)
     * get NO worker — words must never be routed to them, and a worker
     * tight-looping on dispatch_*-returns-NULL would consume queue
     * slots and silently drop work (data integrity bug on multi-GPU
     * partial-disable rigs). _n_workers counts what we actually
     * spawned so shutdown sends the right sentinel count.
     * _gpujob_count stays as device count for stats-array indexing. */
    int n_workers = 0;
    for (int i = 0; i < _gpujob_count; i++) {
        if (gpu_opencl_device_disabled(i)) {
            fprintf(stderr, "OpenCL GPU[%d]: gpujob worker NOT spawned (device disabled)\n", i);
            continue;
        }
        launch(gpujob, (void *)(intptr_t)i);
        n_workers++;
    }
    _n_gpujob_workers = n_workers;
    _gpujob_ready = 1;
    if (n_rules > 0)
        tsfprintf(stderr, "OpenCL GPU: %d gpujob thread%s started (%d legacy + %d rules batch buffers)\n",
                n_workers, n_workers > 1 ? "s" : "", n_legacy, n_rules);
    else
        tsfprintf(stderr, "OpenCL GPU: %d gpujob thread%s started (%d batch buffers)\n",
                n_workers, n_workers > 1 ? "s" : "", n_legacy);
    /* Pin summary is deferred to first comfort-line in mdxfind.c
     * ReportStats — by that point every malloc_pinned call site
     * (lazy chokepoint fills, hashes_shown alloc, etc.) has run, so
     * the summary is comprehensive. Emitting here would yield zero
     * because all pinning is lazy from this point forward. */
    if (n_workers == 0) {
        /* All devices disabled. Caller's gpu_avail check should already
         * have prevented this via gpu_opencl_finalize_active_count flipping
         * ocl_ready=0, but if we somehow got here with zero active
         * devices, signal init failure so mdxfind takes the no-GPU path. */
        fprintf(stderr, "OpenCL GPU: no workers spawned (all devices disabled) — gpujob_init failing\n");
        _gpujob_ready = 0;
        return -1;
    }
    return 0;
}

void gpujob_shutdown(void) {
    if (!_gpujob_ready) return;

    /* Wait for GPU work queue to drain — procjob threads may still be
     * flushing partial JOBGs. Wait until all batch buffers in both pools
     * are returned to their free lists (meaning no work is in flight). */
    possess(GPULegacyFreeWaiting);
    wait_for(GPULegacyFreeWaiting, TO_BE, _num_legacy_jobg);
    release(GPULegacyFreeWaiting);
    if (_num_rules_jobg > 0) {
        possess(GPURulesFreeWaiting);
        wait_for(GPURulesFreeWaiting, TO_BE, _num_rules_jobg);
        release(GPURulesFreeWaiting);
    }

    /* One sentinel per spawned worker (NOT per device — disabled-device
     * workers were never spawned in gpujob_init). Sending too many would
     * leave dangling sentinels on the queue; sending too few would leave
     * a worker waiting forever. */
    for (int i = 0; i < _n_gpujob_workers; i++) {
        struct jobg *sentinel = gpujob_get_free(NULL, 0);
        sentinel->op = 2000;
        sentinel->count = 0;
        gpujob_submit(sentinel);
    }
    _gpujob_ready = 0;

    /* Capture per-device PCIe LnkSta at shutdown for delta vs startup. */
    for (int i = 0; i < _gpujob_count && i < MAX_GPU_SLOTS; i++) {
        char bdf[32] = "";
        gpu_opencl_device_bdf(i, bdf, sizeof(bdf));
        gpu_capture_lnksta(i, bdf, _gpu_lnksta_end[i], sizeof(_gpu_lnksta_end[i]));
    }

    /* Report per-GPU dispatch statistics */
    for (int i = 0; i < _gpujob_count && i < MAX_GPU_SLOTS; i++) {
        uint64_t units = _gpu_words[i] + _gpu_legacy_ent[i] + _gpu_slab_hashes[i];
        if (!(_gpu_batches[i] || units)) continue;
        const char *dname = gpu_opencl_device_name(i);
        char bdf[32] = "";
        gpu_opencl_device_bdf(i, bdf, sizeof(bdf));
        uint64_t wall_us = (_gpu_last_us[i] && _gpu_first_us[i] && _gpu_last_us[i] >= _gpu_first_us[i])
                          ? (_gpu_last_us[i] - _gpu_first_us[i]) : 0;
        uint64_t busy_us = _gpu_busy_us[i];
        uint64_t idle_us = (wall_us > busy_us) ? (wall_us - busy_us) : 0;
        double idle_pct = wall_us > 0 ? (100.0 * (double)idle_us / (double)wall_us) : 0.0;
        /* Total hashes computed by this device, summed across paths:
         *   rules-engine: words × gpu_rule_count × Maxiter
         *   legacy:       entries × Maxiter   (each entry is one hash)
         *   slab:         hashes × Maxiter    (each is one hash, possibly iterated)
         * Prior single-counter code multiplied the entire mixed-unit count
         * by gpu_rule_count, producing a 2× over-count on rules+legacy
         * workloads (each word generated 1 word entry + 1 legacy duplicate,
         * both inflated by × 100k). Task #47. */
        int      rcount    = gpu_rule_count;
        int      iterc     = (Maxiter > 0) ? Maxiter : 1;
        /* Use _gpu_rules_hashes[] (candidate count, salt-axis correct) instead of
         * _gpu_words[i] × rcount × iterc which lost the salt fan-out for the
         * rules-engine path. Salt-axis fix 2026-05-09; see _gpu_rules_hashes
         * declaration comment. */
        uint64_t hashes    = _gpu_rules_hashes[i]
                           + (uint64_t)_gpu_legacy_ent[i] * (uint64_t)iterc
                           + (uint64_t)_gpu_slab_hashes[i] * (uint64_t)iterc;
        (void)rcount;  /* kept for share-line compat; rules hashes now self-contained */
        double word_mhps = busy_us > 0 ? ((double)units / (double)busy_us) : 0.0;
        double hash_ghps = busy_us > 0 ? ((double)hashes / (double)busy_us / 1e3) : 0.0;
        double h2d_GBps  = busy_us > 0 ? ((double)_gpu_h2d_bytes[i] / (double)busy_us / 1e3) : 0.0; /* B/us / 1e3 = GB/s */
        /* legacy/slab columns retired 2026-05-09 — all algorithms converted to
         * the unified template path; legacy + slab counters are dead. */
        fprintf(stderr,
                "OpenCL GPU[%d]: %s%s%s%s | %llu batches | %llu words | %llu hashes | %llu hits\n"
                "                wall=%.2fs busy=%.2fs idle=%.2fs (%.0f%%) | hash_Gh/s=%.3f  unit_Mh/s=%.2f | h2d_GBps=%.3f\n",
                i, dname,
                bdf[0] ? " [" : "", bdf[0] ? bdf : "", bdf[0] ? "]" : "",
                (unsigned long long)_gpu_batches[i],
                (unsigned long long)_gpu_words[i],
                (unsigned long long)hashes,
                (unsigned long long)_gpu_hits[i],
                wall_us / 1e6, busy_us / 1e6, idle_us / 1e6, idle_pct,
                hash_ghps, word_mhps, h2d_GBps);
        if (_gpu_lnksta_start[i][0] || _gpu_lnksta_end[i][0]) {
            fprintf(stderr, "                LnkSta start: %s\n"
                            "                LnkSta end:   %s\n",
                    _gpu_lnksta_start[i][0] ? _gpu_lnksta_start[i] : "(unknown)",
                    _gpu_lnksta_end[i][0]   ? _gpu_lnksta_end[i]   : "(unknown)");
        } else {
            fprintf(stderr, "                LnkSta: (BDF unknown or lspci unavailable)\n");
        }
    }
    if (gpu_dispatch_trace_fp && gpu_dispatch_trace_fp != stderr) {
        fclose(gpu_dispatch_trace_fp);
        gpu_dispatch_trace_fp = NULL;
    }
}

/* Emit a one-line per-device dispatch share view for multi-GPU runs.
 * Format:    GPU share: [0]=56.1% [1]=43.9%
 * Skips devices with zero dispatches. For >8 active devices, abbreviates
 * to top-4 + "..." + bottom-1 (by dispatch count) so the line stays
 * readable on an 80/120-column stderr. Called from mdxfind.c just before
 * the "Done - N threads caught" wrap-up banner. Safe to call after
 * gpujob_shutdown(); reads only the static counters. */
/* Per-device hash count for share-line accounting. Mirrors the formula
 * in the per-device print: rules-engine hashes + legacy hashes + slab
 * hashes. The share fraction reflects work-done, not just dispatch
 * count, so a device that ran ten rules-engine batches is correctly
 * shown as having done much more work than one that ran ten legacy
 * single-hash dispatches. */
static uint64_t _gpu_device_hashes(int i) {
    int      iterc  = (Maxiter > 0) ? Maxiter : 1;
    /* Mirrors hashes formula in gpujob_print_summary (salt-axis-correct via
     * _gpu_rules_hashes[]; 2026-05-09 salt-axis fix). */
    return _gpu_rules_hashes[i]
         + (uint64_t)_gpu_legacy_ent[i] * (uint64_t)iterc
         + (uint64_t)_gpu_slab_hashes[i] * (uint64_t)iterc;
}

void gpujob_print_share_line(FILE *fp) {
    if (!fp) return;
    /* Sum all active devices first */
    uint64_t total = 0;
    int active_idx[MAX_GPU_SLOTS];
    int n_active = 0;
    uint64_t per[MAX_GPU_SLOTS] = {0};
    for (int i = 0; i < _gpujob_count && i < MAX_GPU_SLOTS; i++) {
        per[i] = _gpu_device_hashes(i);
        if (per[i] == 0) continue;
        active_idx[n_active++] = i;
        total += per[i];
    }
    if (n_active <= 1 || total == 0) return;  /* single-GPU has nothing useful to show */

    if (n_active <= 8) {
        fprintf(fp, "GPU share:");
        for (int k = 0; k < n_active; k++) {
            int i = active_idx[k];
            double pct = 100.0 * (double)per[i] / (double)total;
            fprintf(fp, " [%d]=%.1f%%", i, pct);
        }
        fprintf(fp, "\n");
    } else {
        /* Sort active_idx by per-device hashes descending — simple
         * insertion sort, n is tiny (<=64). */
        int sorted[MAX_GPU_SLOTS];
        for (int k = 0; k < n_active; k++) sorted[k] = active_idx[k];
        for (int k = 1; k < n_active; k++) {
            int v = sorted[k];
            uint64_t vd = per[v];
            int j = k - 1;
            while (j >= 0 && per[sorted[j]] < vd) {
                sorted[j+1] = sorted[j];
                j--;
            }
            sorted[j+1] = v;
        }
        fprintf(fp, "GPU share:");
        for (int k = 0; k < 4; k++) {
            int i = sorted[k];
            double pct = 100.0 * (double)per[i] / (double)total;
            fprintf(fp, " [%d]=%.1f%%", i, pct);
        }
        fprintf(fp, " ...");
        int last = sorted[n_active - 1];
        double pct_last = 100.0 * (double)per[last] / (double)total;
        fprintf(fp, " [%d]=%.1f%%\n", last, pct_last);
    }
}

/* Internal: pull a slot from the free-list identified by `kind`. The
 * `filename` and `startline` args are part of the public API (Metal's
 * priority scheduler in gpujob_metal.m USES startline to prefer earlier
 * lines), but the OpenCL backend has a single shared work queue and
 * doesn't reorder, so they're ignored here. Keeping the parameters
 * preserves the cross-backend ABI. */
static struct jobg *_gpujob_get_free_kind(char *filename, unsigned long long startline,
                                          enum jobg_kind kind) {
    (void)filename;
    (void)startline;

    if (MDXpause) {
        __sync_fetch_and_add(&MDXpaused_count, 1);
#ifdef _WIN32
        while (MDXpause) Sleep(2000);
#else
        while (MDXpause) sleep(2);
#endif
        __sync_fetch_and_sub(&MDXpaused_count, 1);
    }
    {
        lock *waiter           = (kind == JOBG_KIND_RULES) ? GPURulesFreeWaiting  : GPULegacyFreeWaiting;
        struct jobg **headp    = (kind == JOBG_KIND_RULES) ? &GPURulesFreeHead    : &GPULegacyFreeHead;
        struct jobg ***tailpp  = (kind == JOBG_KIND_RULES) ? &GPURulesFreeTail    : &GPULegacyFreeTail;
        possess(waiter);
        __sync_fetch_and_add(&MDXpaused_count, 1);
        wait_for(waiter, NOT_TO_BE, 0);
        __sync_fetch_and_sub(&MDXpaused_count, 1);
        struct jobg *g = *headp;
        *headp = g->next;
        g->next = NULL;
        if (*headp == NULL)
            *tailpp = headp;
        twist(waiter, BY, -1);
        g->count = 0;
        g->passbuf_pos = 0;
        g->t_acquired = gpu_now_us();   /* pipeline-trace start */
        return g;
    }
}

struct jobg *gpujob_get_free(char *filename, unsigned long long startline) {
    return _gpujob_get_free_kind(filename, startline, JOBG_KIND_LEGACY);
}

struct jobg *gpujob_get_free_rules(char *filename, unsigned long long startline) {
    if (_num_rules_jobg == 0) return NULL;
    return _gpujob_get_free_kind(filename, startline, JOBG_KIND_RULES);
}

void gpujob_submit(struct jobg *g) {
    g->t_added = gpu_now_us();
    if (gpu_pipe_trace_enabled == 1 && g->t_acquired != 0) {
        fprintf(stderr, "[pipe] g=%p fill_us=%llu pc=%u\n",
                (void *)g,
                (unsigned long long)(g->t_added - g->t_acquired),
                g->packed_count);
    }
    g->next = NULL;
    possess(GPUWorkWaiting);
    *GPUWorkTail = g;
    GPUWorkTail = &(g->next);
    twist(GPUWorkWaiting, BY, +1);
}

/* Return a jobg buffer to its origin free list without submitting it for work. */
void gpujob_return_free(struct jobg *g) {
    g->next = NULL;
    g->count = 0;
    g->passbuf_pos = 0;
    g->packed = 0;
    g->packed_count = 0;
    g->packed_pos = 0;
    g->rules_engine = 0;
    /* Tranche 1 BF chunk-as-job: keep additive plumbing inert. */
    g->bf_chunk = 0;
    g->bf_mask_start = 0;
    g->bf_offset_per_word = 0;
    g->bf_num_masks = 0;
    g->bf_inner_iter = 0;
    /* Phase 1.9 A1 (2026-05-10): clear fast-path eligibility on slot
     * return-free; next acquirer must affirmatively re-arm. */
    g->bf_fast_eligible = 0;
    if (g->slot_kind == JOBG_KIND_RULES) {
        possess(GPURulesFreeWaiting);
        if (GPURulesFreeTail) {
            *GPURulesFreeTail = g;
            GPURulesFreeTail = &(g->next);
        } else {
            GPURulesFreeHead = g;
            GPURulesFreeTail = &(g->next);
        }
        twist(GPURulesFreeWaiting, BY, +1);
    } else {
        possess(GPULegacyFreeWaiting);
        if (GPULegacyFreeTail) {
            *GPULegacyFreeTail = g;
            GPULegacyFreeTail = &(g->next);
        } else {
            GPULegacyFreeHead = g;
            GPULegacyFreeTail = &(g->next);
        }
        twist(GPULegacyFreeWaiting, BY, +1);
    }
}

/* Non-blocking version: returns NULL immediately if no free legacy buffer.
 * Used by hybrid types where CPU fallback is preferred over waiting; only
 * the legacy pool is relevant for those callers. */
struct jobg *gpujob_try_get_free(void) {
    if (!_gpujob_ready) return NULL;
    possess(GPULegacyFreeWaiting);
    if (peek_lock(GPULegacyFreeWaiting) == 0) {
        release(GPULegacyFreeWaiting);
        return NULL;
    }
    struct jobg *g = GPULegacyFreeHead;
    GPULegacyFreeHead = g->next;
    g->next = NULL;
    if (GPULegacyFreeHead == NULL)
        GPULegacyFreeTail = &GPULegacyFreeHead;
    twist(GPULegacyFreeWaiting, BY, -1);
    g->count = 0;
    g->passbuf_pos = 0;
    return g;
}

int gpujob_available(void) {
    return _gpujob_ready;
}

int gpujob_batch_max(void) {
    return _gpu_batch_max;
}

int gpujob_queue_depth(void) {
    if (!_gpujob_ready) return 0;
    return (int)peek_lock(GPUWorkWaiting);
}

int gpujob_free_count(void) {
    if (!_gpujob_ready) return 0;
    return (int)peek_lock(GPULegacyFreeWaiting)
         + (int)peek_lock(GPURulesFreeWaiting);
}

int gpu_op_category(int op) {
    switch (op) {
    /* Salted MD5 variants -- GPU iterates over salts.
     * B6 salt-axis (2026-05-06; §11 row 25): JOB_MD5SALT and
     * JOB_MD5SALTPASS MOVED to GPU_CAT_MASK below — they're now
     * template-routed via gpu_template.cl + the salted variant cores
     * (gpu_md5salt_core.cl / gpu_md5saltpass_core.cl). The slab path's
     * gpu_try_pack at line 635 gates on GPU_CAT_SALTED — once moved,
     * the slab kernel md5salt_batch is structurally unreachable for
     * these two ops (B8 slab-retirement candidate). The other three
     * MD5SALT-family variants remain in GPU_CAT_SALTED until they
     * get template cores. */
    /* B6.6 (2026-05-06): JOB_MD5UCSALT / JOB_MD5revMD5SALT / JOB_MD5sub8_24SALT
     * MOVED out of GPU_CAT_SALTED — these 3 share the SAME GPU template kernel
     * as e31 MD5SALT via params.algo_mode runtime flag. Slab kernel md5salt_batch
     * (which served all 4 via the slab path's variant switch at gpu_md5salt.cl)
     * is structurally unreachable for these 4 ops post-this-change; B8 candidate. */
    /* Salt + raw password.
     * B6 salt-axis: JOB_MD5SALTPASS MOVED to GPU_CAT_MASK below.
     * B6.1 SHA1 fan-out (2026-05-06): JOB_SHA1SALTPASS MOVED to GPU_CAT_MASK
     * below — first SHA-family salted variant on the unified template path.
     * B6.2 SHA256 fan-out (2026-05-06): JOB_SHA256SALTPASS MOVED to
     * GPU_CAT_MASK below — second SHA-family salted variant. The slab
     * kernel sha256saltpass_batch (gpu_sha256.cl:75) is structurally
     * unreachable post-move (gpu_op_category returns GPU_CAT_MASK,
     * gpu_try_pack returns 0 for non-GPU_CAT_SALTED/SALTPASS ops);
     * candidate for B8 retirement.
     * B6.4 MD5PASSSALT fan-out (2026-05-06): JOB_MD5PASSSALT MOVED to
     * GPU_CAT_MASK below — first APPEND-shape salted variant on the
     * unified template path (gpu_template.cl + gpu_md5passsalt_core.cl).
     * The slab kernel md5passsalt_batch (gpu_md5saltpass.cl:97) is
     * structurally unreachable post-move — gpu_op_category returns
     * GPU_CAT_MASK, gpu_try_pack returns 0 for non-GPU_CAT_SALTED/
     * SALTPASS ops. Candidate for B8 slab-retirement alongside
     * md5saltpass_batch / sha256saltpass_batch / sha1saltpass_batch.
     * B6.5 SHA1PASSSALT fan-out (2026-05-06): JOB_SHA1PASSSALT MOVED to
     * GPU_CAT_MASK below — first SHA-family APPEND-shape salted variant
     * on the unified template path (gpu_template.cl + gpu_sha1passsalt_-
     * core.cl). The slab kernel sha1passsalt_batch (gpu_sha1.cl) is
     * structurally unreachable post-move; B8 retirement candidate
     * alongside the other salted slab kernels.
     * B6.7 SHA256PASSSALT fan-out (2026-05-06): JOB_SHA256PASSSALT MOVED to
     * GPU_CAT_MASK below — second SHA-family APPEND-shape salted variant
     * on the unified template path (gpu_template.cl + gpu_sha256passsalt_-
     * core.cl). Pure spec reuse — both the main template
     * (sha256_style_salted.cl.tmpl) and finalize fragment
     * (finalize_append_be.cl.frag) were already shipped at B6.2 / B6.5.
     * The slab kernel sha256passsalt_batch is structurally unreachable
     * post-move; B8 retirement candidate alongside the other salted
     * slab kernels. */
    /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455) MOVED to GPU_CAT_MASK
     * below — routes through the hand-written Path A salted-template kernel
     * (gpu_phpbb3_core.cl) via the unified template path. The iterated MD5
     * chain runs INSIDE template_finalize; max_iter=1 forced host-side at
     * the rules-engine pack site so the kernel's outer iter loop runs
     * exactly once and only the FINAL state is probed (matches CPU
     * semantics at mdxfind.c:13620 which has ONE checkhashbb call after
     * the inner for-loop). The slab kernel phpbb3_batch (gpu_phpbb3.cl)
     * is RETIRED in this same commit (whole-file retirement; FAM_PHPBB3
     * had only this one live kernel). probe_max_dispatch is anchored on
     * FAM_MD5SALT (hmac_md5_ksalt_batch), not FAM_PHPBB3, so retirement
     * does not require a probe migration. Templated count delta: 55 -> 56.
     * First iterated-crypt with salt-carried iter count on the unified
     * template path. */
    /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511) MOVED to GPU_CAT_MASK
     * below — routes through the hand-written Path A salted-template kernel
     * (gpu_md5crypt_core.cl) via the unified template path. The iterated MD5
     * chain (FIXED 1000 iters) runs INSIDE template_finalize; max_iter=1
     * forced host-side at the rules-engine pack site so the kernel's outer
     * iter loop runs exactly once and only the FINAL state is probed
     * (matches CPU semantics at mdxfind.c:13071 which has ONE hybrid_check
     * call after the inner for-loop). The slab kernel md5crypt_batch
     * (gpu_md5crypt.cl) is RETIRED in this same commit (whole-file
     * retirement; FAM_MD5CRYPT had only this one live kernel).
     * probe_max_dispatch is anchored on FAM_MD5SALT (hmac_md5_ksalt_batch),
     * not FAM_MD5CRYPT, so retirement does not require a probe migration.
     * Templated count delta: 56 -> 57. Phase 1 of the Unix-crypt ladder. */
    case JOB_DESCRYPT:
    /* Family A (2026-05-07): JOB_HMAC_MD5 (e214) + JOB_HMAC_MD5_KPASS
     * (e792) MOVED to GPU_CAT_MASK below — share the MD5SALT GPU template
     * kernel via params.algo_mode = 5 / 6. The slab kernels hmac_md5_-
     * ksalt_batch + hmac_md5_kpass_batch (gpu_md5salt.cl:33,154) are
     * RETAINED as PERMANENT capacity-probe infrastructure — they serve
     * as the known-answer anchor for probe_max_dispatch (gpu_opencl.c:
     * 2291-2471), which sweeps salt counts up to 4M to find the largest
     * NDRange that doesn't drop work-items (critical for Mali's silent
     * 17-bit dispatch limit and similar embedded-GPU corner cases). For
     * actual op dispatch these kernels are STRUCTURALLY UNREACHABLE
     * (gpu_op_category returns GPU_CAT_MASK; gpu_try_pack returns 0 for
     * non-GPU_CAT_SALTED/SALTPASS); the kernel_map[] entries exist for
     * the probe path only. Architect 2026-05-09 (memo task #214) confirmed
     * the probe needs a real-hash known-answer signal: stub kernels weaken
     * the correctness model (compute corruption can pass with synthetic
     * hits=32); template kernels compile lazily and would add ~10×
     * eager-compile cost to device init; static clGetDeviceInfo caps
     * regress on Mali. NOT a future retirement candidate. */
    /* Family B (2026-05-07): JOB_HMAC_SHA1 (e215) + JOB_HMAC_SHA1_KPASS
     * (e793) MOVED to GPU_CAT_MASK below — share the SHA1SALTPASS GPU
     * template kernel via params.algo_mode = 5 / 6. The slab kernels
     * hmac_sha1_ksalt_batch + hmac_sha1_kpass_batch (gpu_sha1.cl) are
     * RETIRED in this same commit (probe_max_dispatch is anchored on
     * FAM_MD5SALT, not FAM_SHA1, so retirement does not require a
     * probe migration). */
    /* Family C (2026-05-07): JOB_HMAC_SHA224 (e216) + JOB_HMAC_SHA224_KPASS
     * (e794) MOVED to GPU_CAT_MASK below — share the SHA224SALTPASS GPU
     * template kernel via params.algo_mode = 5 / 6. The slab kernels
     * hmac_sha224_ksalt_batch + hmac_sha224_kpass_batch (gpu_sha256.cl)
     * are RETIRED in this same commit (probe_max_dispatch is anchored
     * on FAM_MD5SALT, not FAM_SHA256, so retirement does not require a
     * probe migration). FAM_SHA256 remains live for SHA256 HMAC kernels
     * (Family D scope). */
    /* Family D (2026-05-08): JOB_HMAC_SHA256 (e217) + JOB_HMAC_SHA256_KPASS
     * (e795) MOVED to GPU_CAT_MASK below — share the SHA256SALTPASS GPU
     * template kernel via params.algo_mode = 5 / 6. The slab kernels
     * hmac_sha256_ksalt_batch + hmac_sha256_kpass_batch (gpu_sha256.cl)
     * are RETIRED in this same commit. The 2026-05-07 first attempt was
     * ABORTED because the HMAC body was authored with `#if HASH_WORDS == 8`
     * which produced wrong PTX on Pascal NVIDIA at packed_count > 1.
     * The runtime-gate form `if (HASH_WORDS == 8 && algo_mode >= 5u)`
     * (matching Families B/C/E/F/G/H/J/K convention) ships clean.
     * Diagnostic agent a97e0c9ac7747151e (2026-05-08) localized + verified
     * the fix. Final HMAC family in the ladder; HMAC ladder COMPLETE
     * 21/21 algos shipped. probe_max_dispatch is anchored on FAM_MD5SALT
     * (hmac_md5_ksalt_batch), not FAM_SHA256, so retirement does not
     * require a probe migration. After this MOVE + retirement,
     * gpu_sha256.cl has zero live `__kernel` decls (sole remaining
     * content is retirement-record comments) and is whole-file deleted
     * in this same commit (mirrors Family J pattern). */
    /* B6.9 SHA512 fan-out (2026-05-06): JOB_SHA512SALTPASS MOVED to
     * GPU_CAT_MASK below — first 64-bit-state salted variant on the
     * unified template path (gpu_template.cl + gpu_sha512saltpass_-
     * core.cl).
     * B6.10 SHA512PASSSALT fan-out (2026-05-06): JOB_SHA512PASSSALT
     * MOVED to GPU_CAT_MASK below — second 64-bit-state salted variant
     * (APPEND-shape sibling of SHA512SALTPASS). FINAL B6 ladder step.
     * gpu_op_category returns GPU_CAT_MASK for both ops post-these-
     * moves; gpu_try_pack returns 0 for non-GPU_CAT_SALTED/SALTPASS
     * ops, so any legacy slab dispatch for SHA-512-family salted ops
     * is structurally unreachable. B8 retirement candidate alongside
     * the other salted slab kernels (the entire SHA-512-family salted
     * slab dispatcher is now reachable for ZERO ops; both arms have
     * been moved out). */
    /* Family E HMAC-SHA384 carrier (2026-05-08): JOB_HMAC_SHA384 (e543) +
     * JOB_HMAC_SHA384_KPASS (e796) MOVED to GPU_CAT_MASK below — share
     * the SHA384SALTPASS-shaped carrier GPU template kernel via
     * params.algo_mode = 5 / 6. The slab kernels hmac_sha384_ksalt_batch
     * + hmac_sha384_kpass_batch (gpu_hmac_sha512.cl) are RETIRED in this
     * same commit. probe_max_dispatch is anchored on FAM_MD5SALT (hmac_-
     * md5_ksalt_batch), not FAM_HMAC_SHA512, so retirement does not
     * require a probe migration.
     * Family F (2026-05-08): JOB_HMAC_SHA512 (e218) + JOB_HMAC_SHA512_KPASS
     * (e797) MOVED to GPU_CAT_MASK below — share the SHA512SALTPASS GPU
     * template kernel via params.algo_mode = 5 / 6. The slab kernels
     * hmac_sha512_ksalt_batch + hmac_sha512_kpass_batch (gpu_hmac_sha512.cl)
     * are RETIRED in this same commit. After this MOVE, FAM_HMAC_SHA512
     * has zero live slab kernels; the entire gpu_hmac_sha512.cl file is
     * retired in the same commit. */
    /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): JOB_HMAC_RMD160 (e211)
     * + JOB_HMAC_RMD160_KPASS (e798) MOVED to GPU_CAT_MASK below — share
     * the RIPEMD160SALTPASS-shaped carrier GPU template kernel via
     * params.algo_mode = 5 / 6. The slab kernels hmac_rmd160_ksalt_batch
     * + hmac_rmd160_kpass_batch (gpu_hmac_rmd160.cl) are RETIRED in this
     * same commit. probe_max_dispatch is anchored on FAM_MD5SALT (hmac_-
     * md5_ksalt_batch), not FAM_HMAC_RMD160, so retirement does not
     * require a probe migration. After this MOVE, FAM_HMAC_RMD160 has
     * zero live slab kernels; the entire gpu_hmac_rmd160.cl file is
     * retired in the same commit. */
    /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): JOB_HMAC_RMD320 (e213)
     * + JOB_HMAC_RMD320_KPASS (e799) MOVED to GPU_CAT_MASK below — share
     * the RIPEMD320SALTPASS-shaped carrier GPU template kernel via
     * params.algo_mode = 5 / 6. The slab kernels hmac_rmd320_ksalt_batch
     * + hmac_rmd320_kpass_batch (gpu_hmac_rmd320.cl) are RETIRED in this
     * same commit (whole-file retirement; FAM_HMAC_RMD320 had only these
     * two live kernels). probe_max_dispatch is anchored on FAM_MD5SALT
     * (hmac_md5_ksalt_batch), not FAM_HMAC_RMD320, so retirement does
     * not require a probe migration. */
    /* Family I HMAC-BLAKE2S carrier (2026-05-08): JOB_HMAC_BLAKE2S (e828)
     * MOVED to GPU_CAT_MASK below — routes through the hand-written Path A
     * carrier GPU template kernel (gpu_hmac_blake2s_core.cl) via params.
     * algo_mode = 5. The slab kernel hmac_blake2s_kpass_batch
     * (gpu_hmac_blake2s.cl) is RETIRED in this same commit (whole-file
     * retirement; FAM_HMAC_BLAKE2S had only this one live kernel).
     * probe_max_dispatch is anchored on FAM_MD5SALT (hmac_md5_ksalt_batch),
     * not FAM_HMAC_BLAKE2S, so retirement does not require a probe
     * migration. */
    case JOB_BCRYPT:
    /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT (e512) MOVED to
     * GPU_CAT_MASK below — routes through the hand-written Path A salted-
     * template kernel (gpu_shacrypt_core.cl at HASH_WORDS=8) via the
     * unified template path. The 5-step SHA-crypt chain (default 5000
     * rounds; configurable via "rounds=N$" salt prefix decoded INSIDE
     * the kernel) runs INSIDE template_finalize; max_iter=1 forced host-
     * side at the rules-engine pack site so the kernel's outer iter loop
     * runs exactly once and only the FINAL state is probed (matches CPU
     * semantics at mdxfind.c:12290 -- ONE checkhash-equivalent call after
     * the inner for-loop). The slab kernel sha256crypt_batch
     * (gpu_sha256crypt.cl) is RETIRED in this same commit (whole-file
     * retirement; FAM_SHA256CRYPT had only this one live kernel).
     * probe_max_dispatch is anchored on FAM_MD5SALT (hmac_md5_ksalt_batch),
     * not FAM_SHA256CRYPT, so retirement does not require a probe
     * migration. Templated count delta: 57 -> 58. Phase 2 of the Unix-
     * crypt ladder. SHA512CRYPT + SHA512CRYPTMD5 stay in GPU_CAT_SALTPASS
     * for now; Phases 3 + 4 will MOVE them to GPU_CAT_MASK alongside this
     * SHA256CRYPT MOVE. */
    /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT (e513) MOVED to
     * GPU_CAT_MASK below -- routes through the hand-written Path A
     * salted-template kernel (gpu_shacrypt_core.cl at HASH_WORDS=16) via
     * the unified template path. The 5-step SHA-crypt chain (default
     * 5000 rounds; configurable via "rounds=N$" salt prefix decoded
     * INSIDE the kernel) runs INSIDE template_finalize; max_iter=1
     * forced host-side at the rules-engine pack site so the kernel's
     * outer iter loop runs exactly once and only the FINAL state is
     * probed (matches CPU semantics at mdxfind.c:12290 -- ONE checkhash-
     * equivalent call after the inner for-loop). Templated count delta:
     * 58 -> 59. Phase 3 of the Unix-crypt ladder. */
    /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_SHA512CRYPTMD5 (e510)
     * MOVED from GPU_CAT_SALTPASS to GPU_CAT_MASK below -- REUSES Phase
     * 3 SHA512CRYPT's compiled kernel (gpu_shacrypt_core.cl at HASH_-
     * WORDS=16; same BASE_ALGO=sha512crypt cache key) via the unified
     * template path. The MD5-preprocess of the password is performed
     * HOST-side at mdxfind.c:12256-12258 BEFORE gpu_try_pack -- the GPU
     * runs the IDENTICAL SHA-512 crypt chain as SHA512CRYPT, just over
     * a 32-byte ASCII-hex string instead of the original password.
     * After this MOVE, the slab kernel sha512crypt_batch (gpu_sha512-
     * crypt.cl) becomes structurally unreachable -- gpu_op_category
     * returns GPU_CAT_MASK for both SHA512CRYPT and SHA512CRYPTMD5;
     * gpu_try_pack returns 0 for non-GPU_CAT_SALTED/SALTPASS ops; the
     * sha512crypt slab is retired in this same commit (whole-file
     * deletion of gpu_sha512crypt.cl + metal_sha512crypt.metal +
     * their _str.h pairs per feedback_remove_retired_gpu_kernels.md;
     * RCS history retained at gpu/RCS/). probe_max_dispatch is
     * anchored on FAM_MD5SALT, not FAM_SHA512CRYPT, so retirement
     * does not require a probe migration. Templated count delta:
     * 59 -> 59 (kernel is shared with Phase 3, so no new template
     * slot consumed). Phase 4 of the Unix-crypt ladder (final phase;
     * Unix-crypt slab path fully retired). */
    /* B6.8 (2026-05-06): JOB_MD5_MD5SALTMD5PASS (e367) MOVED to
     * GPU_CAT_MASK below — shares the MD5SALT GPU kernel via
     * params.algo_mode=4. Salt-hex (saltsnap[].hashsalt) is packed
     * into salt_buf via gpu_pack_salts(use_hashsalt=1). The slab
     * kernel md5_md5saltmd5pass_batch (gpu_opencl.c:1219) is now
     * structurally unreachable post-move (gpu_op_category returns
     * GPU_CAT_MASK; gpu_try_pack returns 0 for non-GPU_CAT_SALTED/
     * SALTPASS ops); B8 retirement candidate. */
    /* B6.11 (2026-05-06): JOB_SHA1DRU (e404) MOVED to GPU_CAT_MASK below
     * — first 1M-iteration algorithm on the unified template path. The
     * 1M loop runs INSIDE template_finalize (gpu_sha1dru_core.cl); host
     * forces params.max_iter=1 at the rules-engine dispatch site (gpu_-
     * opencl.c, in the B1 payload pack block) so the kernel's outer iter
     * loop runs exactly once and only the FINAL state is probed (matches
     * CPU semantics at mdxfind.c:14261-14285). The slab kernel sha1dru_-
     * batch (gpu_sha1.cl) is removed in this same commit (B8 cleanup).
     * GPU_CAT_ITER is now unused by any live op — kept as defensive
     * defense-in-depth (the dispatcher arms at gpujob_opencl.c:1425 +
     * :1767 become dead code but harmless). */
    case JOB_MD5: case JOB_MD5UC:
    case JOB_MD4:
    case JOB_NTLMH:
    case JOB_NTLM:
    case JOB_MD4UTF16:  /* B5 sub-batch 8 (2026-05-05): MD4UTF16 GPU template path. */
    case JOB_SHA1:
    case JOB_SHA224:
    case JOB_SHA256:
    case JOB_SHA256RAW:
    case JOB_SHA384:
    case JOB_SHA512:
    case JOB_WRL:
    case JOB_MD6256:
    case JOB_KECCAK224: case JOB_KECCAK256: case JOB_KECCAK384: case JOB_KECCAK512:
    case JOB_SHA3_224: case JOB_SHA3_256: case JOB_SHA3_384: case JOB_SHA3_512:
    case JOB_SQL5: case JOB_SHA1RAW: case JOB_MD5RAW:
    /* B6.11 (2026-05-06): JOB_SHA1DRU joins as first 1M-iter algo on the
     * unified template path. 1M loop runs INSIDE template_finalize
     * (gpu_sha1dru_core.cl); host forces params.max_iter=1 so kernel's
     * outer iter loop runs exactly once. */
    case JOB_SHA1DRU:
    /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455) joins the unified
     * template path. Iterated MD5 chain (count = 1 << itoa64(salt[3]);
     * typically 128..2^30 iters) runs INSIDE template_finalize
     * (gpu_phpbb3_core.cl); host forces params.max_iter=1 so kernel's
     * outer iter loop runs exactly once and template_iterate (a stub)
     * is never called. Mirrors SHA1DRU pattern. Hit-replay arm in the
     * GPU_CAT_MASK rules-engine path uses checkhashbb (NOT
     * checkhashkey/checkhashsalt) — PHPBB3 has its own bb-specific
     * digest output format with phpitoa64-encoded 22-char hash. */
    case JOB_PHPBB3:
    /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511) joins the unified
     * template path. Iterated MD5 chain (FIXED 1000 iters per BSD $1$
     * md5crypt) runs INSIDE template_finalize (gpu_md5crypt_core.cl);
     * host forces params.max_iter=1 so kernel's outer iter loop runs
     * exactly once and template_iterate (a stub) is never called.
     * Mirrors PHPBB3 / SHA1DRU pattern. Hit-replay arm in the
     * GPU_CAT_MASK rules-engine path uses hybrid_check (16-byte binary
     * digest) + md5crypt_b64encode reconstruction (mirrors existing
     * slab arm at line 1723). Phase 1 of the Unix-crypt ladder. */
    case JOB_MD5CRYPT:
    /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT (e512) joins the
     * unified template path. 5-step SHA-crypt chain (default 5000 rounds;
     * configurable via "rounds=N$" salt prefix decoded INSIDE the kernel)
     * runs INSIDE template_finalize (gpu_shacrypt_core.cl at HASH_WORDS=8);
     * host forces params.max_iter=1 so kernel's outer iter loop runs
     * exactly once and template_iterate (a stub) is never called. Mirrors
     * MD5CRYPT pattern. Hit-replay arm in the GPU_CAT_MASK rules-engine
     * path uses hybrid_check (32-byte binary SHA-256 digest as 8 LE uint32
     * words) + sha256crypt_b64encode reconstruction (NEW helper added in
     * this same commit; ports the per-byte permutation table from
     * mdxfind.c:12753-12980). Phase 2 of the Unix-crypt ladder. */
    case JOB_SHA256CRYPT:
    /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT (e513) joins the
     * unified template path. 5-step SHA-crypt chain (default 5000 rounds;
     * configurable via "rounds=N$" salt prefix decoded INSIDE the kernel)
     * runs INSIDE template_finalize (gpu_shacrypt_core.cl at HASH_WORDS=16
     * -- shares the SAME core source with Phase 2 SHA256CRYPT, which uses
     * HASH_WORDS=8); host forces params.max_iter=1 so kernel's outer iter
     * loop runs exactly once and template_iterate (a stub) is never
     * called. Mirrors SHA256CRYPT pattern. Hit-replay arm in the GPU_-
     * CAT_MASK rules-engine path uses hybrid_check (64-byte binary SHA-
     * 512 digest as 16 LE uint32 words) + sha512crypt_b64encode
     * reconstruction (NEW helper added in this same commit; ports the
     * per-byte permutation table from mdxfind.c:12361-12780, which
     * matches the compact-table loader's permutation at mdxfind.c:
     * 37036-37041). SHA512CRYPTMD5 (e510) joins this same arm in
     * Phase 4 (the `case JOB_SHA512CRYPTMD5:` immediately below) --
     * REUSES the same compiled SHA512CRYPT kernel (host-side MD5-pre-
     * process of the password at mdxfind.c:12256-12258 makes algo_-
     * mode=1 unnecessary). Phase 3 of the Unix-crypt ladder. */
    case JOB_SHA512CRYPT:
    /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_SHA512CRYPTMD5 (e510)
     * joins the unified template path. REUSES Phase 3 SHA512CRYPT's
     * compiled kernel (gpu_shacrypt_core.cl at HASH_WORDS=16) -- the
     * MD5-preprocess of the password is performed HOST-side at
     * mdxfind.c:12256-12258 BEFORE gpu_try_pack (job->pass swapped
     * with 32-char MD5 hex of the original password); the GPU sees a
     * 32-byte ASCII-hex string and runs the IDENTICAL SHA-512 crypt
     * chain. Hit-replay arm (NEW in this commit) reuses sha512crypt_-
     * b64encode and probes Typesalt[JOB_SHA512CRYPTMD5] -- which
     * carries the SAME "$6$[rounds=N$]<salt>$" prefix as Typesalt[JOB_-
     * SHA512CRYPT] (mdxfind.c:47077-47087 inserts the same line into
     * BOTH Judy arrays). Phase 4 of the Unix-crypt ladder (final
     * phase; Unix-crypt slab path fully retired). */
    case JOB_SHA512CRYPTMD5:
    case JOB_SHA384RAW: case JOB_SHA512RAW:
    case JOB_MYSQL3:
    case JOB_STREEBOG_32: case JOB_STREEBOG_64:
    case JOB_RMD160:
    case JOB_RMD320:  /* B5 sub-batch 2 (2026-05-05): RIPEMD-320 unsalted GPU. */
    case JOB_BLAKE2S256:
    /* B6 salt-axis (2026-05-06; §11 row 25): MOVED from GPU_CAT_SALTED /
     * GPU_CAT_SALTPASS to GPU_CAT_MASK because they're template-routed
     * (gpu_template.cl + gpu_md5salt_core.cl / gpu_md5saltpass_core.cl).
     * GPU_CAT_MASK is already the "template-routed" router flag (see
     * §9.1) — salt status is now orthogonal to category. The chokepoint
     * gate at mdxfind.c:10250 admits these two ops; the dispatcher at
     * gpu_opencl.c gpu_opencl_dispatch_md5_rules picks the right
     * salted-template kernel via gpu_template_resolve_kernel, and the
     * SETARG block binds the 3 extra salt args under
     * kern_is_salted_template. The salt-snapshot needs (Typesalt[]) are
     * still wired via gpu_salt_judy(g->op) — see needs_salt_snapshot
     * derivation at line ~575 (§11 row 20).
     *
     * B6.1 SHA1 fan-out (2026-05-06): JOB_SHA1SALTPASS joins. Same
     * template-routed path; gpu_template_resolve_kernel returns the
     * SHA1SALTPASS kernel via gpu_sha1saltpass_core.cl. Hit-replay
     * already includes JOB_SHA1SALTPASS in is_salted_op above.
     * B6.2 SHA256 fan-out (2026-05-06): JOB_SHA256SALTPASS joins.
     * Template-routed via gpu_sha256saltpass_core.cl. Hit-replay path
     * for SHA256SALTPASS at line ~1561 already exists for the legacy
     * slab path and works as-is (it dispatches to checkhashsalt with
     * a 64-byte digest length, identical semantics). is_salted_op below
     * needs JOB_SHA256SALTPASS added so the combined_ridx decompose
     * uses nsalts_packed for 3-axis decomposition.
     * B6.3 SHA224 fan-out (2026-05-06): JOB_SHA224SALTPASS joins.
     * Template-routed via gpu_sha224saltpass_core.cl — sha256_block
     * compression with 7-word truncated output. Hit-replay path uses
     * the same checkhashsalt-based replay (digest length 56 bytes for
     * SHA224 hex). is_salted_op above includes JOB_SHA224SALTPASS for
     * 3-axis decompose. SHA224SALTPASS was not previously in any
     * gpu_op_category case (defaulted to GPU_CAT_NONE / CPU-only); this
     * change adds first-class GPU template routing. */
    case JOB_MD5SALT:
    /* B6.6 (2026-05-06): MD5SALT family variants — share the SAME GPU
     * template kernel as e31 MD5SALT via params.algo_mode runtime flag.
     * No new GPU_TEMPLATE_* enums; resolver returns kern_template_phase0_md5salt
     * for all 4 ops. Slab kernel md5salt_batch is structurally unreachable for
     * these 3 ops post-this-move (B8 candidate). */
    case JOB_MD5UCSALT:
    case JOB_MD5revMD5SALT:
    case JOB_MD5sub8_24SALT:
    /* B6.8 (2026-05-06): JOB_MD5_MD5SALTMD5PASS (e367) — fifth
     * MD5SALT-family variant on the unified template path. Shares
     * the MD5SALT GPU kernel via params.algo_mode=4 (no new
     * GPU_TEMPLATE_* enum). Salt-pack geometry identical to e31
     * MD5SALT; only the salt CONTENT differs (host packs the
     * pre-computed salt-hex via gpu_pack_salts use_hashsalt=1). */
    case JOB_MD5_MD5SALTMD5PASS:
    /* Family A (2026-05-07): JOB_HMAC_MD5 (e214) + JOB_HMAC_MD5_KPASS
     * (e792) — sixth + seventh MD5SALT-template-kernel-sharing variants.
     * algo_mode=5 (KSALT: key=salt, msg=pass) and 6 (KPASS: key=pass,
     * msg=salt). HMAC body branches at the top of template_finalize and
     * returns early — bypasses the double-MD5 chain. Salt-pack uses raw
     * salt bytes (use_hashsalt=0); for KSALT the "salt" comes from
     * Typeuser via gpu_salt_judy(JOB_HMAC_MD5). max_iter forced to 1 at
     * the rules-engine dispatch site (gpu_opencl.c, mirrors SHA1DRU)
     * since HMAC has no CPU iter loop. CPU references: mdxfind.c:29250
     * (KSALT, checkhashkey label) + mdxfind.c:29423 (KPASS, checkhashsalt
     * with iter=0). Slab oracle: gpu_md5salt.cl hmac_md5_ksalt_batch +
     * hmac_md5_kpass_batch. */
    case JOB_HMAC_MD5:
    case JOB_HMAC_MD5_KPASS:
    /* Family B (2026-05-07): JOB_HMAC_SHA1 (e215) + JOB_HMAC_SHA1_KPASS
     * (e793) — share the SHA1SALTPASS GPU template kernel via
     * params.algo_mode = 5 (KSALT: key=salt, msg=pass) / 6 (KPASS:
     * key=pass, msg=salt). HMAC body branches at the top of template_-
     * finalize in finalize_prepend_be.cl.frag (gated on HASH_WORDS==5
     * so SHA224/SHA256 cores using the same fragment never execute the
     * SHA1-width HMAC body) and returns early. Salt-pack uses raw salt
     * bytes (use_hashsalt=0); for KSALT the "salt" comes from Typeuser
     * via gpu_salt_judy(JOB_HMAC_SHA1). max_iter forced to 1 at the
     * rules-engine dispatch site (gpu_opencl.c, mirrors Family A) since
     * HMAC has no CPU iter loop. CPU references: mdxfind.c:29301 (KSALT,
     * checkhashkey label) + mdxfind.c:29454 (KPASS, checkhashsalt with
     * iter=0). Slab oracle: gpu_sha1.cl hmac_sha1_ksalt_batch +
     * hmac_sha1_kpass_batch — both RETIRED in this same commit. */
    case JOB_HMAC_SHA1:
    case JOB_HMAC_SHA1_KPASS:
    /* Family C (2026-05-07): JOB_HMAC_SHA224 (e216) + JOB_HMAC_SHA224_KPASS
     * (e794) — share the SHA224SALTPASS GPU template kernel via
     * params.algo_mode = 5 (KSALT: key=salt, msg=pass) / 6 (KPASS:
     * key=pass, msg=salt). HMAC body branches at the top of template_-
     * finalize in finalize_prepend_be.cl.frag (gated on HASH_WORDS==7
     * so SHA1/SHA256 cores using the same fragment never execute the
     * SHA224-width HMAC body) and returns early. Salt-pack uses raw salt
     * bytes (use_hashsalt=0); for KSALT the "salt" comes from Typeuser
     * via gpu_salt_judy(JOB_HMAC_SHA224). max_iter forced to 1 at the
     * rules-engine dispatch site (gpu_opencl.c, mirrors Families A and
     * B) since HMAC has no CPU iter loop. CPU references: mdxfind.c:29326
     * case + HMAC_start label (KSALT, checkhashkey label) +
     * mdxfind.c:29479 case + HMAC_KPASS_start label (KPASS,
     * checkhashsalt with iter=0). Slab oracle: gpu_sha256.cl
     * hmac_sha224_ksalt_batch + hmac_sha224_kpass_batch — both RETIRED
     * in this same commit. */
    case JOB_HMAC_SHA224:
    case JOB_HMAC_SHA224_KPASS:
    /* Family D (2026-05-08): JOB_HMAC_SHA256 (e217) + JOB_HMAC_SHA256_KPASS
     * (e795) — share the SHA256SALTPASS GPU template kernel via
     * params.algo_mode = 5 (KSALT: key=salt, msg=pass) / 6 (KPASS:
     * key=pass, msg=salt). HMAC body branches at the top of template_-
     * finalize in finalize_prepend_be.cl.frag using RUNTIME gate
     * `if (HASH_WORDS == 8 && algo_mode >= 5u)` (NEVER `#if`; rev 1.7
     * Pascal NVIDIA ABORT lesson — see prominent CRITICAL comment in
     * finalize_prepend_be.cl.frag). Returns early after writing the
     * full 8-word digest. Salt-pack uses raw salt bytes (use_hashsalt=0);
     * for KSALT the "salt" comes from Typeuser via gpu_salt_judy(JOB_-
     * HMAC_SHA256). max_iter forced to 1 at the rules-engine dispatch
     * site (gpu_opencl.c, mirrors Families A/B/C) since HMAC has no
     * CPU iter loop. CPU references: mdxfind.c:29581 (KSALT, case
     * JOB_HMAC_SHA256 + HMAC_start label, hmac_len=64) +
     * mdxfind.c:29734 (KPASS, case JOB_HMAC_SHA256_KPASS + HMAC_KPASS_-
     * start label, hmac_len=64). Slab oracle: gpu_sha256.cl
     * hmac_sha256_ksalt_batch + hmac_sha256_kpass_batch — both RETIRED
     * in this same commit. Final HMAC family in the ladder. */
    case JOB_HMAC_SHA256:
    case JOB_HMAC_SHA256_KPASS:
    case JOB_MD5SALTPASS:
    case JOB_SHA1SALTPASS:
    case JOB_SHA256SALTPASS:
    case JOB_SHA224SALTPASS:
    /* B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted
     * variant. Same template-routed path as MD5SALTPASS but salt POSITION
     * differs at finalize time (APPEND vs PREPEND). defines_str
     * disambiguation: SALT_POSITION=APPEND vs PREPEND. */
    case JOB_MD5PASSSALT:
    /* B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-
     * shape salted variant. Template-routed via gpu_sha1passsalt_core.cl.
     * Cache disambiguated from SHA1SALTPASS via SALT_POSITION=APPEND
     * (vs PREPEND); same BASE_ALGO=sha1 + HASH_WORDS=5. From MD5PASSSALT
     * via HASH_WORDS=5 + BASE_ALGO=sha1 (both axes differ). */
    case JOB_SHA1PASSSALT:
    /* B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family APPEND-
     * shape salted variant. Template-routed via gpu_sha256passsalt_core.cl.
     * Cache disambiguated from SHA256SALTPASS via SALT_POSITION=APPEND
     * (vs PREPEND); same BASE_ALGO=sha256 + HASH_WORDS=8. From SHA1PASSSALT
     * via HASH_WORDS=8 + BASE_ALGO=sha256 (both axes differ). Pure spec
     * reuse — codegen tool produced gpu_sha256passsalt_core.cl from
     * specs.py with no new templates/fragments authored. */
    case JOB_SHA256PASSSALT:
    /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS — first 64-bit-
     * state salted variant on the unified template path (gpu_template.cl
     * + gpu_sha512saltpass_core.cl). Cache disambiguated via
     * HASH_BLOCK_BYTES=128 (unique among salted variants on the codegen
     * path) + HASH_WORDS=16 + BASE_ALGO=sha512. Authors a new sibling
     * main template (sha512_style_salted.cl.tmpl) AND a new sibling
     * fragment (finalize_prepend_be64.cl.frag) — width-bearing constants
     * (block_size, word_width, length-field-width) are not parameterized
     * into the SHA-256 template/fragment per the codegen-reconsideration
     * memo. */
    case JOB_SHA512SALTPASS:
    /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT — second
     * 64-bit-state salted variant; APPEND-shape sibling of SHA512SALTPASS.
     * FINAL B6 ladder step. Template-routed via gpu_sha512passsalt_core.cl.
     * Cache disambiguated from SHA512SALTPASS via SALT_POSITION=APPEND
     * (vs PREPEND); same BASE_ALGO=sha512 + HASH_WORDS=16 +
     * HASH_BLOCK_BYTES=128 axes — single-axis delta. Pure spec reuse
     * on the SHA-512 main template (B6.9), plus ONE new fragment
     * (finalize_append_be64.cl.frag — sibling of finalize_prepend_be64).
     * After this lands the entire SHA-512-family salted slab dispatcher
     * is unreachable — full B8 slab-retirement opens up. */
    case JOB_SHA512PASSSALT:
    /* Family E HMAC-SHA384 carrier (2026-05-08): JOB_HMAC_SHA384 (e543) +
     * JOB_HMAC_SHA384_KPASS (e796) — share the SHA384SALTPASS-shaped
     * carrier GPU template kernel via params.algo_mode = 5 (KSALT:
     * key=salt, msg=pass) / 6 (KPASS: key=pass, msg=salt). HMAC body
     * branches at the top of template_finalize in finalize_prepend_-
     * be64.cl.frag (gated on HASH_WORDS==12 so HASH_WORDS=16 SHA-512
     * cores using the same fragment never execute the SHA-384-width
     * HMAC body) and returns early. Salt-pack uses raw salt bytes
     * (use_hashsalt=0); for KSALT the "salt" comes from Typeuser via
     * gpu_salt_judy(JOB_HMAC_SHA384). max_iter forced to 1 at the
     * rules-engine dispatch site (gpu_opencl.c, mirrors Families A/B/C)
     * since HMAC has no CPU iter loop. CPU references: mdxfind.c:29369
     * case JOB_HMAC_SHA384 + HMAC_start label (KSALT, checkhashkey,
     * hmac_len=96) + mdxfind.c:29522 case JOB_HMAC_SHA384_KPASS +
     * HMAC_KPASS_start label (KPASS, checkhashsalt with iter=0,
     * hmac_len=96). Slab oracle: gpu_hmac_sha512.cl hmac_sha384_-
     * ksalt_batch (lines 267-367) + hmac_sha384_kpass_batch (lines
     * 369-467) — both RETIRED in this same commit (#if 0 wrapped). */
    case JOB_HMAC_SHA384:
    case JOB_HMAC_SHA384_KPASS:
    /* Family F (2026-05-08): JOB_HMAC_SHA512 (e218) + JOB_HMAC_SHA512_KPASS
     * (e797) — share the SHA512SALTPASS GPU template kernel via
     * params.algo_mode = 5 (KSALT: key=salt, msg=pass) / 6 (KPASS:
     * key=pass, msg=salt). HMAC body branches at the top of template_-
     * finalize in finalize_prepend_be64.cl.frag (gated on HASH_WORDS==16
     * so HASH_WORDS=12 SHA-384 cores using the same fragment never
     * execute the SHA-512-width HMAC body) and returns early. Salt-pack
     * uses raw salt bytes (use_hashsalt=0); for KSALT the "salt" comes
     * from Typeuser via gpu_salt_judy(JOB_HMAC_SHA512). max_iter forced
     * to 1 at the rules-engine dispatch site (gpu_opencl.c, mirrors
     * Families A/B/C/E) since HMAC has no CPU iter loop. CPU references:
     * mdxfind.c:29400 case JOB_HMAC_SHA512 + HMAC_start label (KSALT,
     * checkhashkey, hmac_len=128) + mdxfind.c:29553 case JOB_HMAC_SHA512_-
     * KPASS + HMAC_KPASS_start label (KPASS, checkhashsalt with iter=0,
     * hmac_len=128). Slab oracle: gpu_hmac_sha512.cl hmac_sha512_-
     * ksalt_batch (lines 26-143) + hmac_sha512_kpass_batch (lines
     * 146-254) — both RETIRED in this same commit (whole file
     * retirement; gpu_hmac_sha512.cl had no live kernels remaining
     * after Family E retired the SHA-384 slabs). */
    case JOB_HMAC_SHA512:
    case JOB_HMAC_SHA512_KPASS:
    /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): JOB_HMAC_RMD160 (e211)
     * + JOB_HMAC_RMD160_KPASS (e798) — share the RIPEMD160SALTPASS-shaped
     * carrier GPU template kernel via params.algo_mode = 5 (KSALT:
     * key=salt, msg=pass) / 6 (KPASS: key=pass, msg=salt). HMAC body
     * branches at the top of template_finalize in finalize_prepend_-
     * rmd.cl.frag (gated on HASH_WORDS==5; future RMD320 with HASH_WORDS=10
     * will share the same fragment via a sibling branch) and returns early.
     * Salt-pack uses raw salt bytes (use_hashsalt=0); for KSALT the "salt"
     * comes from Typeuser via gpu_salt_judy(JOB_HMAC_RMD160). max_iter
     * forced to 1 at the rules-engine dispatch site (gpu_opencl.c, mirrors
     * Families A/B/C/E/F) since HMAC has no CPU iter loop. CPU references:
     * mdxfind.c:29391 case JOB_HMAC_RMD160 + HMAC_start (KSALT,
     * checkhashkey, hmac_len=40) + mdxfind.c:29584 case JOB_HMAC_RMD160_-
     * KPASS + HMAC_KPASS_start (KPASS, checkhashsalt with iter=0,
     * hmac_len=40). Slab oracle: gpu_hmac_rmd160.cl hmac_rmd160_ksalt_-
     * batch (lines 24-135) + hmac_rmd160_kpass_batch (lines 141-242) —
     * both RETIRED in this same commit (whole-file retirement; FAM_HMAC_-
     * RMD160 had only these two live kernels). */
    case JOB_HMAC_RMD160:
    case JOB_HMAC_RMD160_KPASS:
    /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): JOB_HMAC_RMD320 (e213)
     * + JOB_HMAC_RMD320_KPASS (e799) — share the RIPEMD320SALTPASS-shaped
     * carrier GPU template kernel via params.algo_mode = 5 (KSALT:
     * key=salt, msg=pass) / 6 (KPASS: key=pass, msg=salt). HMAC body
     * branches at the top of template_finalize in finalize_prepend_-
     * rmd.cl.frag (gated on HASH_WORDS==10; sibling of the HASH_WORDS==5
     * RMD160 branch — both live in the same fragment) and returns early.
     * Salt-pack uses raw salt bytes (use_hashsalt=0); for KSALT the "salt"
     * comes from Typeuser via gpu_salt_judy(JOB_HMAC_RMD320). max_iter
     * forced to 1 at the rules-engine dispatch site (gpu_opencl.c, mirrors
     * Families A/B/C/E/F/G) since HMAC has no CPU iter loop. CPU references:
     * mdxfind.c:29428 case JOB_HMAC_RMD320 + HMAC_start (KSALT,
     * checkhashkey, hmac_len=80) + mdxfind.c:29616 case JOB_HMAC_RMD320_-
     * KPASS + HMAC_KPASS_start (KPASS, checkhashsalt with iter=0,
     * hmac_len=80). Slab oracle: gpu_hmac_rmd320.cl hmac_rmd320_ksalt_-
     * batch (lines 207-306) + hmac_rmd320_kpass_batch (lines 308-404) —
     * both RETIRED in this same commit (whole-file retirement; FAM_HMAC_-
     * RMD320 had only these two live kernels). */
    case JOB_HMAC_RMD320:
    case JOB_HMAC_RMD320_KPASS:
    /* Family I HMAC-BLAKE2S carrier (2026-05-08): JOB_HMAC_BLAKE2S (e828)
     * — share a hand-written Path A carrier GPU template kernel
     * (gpu_hmac_blake2s_core.cl) via params.algo_mode = 5. Single algo_mode
     * — no KPASS sibling op exists in mdxfind for HMAC-BLAKE2S. HMAC body
     * branches at the top of template_finalize (gated on algo_mode == 5u
     * inline in the hand-written core, not in a fragment) and returns early.
     * Salt-pack uses raw salt bytes (use_hashsalt=0); the "salt" comes from
     * Typesalt via gpu_salt_judy(JOB_HMAC_BLAKE2S) — the algorithm is KPASS-
     * shape (key=pass, msg=salt) but the op is named JOB_HMAC_BLAKE2S
     * (no -KPASS suffix because mdxfind never named a KSALT sibling).
     * max_iter forced to 1 at the rules-engine dispatch site (gpu_opencl.c,
     * mirrors Families A/B/C/E/F/G/H) since HMAC has no CPU iter loop. CPU
     * reference: mdxfind.c:30341 case JOB_HMAC_BLAKE2S, checkhashsalt with
     * iter=0 (mdxfind.c:30391, hmac_len=64 hex chars / 32 bytes). Slab
     * oracle: gpu_hmac_blake2s.cl hmac_blake2s_kpass_batch (lines 38-97) —
     * RETIRED in this same commit (whole-file retirement; FAM_HMAC_BLAKE2S
     * had only this one live kernel). */
    case JOB_HMAC_BLAKE2S:
    /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): JOB_HMAC_STREEBOG256_-
     * KSALT (e838) + JOB_HMAC_STREEBOG256_KPASS (e837) MOVED from GPU_CAT_-
     * SALTPASS (with the streebog512 siblings) to GPU_CAT_MASK — share the
     * hand-written Path A carrier GPU template kernel (gpu_hmac_streebog256_-
     * core.cl) via params.algo_mode = 5/6. Salt-pack geometry identical to
     * BLAKE2S256 unsalted at the host layer (1024-salt page cap, num_salts_-
     * per_page derivation, salt_start advance). For KSALT (e838) the "salt"
     * source is Typeuser via gpu_salt_judy(JOB_HMAC_STREEBOG256_KSALT); for
     * KPASS (e837) it's Typesalt. max_iter forced to 1 at the rules-engine
     * dispatch site since HMAC has no CPU iter loop. CPU references:
     * mdxfind.c:30764 (JOB_HMAC_STREEBOG256_KPASS) + mdxfind.c:30822 (JOB_-
     * HMAC_STREEBOG256_KSALT). Slab oracles: gpu_streebog.cl hmac_streebog-
     * 256_kpass_batch (lines 837-900) + hmac_streebog256_ksalt_batch (lines
     * 902-966) — both surgically deleted in this same commit. KEEP streebog-
     * 512 HMAC kernels (Family K scope, separate retirement). */
    case JOB_HMAC_STREEBOG256_KSALT:
    case JOB_HMAC_STREEBOG256_KPASS:
    /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): JOB_HMAC_STREEBOG512_-
     * KSALT (e840) + JOB_HMAC_STREEBOG512_KPASS (e839) MOVED from GPU_CAT_-
     * SALTPASS (where they had been alongside the prior slab path) to
     * GPU_CAT_MASK - share the hand-written Path A carrier GPU template
     * kernel (gpu_hmac_streebog512_core.cl) via params.algo_mode = 5/6.
     * Mirrors Family J STREEBOG-256 retirement pattern at HASH_WORDS=16
     * instead of 8. Salt-pack geometry identical to Family J at the host
     * layer. BOTH ops have TYPEOPT_NEEDSALT (mdxfind.c lines 7057-7058) -
     * salt-data lives in Typesalt[op] for both ops; gpu_salt_judy() falls
     * through to the Typesalt default arm. max_iter forced to 1 at the
     * rules-engine dispatch site since HMAC has no CPU iter loop. CPU
     * references: mdxfind.c:30918 (JOB_HMAC_STREEBOG512_KPASS) +
     * mdxfind.c:30975 (JOB_HMAC_STREEBOG512_KSALT). Slab oracles:
     * gpu_streebog.cl hmac_streebog512_kpass_batch (lines 859-923) +
     * hmac_streebog512_ksalt_batch (lines 925-989) - both surgically
     * deleted in this same commit (whole-file #if 0 wrap; gpu_streebog.cl
     * is empty post-Family-K retirement). Final HMAC family in the ladder. */
    case JOB_HMAC_STREEBOG512_KSALT:
    case JOB_HMAC_STREEBOG512_KPASS:
        return GPU_CAT_MASK;
    /* B5 sub-batch 3 (2026-05-06): BLAKE2B-256 / BLAKE2B-512 are wired only
     * through the per-word chokepoint → template_phase0 path (gpu_template.cl
     * + gpu_blake2b{256,512}_core.cl). No slab kernel exists for BLAKE2B
     * (gpu_blake2b256unsalted.cl is intentionally absent). Leaving these as
     * GPU_CAT_NONE prevents the slab path's gpu_try_pack_unsalted from
     * attempting a kernel selection that has no FAM_ entry — the chokepoint
     * widening (mdxfind.c rules-engine gate) handles GPU dispatch. */
    default:
        return GPU_CAT_NONE;
    }
}

int is_gpu_op(int op) {
    return gpu_op_category(op) != GPU_CAT_NONE;
}

/* Map JOB_* op code to FAM_* family index for timing state.
 * Derived from kernel_map[] in gpu_opencl.c / metal_kernel_map[] in gpu_metal.m. */
int gpu_op_family(int op) {
    switch (op) {
    case JOB_MD5SALT: case JOB_MD5UCSALT: case JOB_MD5revMD5SALT:
    case JOB_MD5sub8_24SALT:
    case JOB_HMAC_MD5: case JOB_HMAC_MD5_KPASS:
        return FAM_MD5SALT;
    case JOB_MD5SALTPASS: case JOB_MD5PASSSALT:
        return FAM_MD5SALTPASS;
    case JOB_MD5_MD5SALTMD5PASS:
        return FAM_MD5_MD5SALTMD5PASS;
    case JOB_SHA256PASSSALT: case JOB_SHA256SALTPASS:
    case JOB_SHA224SALTPASS:  /* B6.3 SHA224 fan-out (2026-05-06): SHA224 reuses
                               * sha256_block compression core; FAM_SHA256 covers
                               * timing/family bookkeeping for both. Output
                               * truncation to 7 words happens post-compression. */
    case JOB_HMAC_SHA256: case JOB_HMAC_SHA256_KPASS:
    case JOB_HMAC_SHA224: case JOB_HMAC_SHA224_KPASS:
        return FAM_SHA256;
    case JOB_PHPBB3:
        return FAM_PHPBB3;
    case JOB_MD5CRYPT:
        return FAM_MD5CRYPT;
    case JOB_DESCRYPT:
        return FAM_DESCRYPT;
    case JOB_SHA1DRU:
    case JOB_SHA1PASSSALT: case JOB_SHA1SALTPASS:
    case JOB_HMAC_SHA1: case JOB_HMAC_SHA1_KPASS:
        return FAM_SHA1;
    case JOB_MD5: case JOB_MD5UC: case JOB_MD5RAW:
        return FAM_MD5UNSALTED;
    case JOB_MD4: case JOB_NTLMH: case JOB_NTLM: case JOB_MD4UTF16:
        return FAM_MD4UNSALTED;
    case JOB_SHA1: case JOB_SHA1RAW: case JOB_SQL5:
        return FAM_SHA1UNSALTED;
    case JOB_SHA224: case JOB_SHA256: case JOB_SHA256RAW:
        return FAM_SHA256UNSALTED;
    case JOB_SHA384: case JOB_SHA384RAW:
    case JOB_SHA512: case JOB_SHA512RAW:
        return FAM_SHA512UNSALTED;
    case JOB_WRL:
        return FAM_WRLUNSALTED;
    case JOB_MD6256:
        return FAM_MD6256UNSALTED;
    case JOB_KECCAK224: case JOB_KECCAK256: case JOB_KECCAK384: case JOB_KECCAK512:
    case JOB_SHA3_224: case JOB_SHA3_256: case JOB_SHA3_384: case JOB_SHA3_512:
        return FAM_KECCAKUNSALTED;
    case JOB_SHA512PASSSALT: case JOB_SHA512SALTPASS:
    case JOB_HMAC_SHA512: case JOB_HMAC_SHA512_KPASS:
    case JOB_HMAC_SHA384: case JOB_HMAC_SHA384_KPASS:
        return FAM_HMAC_SHA512;
    case JOB_MYSQL3:
        return FAM_MYSQL3UNSALTED;
    case JOB_HMAC_RMD160: case JOB_HMAC_RMD160_KPASS:
        return FAM_HMAC_RMD160;
    case JOB_HMAC_RMD320: case JOB_HMAC_RMD320_KPASS:
        return FAM_HMAC_RMD320;
    case JOB_HMAC_BLAKE2S:
        return FAM_HMAC_BLAKE2S;
    case JOB_STREEBOG_32: case JOB_STREEBOG_64:
    case JOB_HMAC_STREEBOG256_KPASS: case JOB_HMAC_STREEBOG256_KSALT:
    case JOB_HMAC_STREEBOG512_KPASS: case JOB_HMAC_STREEBOG512_KSALT:
        return FAM_STREEBOG;
    case JOB_SHA512CRYPT: case JOB_SHA512CRYPTMD5:
        return FAM_SHA512CRYPT;
    case JOB_SHA256CRYPT:
        return FAM_SHA256CRYPT;
    case JOB_RMD160:
        return FAM_RMD160UNSALTED;
    case JOB_BLAKE2S256:
        return FAM_BLAKE2S256UNSALTED;
    case JOB_BCRYPT:
        return FAM_BCRYPT;
    default:
        return -1;
    }
}

/* Number of uint32 hash output words for this algorithm.
 * Determines hit_stride, hash verification length (hexlen = words * 8). */
int gpu_hash_words(int op) {
    switch (op) {
    /* 512-bit output = 16 words */
    case JOB_SHA512: case JOB_SHA512RAW:
    case JOB_SHA512PASSSALT: case JOB_SHA512SALTPASS:
    case JOB_WRL:
    case JOB_STREEBOG_64:
    case JOB_HMAC_SHA512: case JOB_HMAC_SHA512_KPASS:
    case JOB_HMAC_STREEBOG512_KSALT: case JOB_HMAC_STREEBOG512_KPASS:
    case JOB_KECCAK512: case JOB_SHA3_512:
    case JOB_SHA512CRYPT: case JOB_SHA512CRYPTMD5:
    /* B5 sub-batch 3: BLAKE2B-512 (full 64-byte digest = 16 uint32 LE). */
    case JOB_BLAKE2B512:
        return 16;
    /* 384-bit output = 12 words */
    case JOB_SHA384: case JOB_SHA384RAW:
    case JOB_HMAC_SHA384: case JOB_HMAC_SHA384_KPASS:
    case JOB_KECCAK384: case JOB_SHA3_384:
        return 12;
    /* 256-bit output = 8 words */
    case JOB_SHA256: case JOB_SHA256RAW:
    case JOB_SHA256PASSSALT: case JOB_SHA256SALTPASS:
    /* Family C (2026-05-07): JOB_HMAC_SHA224 + JOB_HMAC_SHA224_KPASS MOVED
     * from this 8-word arm to the 7-word arm below. Slab path used
     * EMIT_HIT_8 and emitted 8 words (with the discarded h[7] = whatever
     * sha256_block left there); template path uses EMIT_HIT_7 and emits
     * the 7 valid SHA224 digest words only. The 7-word value matches
     * the CPU `hmac_len = 56` (= 56 hex chars = 28 bytes = 7 uint32). */
    case JOB_HMAC_SHA256: case JOB_HMAC_SHA256_KPASS:
    case JOB_HMAC_STREEBOG256_KSALT: case JOB_HMAC_STREEBOG256_KPASS:
    case JOB_HMAC_BLAKE2S:
    case JOB_BLAKE2S256:
    /* B5 sub-batch 3: BLAKE2B-256 (truncated 32-byte digest = first 4-of-8
     * ulong → 8 uint32 LE). */
    case JOB_BLAKE2B256:
    case JOB_KECCAK256: case JOB_SHA3_256:
    case JOB_STREEBOG_32:
    case JOB_MD6256:
    case JOB_SHA256CRYPT:
        return 8;
    /* 224-bit output = 7 words */
    case JOB_SHA224:
    case JOB_SHA224SALTPASS:  /* B6.3 SHA224 fan-out (2026-05-06): truncated
                               * SHA256 state to 7 words = 28 bytes = 224 bits. */
    /* Family C (2026-05-07): JOB_HMAC_SHA224 (e216) + JOB_HMAC_SHA224_KPASS
     * (e794) moved from the 8-word arm. Template path emits 7 words via
     * EMIT_HIT_7; CPU hexlen = 56 (`hmac_len = 56` in MDstart at
     * mdxfind.c:29328 / 29481). */
    case JOB_HMAC_SHA224: case JOB_HMAC_SHA224_KPASS:
    case JOB_KECCAK224: case JOB_SHA3_224:
        return 7;
    /* 320-bit output = 10 words (RIPEMD-320 + HMAC-RMD320) */
    case JOB_RMD320:
    case JOB_HMAC_RMD320: case JOB_HMAC_RMD320_KPASS:
        return 10;
    /* 192-bit output = 6 words (bcrypt) */
    case JOB_BCRYPT:
        return 6;
    /* 160-bit output = 5 words */
    case JOB_SHA1: case JOB_SHA1RAW: case JOB_SQL5:
    case JOB_SHA1PASSSALT: case JOB_SHA1SALTPASS: case JOB_SHA1DRU:
    case JOB_HMAC_SHA1: case JOB_HMAC_SHA1_KPASS:
    case JOB_RMD160:
    case JOB_HMAC_RMD160: case JOB_HMAC_RMD160_KPASS:
        return 5;
    /* 128-bit output = 4 words (MD5, MD4, NTLM, etc.) */
    default:
        return 4;
    }
}

#endif /* OPENCL_GPU */
