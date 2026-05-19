/*
 * $Revision: 1.1 $
 * $Log: gpujob_metal.m,v $
 * Revision 1.1  2026/05/12 16:23:33  dlr
 * Initial check-in: Phase 2a Metal port GPU worker thread. Single-device gpujob_metal.m, mirrors gpu_gpujob_opencl.c structurally (~3844 LOC OpenCL twin); 784 LOC here strips multi-device, BF telemetry, slab dispatcher, salt-snapshot, hashcat-style mask, and trace-channel scaffolding (memo Phase 2a sections 3-4). Surface: gpujob_init/_shutdown/_available/_batch_max/_queue_depth/_free_count/_print_share_line/_get_free/_get_free_rules/_try_get_free/_submit/_return_free/_overflow_preload_all, gpu_op_category. JOB_MD5 unsalted-with-rules only (rules-engine sole producer of packed slots; slab path retired). Hit-replay collapses 3-axis OpenCL decompose to 1-axis (mask_size=1, nsalts=1). Sentinel op==2000 shutdown protocol matches OpenCL twin. Plain Objective-C (NOT Obj-C++); -fobjc-arc; matches gpu_metal.m build flags. Isolated compile passes on iMac AMD Radeon Pro 580X. Symbols replace Phase 1 stubs at end of gpu_metal.m.
 *
 *
 */
/* gpujob_metal.m — Phase 2a Metal port of the GPU worker thread.
 *
 * Mirrors gpu/gpujob_opencl.c structurally (~3844 LOC OpenCL twin); this
 * file targets ~700 LOC by stripping the OpenCL twin's multi-device, BF
 * telemetry, slab dispatcher, salt-snapshot, hashcat-style mask, and
 * trace-channel scaffolding. Phase 2a scope (memo §3 + §4):
 *
 *   - Single device (mtl_device set by gpu_metal_init in gpu_metal.m).
 *   - One worker thread.
 *   - JOB_MD5 only (GPU_CAT_UNSALTED via gpu_op_category).
 *   - rules_engine path only — every g->packed=1 slot is g->rules_engine=1.
 *     The slab dispatcher (retired in OpenCL B7.9) was never carried over
 *     to Metal; the rules-engine pack is the sole producer.
 *   - Hit-replay decodes (widx, ridx, iter) via simple divmod (mask_size==1
 *     in Phase 2a; salt_size==1 — no salt axis Phase 2a).
 *
 * Phase 2a non-goals (memo §1, §4 reshape):
 *   - BF chunk telemetry (bf_dev_wall_us/_chunk_total/_first_dispatch_done).
 *   - _max_salt_count / _max_salt_bytes sizing.
 *   - Multi-device share-line printing (single-line single-device summary).
 *   - MDXFIND_DISPATCH_TRACE / MDXFIND_PIPE_TRACE env hooks.
 *   - PCIe LnkSta capture (lspci is Linux-only; macOS has no equivalent).
 *
 * --- Public surface (replaces Phase 1 gpujob stubs in gpu_metal.m) ---
 *
 *   gpujob_init(num_jobg)     — allocate jobg pool + spawn worker thread.
 *   gpujob_shutdown()         — drain queues, send sentinel, join worker.
 *   gpujob_available()        — 1 if init succeeded.
 *   gpujob_batch_max()        — real per-dispatch word cap.
 *   gpujob_queue_depth()      — outstanding work-queue length.
 *   gpujob_free_count()       — free pool slot count.
 *   gpujob_print_share_line() — single-line per-device summary at shutdown.
 *   gpu_op_category(op)       — Phase 2a: JOB_MD5 -> GPU_CAT_UNSALTED.
 *   gpujob_get_free(...)      — pull a slot from the legacy free-list.
 *   gpujob_get_free_rules(...) — pull from rules free-list (same shape).
 *   gpujob_try_get_free()     — non-blocking variant.
 *   gpujob_submit(g)          — enqueue a filled slot.
 *   gpujob_return_free(g)     — return a slot without submitting.
 *   gpujob_overflow_preload_all() — Phase 2a: single-device passthrough.
 *
 * --- Sentinel protocol ---
 *
 *   Shutdown sends one jobg with op==2000 per worker thread. The worker
 *   pops it, returns the slot to its origin pool by slot_kind, and breaks
 *   out of its dispatch loop. Same protocol as the OpenCL twin (gpu/
 *   gpujob_opencl.c:718). The sentinel op value 2000 is hard-coded; no
 *   gpujob.h constant for it (mirroring OpenCL choice).
 *
 * --- ARC + Objective-C interaction ---
 *
 *   This file is plain Objective-C (NOT Obj-C++). Compiled with
 *   `-fobjc-arc` (same flags as gpu_metal.m). The host file gpu_metal.m
 *   owns the MTLDevice / MTLCommandQueue / MTLLibrary / PSO statics
 *   (mtl_device etc. are file-static there); this TU calls the public
 *   API gpu_metal_dispatch_md5_rules() to dispatch a batch.
 *
 *   Rationale for not touching the statics directly: avoiding a separate
 *   accessor adds ~6 LOC to gpu_metal.m and keeps the device/queue
 *   ownership scoped to its TU. Memo §3 row 6 lists "real gpu_metal_-
 *   dispatch_md5_rules" as a main-session EDIT — the worker thread here
 *   calls through that single entry point (matches the OpenCL pattern:
 *   gpujob_opencl.c calls gpu_opencl_dispatch_md5_rules(), it doesn't
 *   reach into gpu/gpu_opencl.c's static cl_device statics).
 *
 * --- $Log keyword corruption guard ---
 *
 *   This file will see many commits; per feedback_rcs_log_comment_bug.md
 *   the first ci -l MUST be followed by `rcs -ko gpujob_metal.m` to
 *   disable keyword expansion (otherwise a *.cl-style multi-line $Log
 *   eventually corrupts a `* /` close-comment). All RCS keywords above
 *   appear in a single $Revision / $Log pair to ease that transition.
 */

#if defined(METAL_GPU)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdatomic.h>
#include <time.h>
#include <Judy.h>

#include "mdxfind.h"
#include "job_types.h"
#include "gpujob.h"
#include "gpu_metal.h"
#include "yarn.h"

/* User directive 2026-05-17: debug stderr emissions in GPU paths MUST NOT
 * live in the shipped binary. See gpu/gpu_debug.h. */
#include "gpu_debug.h"

/* OUTBUFSIZE is mdxfind.c's #define (OUTBUFSIZE (MAXLINE+MAXLINE) at line 160)
 * — NOT exposed via mdxfind.h. gpujob_opencl.c carries its own re-definition
 * at line 534; we mirror that pattern. The OpenCL twin uses (1024 * 1024) =
 * 1 MB which is much larger than (MAXLINE+MAXLINE = 80 KB); the larger size
 * accommodates batch output buffering across many concurrent hits. Phase 2a
 * sticks with the 1 MB OpenCL-twin choice for behavioural parity. */
#define OUTBUFSIZE (1024 * 1024)

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

/* ---- External globals from mdxfind.c ----
 *
 * The OpenCL twin reaches into a wider set of externs (Maxiter, Printall,
 * pause flags, salt globals, etc.). Phase 2a's worker only needs the
 * unsalted + rules-engine subset; we extern just what we touch. */
extern int Printall;
extern int Maxiter;
extern volatile int MDXpause;
extern volatile int MDXpaused_count;

extern atomic_ullong Tothash;
extern atomic_ullong Totfound;
extern atomic_ullong Totrules_gpu;

/* Rules-engine globals (set by classify_rules in mdxfind.c main).
 * gpu_rule_program is the NUL-separated bytecode the kernel reads;
 * gpu_rule_origin[ridx] is the index back into the host's Rules[]
 * length-prefixed buffer used for hit-replay applyrule. */
extern unsigned char *gpu_rule_program;
extern uint32_t      *gpu_rule_offsets;
extern int           *gpu_rule_origin;
extern int            gpu_rule_count;
extern char          *Rules;
extern unsigned int   Numrules;

/* Phase 2b row 5: mask descriptor cache. Owned by gpu_metal.m (file-scope
 * non-static; mirrors gpu/gpu_opencl.c's gpu_mask_* externs at lines
 * ~3993-3996 + the gpu/gpujob_opencl.c externs at lines 242-245, 263).
 * Hit-replay reads these to reconstruct the prepend/append bytes for the
 * hit's mask_idx_local. Stays at zero when no mask has been uploaded
 * (gpu_metal_set_mask not called) — the gate gpu_mask_total > 1 below
 * skips the decode entirely in that case. */
extern int      gpu_mask_n_prepend;
extern int      gpu_mask_n_append;
extern uint64_t gpu_mask_total;
extern uint8_t  gpu_mask_sizes[];
/* Charset rows live in the MTLBuffer owned by gpu_metal.m; the host-side
 * descriptor for hit-replay reconstruction needs the actual character
 * tables, so gpujob_metal.m would need a pointer to them. Phase 2b
 * accesses the original (host) tables via gpu_mask_tables_host[] —
 * see below. The decision NOT to extern-into the MTLBuffer is per
 * Phase 2b architect §3: the OpenCL twin uses gpu_mask_desc which IS
 * the host descriptor (the device buffer is a copy); we mirror that
 * pattern. */
extern uint8_t  gpu_mask_charsets_host[][256];

/* CPU hit-replay entry point. The OpenCL twin uses both checkhash (no
 * salt) and checkhashsalt (salted). Phase 2c extends Metal hit-replay
 * with checkhashsalt for JOB_MD5SALT (mirrors gpu/gpujob_opencl.c arm
 * at line 2347 + 1702). Phase 2c only routes JOB_MD5SALT through
 * checkhashsalt at iter > 1; JOB_MD5SALT at iter==1 uses checkhashkey
 * (matches CPU path at mdxfind.c:22198/22207). We pull in
 * checkhashkey too for the iter==1 case. */
extern int checkhash(union HashU *curin, int len, int x, struct job *job);
extern int checkhashsalt(union HashU *curin, int len, char *salt,
                         int saltlen, int x, struct job *job);
extern int checkhashkey(union HashU *curin, int len, char *salt,
                        struct job *job);
/* Phase 2d.8a hit-replay externs:
 *   - checkhashbb: PHPBB3 (JOB_PHPBB3) -- bb-specific 32-hex output with
 *     phpitoa64-encoded 22-char hash + salt prefix. Mirrors
 *     gpujob_opencl.c extern at line 42.
 *   - hybrid_check + md5crypt_b64encode + prfound: MD5CRYPT (JOB_MD5CRYPT)
 *     -- hybrid_check probes the 16-byte binary MD5 digest;
 *     md5crypt_b64encode reconstructs the 22-char phpitoa64 hash;
 *     prfound emits "$1$<salt>$<22>" output via the synthetic job. Mirrors
 *     gpujob_opencl.c externs at lines 39-41. */
extern int checkhashbb(union HashU *curin, int len, char *salt, struct job *job);
extern int hybrid_check(const unsigned char *, int, int *, unsigned short **);
extern void md5crypt_b64encode(const unsigned char *, char *);
extern void prfound(struct job *job, char *str);

/* Phase 2d.9a DESCRYPT hit-replay externs. JudyJ[JOB_DESCRYPT] holds
 * the 13-char crypt(3) hashes loaded at startup; the hit-replay arm
 * probes by reconstructing the 13-char string from the GPU's pre-FP
 * (l, r) pair via the metal_des_reconstruct helper below. Mirrors
 * gpu/gpujob_opencl.c externs at line 53 (JudyJ) and the static
 * des_reconstruct function at line 462 (duplicated below; see
 * comment there for the project_hx_algo_dedup.md citation). */
extern Pvoid_t JudyJ[];

/* Phase 2d.8b SHACRYPT triple hit-replay helpers.
 *
 * sha{256,512}crypt_b64encode: glibc crypt-sha{256,512} byte-permutation
 * b64 encoders. SHA256CRYPT produces 43 chars from a 32-byte digest; SHA-
 * 512CRYPT produces 86 chars from a 64-byte digest. SHA512CRYPTMD5 reuses
 * the SHA512CRYPT encoder (same output format $6$...). Ported verbatim
 * from gpu/gpujob_opencl.c:100-126 (sha256crypt) + :155-184 (sha512crypt).
 *
 * The byte permutation tables match mdxfind.c:37036-37041 (sha512_perm
 * [21][3] verbatim) and the implicit order from mdxfind.c:12964 (cas =
 * (h[31]<<8)|h[30]) for the sha256 final tuple.
 *
 * phpitoa64 is the b64 alphabet "./0-9A-Za-z" defined in mdxfind.c:2811;
 * we need an extern declaration here for the encoders. */
extern char phpitoa64[];

/* Encode a 32-byte SHA-256 digest into the 43-char SHA256CRYPT base64
 * format (glibc crypt-sha256 layout). Output 43 chars + NUL.
 * Ported verbatim from gpu/gpujob_opencl.c:100-126. */
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

/* Encode a 64-byte SHA-512 digest into the 86-char SHA512CRYPT base64
 * format (glibc crypt-sha512 layout). Output 86 chars + NUL.
 * Ported verbatim from gpu/gpujob_opencl.c:155-184. */
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

/* Phase 2d.9a: metal_des_reconstruct -- rebuild the 13-char crypt(3)
 * DES hash from the GPU's pre-FP (l, r) pair + 2-char phpitoa64 salt.
 * Ported byte-for-byte from gpu/gpujob_opencl.c des_reconstruct at line
 * 462 (the OpenCL-side helper is `static` and only linked into the
 * OPENCL_GPU TU, so we duplicate here with a citation per project_hx_-
 * algo_dedup.md discipline). Applies the inverse FP permutation
 * (DES_FP[64] table; final-permutation byte-position lookup) byte-by-
 * byte to (il, ir) -> (r0, r1), then phpitoa64-encodes the 64-bit
 * post-FP block as salt[0..1] + 11 base64 chars.
 *
 * Output buffer `out` must hold at least 14 bytes (13 chars + NUL). */
static void metal_des_reconstruct(uint32_t gl, uint32_t gr,
                                  const char *salt, char *out) {
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

/* Phase 2d.9b BCRYPT hit-replay extern. JudyJ[JOB_BCRYPT] holds the
 * 60-char $2[abxy]$NN$<22-b64-salt><31-b64-hash> crypt(3) hashes loaded
 * at startup; the hit-replay arm reconstructs the 31-char b64 hash
 * portion from the GPU's 24-byte digest via bf_encode_23 (below), splices
 * it onto the salt prefix from Typesalt, and probes JudyJ[JOB_BCRYPT]
 * for the full 60-char string. JudyJ is already externed above for
 * JOB_DESCRYPT in Phase 2d.9a -- no new extern needed; we just reuse
 * the [JOB_BCRYPT] slot. */

/* Phase 2d.9b: bf_encode_23 -- encode 23 raw bytes (bcrypt's 24-byte
 * digest with the trailing byte truncated by BF_encode) into 31 base64
 * characters using bcrypt's custom alphabet ("./ABCDEFGHIJKLMNOPQRSTUV-
 * WXYZabcdefghijklmnopqrstuvwxyz0123456789"). Ported byte-for-byte
 * from gpu/gpujob_opencl.c bf_encode_23 at line 497-531 (the OpenCL-
 * side helper is `static` and only linked into the OPENCL_GPU TU, so
 * we duplicate here with a citation per project_hx_algo_dedup.md
 * discipline -- same pattern as metal_des_reconstruct above).
 *
 * Output buffer `out` must hold at least 32 bytes (31 chars + NUL).
 * `raw` must point to 23 readable bytes; sourced from curin.i[0..5]
 * reinterpreted as a 24-byte LE stream (first 23 are encoded, byte 24
 * is the kernel's BE->LE swap zero pad which BF_encode discards). */
static void metal_bf_encode_23(const unsigned char *raw, char *out) {
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

/* Phase 2d.3.1: hash-words-per-op helper. Mirrors the OpenCL twin's
 * gpu_hash_words() at gpu/gpujob_opencl.c:3773 — but Metal builds don't
 * compile gpujob_opencl.c (OPENCL_GPU gate), so we duplicate the lookup
 * here for the ops Metal currently supports. The hit-replay decode loop
 * reads exactly this many uint32 words from the hits buffer entry into
 * curin.i[]. Prior to Phase 2d.3.1 the loop hardcoded 4 (MD4/MD5
 * digest); SHA-1 needs 5 (20-byte digest), and the host's hybrid_check
 * comparison reads the full digest length so missing bytes 16..19 made
 * the SHA-1 GPU path produce garbage matches.
 *
 * The OpenCL twin covers many more ops (SHA-256/384/512, BLAKE2,
 * STREEBOG, Keccak, HMAC, ...). Add entries here as Metal ports catch
 * up. The default arm returns 4 (MD4/MD5 digest width) so any not-yet-
 * registered op falls through to the prior behavior; if a new op is
 * misconfigured the worst case is the prior bug class. */
static int metal_gpu_hash_words(int op)
{
    switch (op) {
    case JOB_SHA1:        /* 160-bit raw SHA-1 = 5 uint32 */
    case JOB_SHA1RAW:     /* Phase 2d.3.2: 5 uint32 (binary-digest re-feed) */
    case JOB_SHA1DRU:     /* Phase 2d.3.3: 5 uint32 (1M-iter Drupal SHA-1) */
    case JOB_SHA1PASSSALT: /* Phase 2d.3.4: 5 uint32 (sha1(pass||salt)) */
    case JOB_SHA1SALTPASS: /* Phase 2d.3.5: 5 uint32 (sha1(salt||pass)) */
    case JOB_RMD160:      /* Phase 2d.6.1: 5 uint32 (RIPEMD-160 LE; FIRST RIPEMD-family) */
        return 5;
    case JOB_RMD320:      /* Phase 2d.6.2: 10 uint32 (RIPEMD-320 LE; widest mask-category digest) */
        return 10;
    case JOB_SHA224:           /* Forward-stage 2d.4.x: 7 uint32 (28-byte digest) */
    case JOB_SHA224SALTPASS:   /* Forward-stage 2d.4.x: 7 uint32 (sha224(salt||pass)) */
    case JOB_KECCAK224:        /* Phase 2d.7b.2: 7 uint32 (Keccak-224 = 28-byte LE digest) */
    case JOB_SHA3_224:         /* Phase 2d.7b.5: 7 uint32 (SHA3-224 = 28-byte LE digest) */
        return 7;
    case JOB_SHA256:           /* Phase 2d.4.1: 8 uint32 (raw SHA-256, 32-byte digest) -- CANARY */
    case JOB_SHA256RAW:        /* Forward-stage 2d.4.x: 8 uint32 (binary-digest re-feed) */
    case JOB_SHA256SALTPASS:   /* Forward-stage 2d.4.x: 8 uint32 (sha256(salt||pass)) */
    case JOB_SHA256PASSSALT:   /* Forward-stage 2d.4.x: 8 uint32 (sha256(pass||salt)) */
    case JOB_BLAKE2S256:       /* Phase 2d.7a.1: 8 uint32 (BLAKE2S-256 = 32-byte LE digest) */
    case JOB_BLAKE2B256:       /* Phase 2d.7a.2: 8 uint32 (BLAKE2B truncated to 32 bytes) */
    case JOB_KECCAK256:        /* Phase 2d.7b.1: 8 uint32 (Keccak-256 = 32-byte LE digest) -- CANARY (FIRST Keccak/SHA-3 sponge family) */
    case JOB_SHA3_256:         /* Phase 2d.7b.6: 8 uint32 (SHA3-256 = 32-byte LE digest) */
    case JOB_STREEBOG_32:      /* Phase 2d.7c.1: 8 uint32 (Streebog-256 = 32-byte LE digest) -- CANARY (FIRST Streebog family, in-body uchar* cast translator extension) */
    case JOB_HMAC_BLAKE2S:     /* Phase 2d.7d.1: 8 uint32 (HMAC-BLAKE2S = 32-byte LE digest) -- CANARY (FIRST HMAC family on Metal) */
    case JOB_HMAC_STREEBOG256_KSALT: /* Phase 2d.7d.2: 8 uint32 (HMAC-STREEBOG-256 KSALT = 32-byte LE digest) */
    case JOB_HMAC_STREEBOG256_KPASS: /* Phase 2d.7d.2: 8 uint32 (HMAC-STREEBOG-256 KPASS = 32-byte LE digest; shares kernel with KSALT) */
    case JOB_SHA256CRYPT:      /* Phase 2d.8b: 8 uint32 (SHA-256 crypt = 32-byte LE digest; SHACRYPT shared core at HASH_WORDS=8) -- canary for the SHACRYPT triple */
        return 8;
    case JOB_SHA384:           /* Forward-stage 2d.5.x: 12 uint32 (48-byte digest) */
    case JOB_SHA384RAW:        /* Forward-stage 2d.5.x: 12 uint32 (binary-digest re-feed) */
    case JOB_SHA384SALTPASS:   /* Forward-stage 2d.5.x: 12 uint32 (sha384(salt||pass)) */
    case JOB_KECCAK384:        /* Phase 2d.7b.3: 12 uint32 (Keccak-384 = 48-byte LE digest) */
    case JOB_SHA3_384:         /* Phase 2d.7b.7: 12 uint32 (SHA3-384 = 48-byte LE digest) */
        return 12;
    case JOB_SHA512:           /* Phase 2d.5.1: 16 uint32 (raw SHA-512, 64-byte digest) -- CANARY (FIRST 64-bit-state family) */
    case JOB_SHA512RAW:        /* Forward-stage 2d.5.x: 16 uint32 (binary-digest re-feed) */
    case JOB_SHA512SALTPASS:   /* Forward-stage 2d.5.x: 16 uint32 (sha512(salt||pass)) */
    case JOB_SHA512PASSSALT:   /* Forward-stage 2d.5.x: 16 uint32 (sha512(pass||salt)) */
    case JOB_BLAKE2B512:       /* Phase 2d.7a.3: 16 uint32 (BLAKE2B-512 = full 64-byte LE digest) */
    case JOB_KECCAK512:        /* Phase 2d.7b.4: 16 uint32 (Keccak-512 = 64-byte LE digest) */
    case JOB_SHA3_512:         /* Phase 2d.7b.8: 16 uint32 (SHA3-512 = 64-byte LE digest) */
    case JOB_STREEBOG_64:      /* Phase 2d.7c.2: 16 uint32 (Streebog-512 = 64-byte LE digest) */
    case JOB_HMAC_STREEBOG512_KSALT: /* Phase 2d.7d.3: 16 uint32 (HMAC-STREEBOG-512 KSALT = 64-byte LE digest) */
    case JOB_HMAC_STREEBOG512_KPASS: /* Phase 2d.7d.3: 16 uint32 (HMAC-STREEBOG-512 KPASS = 64-byte LE digest; shares kernel with KSALT) */
    case JOB_SHA512CRYPT:      /* Phase 2d.8b: 16 uint32 (SHA-512 crypt = 64-byte LE digest; SHACRYPT shared core at HASH_WORDS=16) */
    case JOB_SHA512CRYPTMD5:   /* Phase 2d.8b: 16 uint32 (SHA-512 crypt with MD5-preprocess = 64-byte LE digest; SHACRYPT shared core at HASH_WORDS=16; algo_mode=1u) */
        return 16;
    case JOB_MD5:         /* 128-bit MD5 / MD4 / MD5RAW / MD4UTF16 = 4 uint32 */
    case JOB_MD4:
    case JOB_MD5RAW:
    case JOB_MD4UTF16:
    case JOB_MD5SALT:
    case JOB_MD5PASSSALT:
    case JOB_MD5SALTPASS:
    case JOB_BCRYPT:      /* Phase 2d.9b: BCRYPT (op=450) emits 6 LE uint32
                           * words = 24 bytes (first 4 LE words = 16 bytes
                           * probe the compact-table fingerprint; all 6
                           * words travel to host hit-replay for the
                           * 31-char bf_encode_23 reconstruction). FIRST
                           * 6-word Metal family. CRITICAL: per feedback_-
                           * metal_hash_words_width_helper.md the default
                           * arm returns 4 -- omitting this case would
                           * leave curin.i[4..5] uninitialized and silently
                           * fail bf_encode_23 reconstruction (TRAP 3 of
                           * Phase 2d.9b architect brief). */
        return 6;
    case JOB_DESCRYPT:    /* Phase 2d.9a: DESCRYPT (op=500) carries pre-FP
                           * (l, r) in h[0..1] + h[2..3] zero-pad = 16 bytes
                           * = 4 uint32. Explicit case per feedback_metal_-
                           * hash_words_width_helper.md discipline (even
                           * though MD5 width is the default arm). */
    default:
        return 4;
    }
}

/* Phase 2c row 18: salt-snapshot externs. mdxfind.c owns Typesalt[]
 * (the per-op salt Judy index keyed by salt string -> PV count).
 * build_salt_snapshot walks Typesalt[op]'s Judy into a flat
 * saltentry[] array; gpu_pack_salts_op packs the snapshot into the
 * contiguous wire format the GPU expects. We mirror gpu/gpujob_opencl.c
 * externs at lines 47-51. */
struct saltentry;
extern void **Typesalt;
extern void **Typeuser;
extern char  Typedone[];
extern int   build_salt_snapshot(void *snap, char *pool, void *judy,
                                 char *keybuf, int printall);
extern Pvoid_t *Typehashsalt;
extern int   build_hashsalt_snapshot(struct saltentry *, char *,
                                     Pvoid_t, char *, int);
extern unsigned char i64hex[];  /* base64url decode table from mdxfind.c */

/* Type-data extents for sizing the per-worker salt-snapshot scratch.
 * Mirrors gpujob_opencl.c statics at lines 440-441 + the init loop at
 * 2534-2543 (computed from Typesaltcnt[]/Typesaltbytes[] in mdxfind.c).
 * Declared file-local; populated in gpujob_init below. */
static int _max_salt_count = 0;
static int _max_salt_bytes = 0;
extern int *Typesaltcnt;
extern long long *Typesaltbytes;

/* PV_DEC macro mirrors gpu/gpujob_opencl.c line 265-268. PV is the
 * Judy-keyed pending-count; we decrement on a confirmed hit so the
 * matching salt drops out of subsequent dispatches. */
/* Phase 2g 2026-05-18: extended to return retirement (1->0 transition)
 * detection via GCC/clang statement-expression. Returns 1 iff this CAS
 * was the one that took PV from 1 to 0 (this salt just retired);
 * 0 otherwise (PV already 0, or still >0 after DEC). Race-correct:
 * only the winning CAS sees _old == 1 in the success branch. */
#define MTL_PV_DEC(pv) ({ unsigned long _old = *(pv); int _retired = 0; \
  while (_old > 0) { \
    if (__sync_bool_compare_and_swap((pv), _old, _old - 1)) { \
      _retired = (_old == 1); break; } \
    _old = *(pv); } _retired; })

/* struct saltentry mirror. The original lives in mdxfind.c at line 2763;
 * gpu/gpujob_opencl.c carries a redeclaration at line 270 (same layout
 * minus the unused fields). We mirror the OpenCL twin's redeclaration
 * exactly so a future struct-layout drift is caught by build_salt_snapshot
 * touching the wrong fields. */
struct saltentry {
    char *salt;
    unsigned long *PV;
    int saltlen;
    char *hashsalt;
    int hashlen;
    uint32_t iter;  /* internal iteration count; see mdxfind.c:struct saltentry */
};

/* gpu_salt_judy(op): resolve the per-op salt Judy index.
 *
 * Phase 2d.7d HMAC siblings (5 ops): per
 * feedback_hmac_salt_judy_typeopt.md, the Metal port currently admits
 * ONLY HMAC families whose ALL ops have TYPEOPT_NEEDSALT:
 *   - JOB_HMAC_BLAKE2S (op=828): Typesalt
 *   - JOB_HMAC_STREEBOG256_KSALT (op=838): Typesalt
 *   - JOB_HMAC_STREEBOG256_KPASS (op=837): Typesalt  -- distinct from
 *     Families A-H KPASS which use Typesalt anyway; here KSALT also uses
 *     Typesalt because both have TYPEOPT_NEEDSALT (NOT NEEDUSER).
 *   - JOB_HMAC_STREEBOG512_KSALT (op=840): Typesalt
 *   - JOB_HMAC_STREEBOG512_KPASS (op=839): Typesalt
 *
 * All 5 ops route via the default Typesalt arm -- mirrors OpenCL twin
 * gpu/gpujob_opencl.c gpu_salt_judy() at line 215-229 where the same
 * five ops fall through to the default Typesalt arm (no explicit case).
 *
 * Future Phase 2d HMAC families whose KSALT/KPASS have TYPEOPT_NEEDUSER
 * (JOB_HMAC_MD5, JOB_HMAC_SHA1/224/256/384/512, JOB_HMAC_RMD160/320 KSALT
 * variants) will need an explicit Typeuser[op] case here -- but those
 * are NOT in Phase 2d.7d scope. Adding them later is mechanical: copy
 * the OpenCL twin's case list. */
static void *gpu_salt_judy(int op) {
    switch (op) {
    /* Future HMAC families with TYPEOPT_NEEDUSER (JOB_HMAC_MD5,
     * JOB_HMAC_SHA1, JOB_HMAC_SHA224, JOB_HMAC_SHA256, JOB_HMAC_SHA384,
     * JOB_HMAC_SHA512, JOB_HMAC_RMD160, JOB_HMAC_RMD320) will add
     * `case JOB_HMAC_<X>: return Typeuser ? Typeuser[op] : NULL;`
     * arms here when Metal ports admit them. Phase 2d.7d ships only
     * the Typesalt-routed HMAC siblings (BLAKE2S, STREEBOG-256, -512). */
    default:
        return Typesalt ? Typesalt[op] : NULL;
    }
}

/* gpu_compute_iter_sum: Metal twin of the same function in
 * gpu/gpujob_opencl.c. Computes the total internal iterations across
 * all packed salts for GPU Tothash accounting. See the OpenCL version
 * for full documentation. */
static uint64_t gpu_compute_iter_sum(int op, struct saltentry *saltsnap,
                                     int *pack_map, int nsalts_packed) {
    if (nsalts_packed <= 0) return 1;
    if (op == JOB_DESCRYPT)     return 25ULL * (uint64_t)nsalts_packed;
    if (op == JOB_MD5CRYPT)     return 1000ULL * (uint64_t)nsalts_packed;
    if (op == JOB_SHA1DRU)      return 1000000ULL * (uint64_t)nsalts_packed;
    uint64_t sum = 0;
    for (int i = 0; i < nsalts_packed; i++) {
        int si = pack_map[i];
        const char *s = saltsnap[si].salt;
        int sl = saltsnap[si].saltlen;
        uint32_t iter = 1;
        if (op == JOB_BCRYPT) {
            if (sl >= 6 && s[0] == '$' && s[2] == '$' && s[3] == '$') {
                int cost = (s[4] - '0') * 10 + (s[5] - '0');
                if (cost >= 4 && cost <= 31)
                    iter = 1u << cost;
            }
        } else if (op == JOB_PHPBB3) {
            if (sl >= 4) {
                int idx = i64hex[(unsigned char)s[3]];
                if (idx >= 7 && idx <= 30)
                    iter = 1u << idx;
            }
        } else if (op == JOB_SHA256CRYPT || op == JOB_SHA512CRYPT ||
                   op == JOB_SHA512CRYPTMD5) {
            iter = 5000;
            const char *p = s + 3;
            if (sl > 10 && strncmp(p, "rounds=", 7) == 0) {
                int n = atoi(p + 7);
                if (n >= 1000 && n <= 999999999)
                    iter = (uint32_t)n;
            }
        } else {
            iter = 1;
        }
        sum += (uint64_t)iter;
    }
    return sum;
}

/* gpu_pack_salts_op: Metal twin of the static helper in
 * gpu/gpujob_opencl.c at line 541. Packs a salt-snapshot into the
 * contiguous wire format (concatenated bytes + uint32 offsets + uint16
 * lengths + per-pack-slot back-map). Phase 2c drops the DESCRYPT
 * filter and the hashsalt branch — neither is reachable for JOB_MD5SALT
 * (the only op gated on by use_salt in 2c). Future Phase 2d siblings
 * needing those filters extend this fn alongside gpu_op_category. */
static int gpu_pack_salts_op(struct saltentry *saltsnap, int nsalts,
                             char *salts_packed, uint32_t *soff,
                             uint16_t *slen, int *pack_map,
                             int use_hashsalt, int op) {
    (void)op;
    int packed = 0;
    uint32_t gsp = 0;
    for (int i = 0; i < nsalts; i++) {
        if (!Printall && *saltsnap[i].PV == 0) continue;
        char *s = (use_hashsalt && saltsnap[i].hashsalt) ?
                  saltsnap[i].hashsalt : saltsnap[i].salt;
        int sl = (use_hashsalt && saltsnap[i].hashsalt) ?
                  32 : saltsnap[i].saltlen;
        soff[packed]     = gsp;
        slen[packed]     = sl;
        pack_map[packed] = i;
        memcpy(salts_packed + gsp, s, sl);
        gsp += sl;
        packed++;
    }
    return packed;
}

/* Pre-existing overflow-preload flag; declared int in gpujob_opencl.c.
 * Phase 2a doesn't separately load overflow (the OpenCL twin's
 * load_overflow() pumps a JudyL of unmatched hashes into
 * gpu_opencl_set_overflow; the Metal analog isn't exercised in Phase 2a
 * because the 2a smoke uses a 1k-hash compact with no overflow chain).
 * We keep the flag for compile-symmetry with mdxfind.c's overflow gate. */
int gpu_overflow_preloaded = 0;

/* ---- Trace-channel placeholders (memo §4 reshape) ----
 *
 * The OpenCL twin has MDXFIND_DISPATCH_TRACE / MDXFIND_PIPE_TRACE env
 * hooks for per-dispatch CSV logging. Phase 2a defers these. A
 * MDXFIND_METAL_TRACE env hook (memo §5) may be added in Phase 2c if
 * useful — for now the worker doesn't emit per-dispatch lines.
 */

/* ---- Time helper ---- */
static uint64_t gpu_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
}

/* ---- GPU work queue + free pools ----
 *
 * Layout mirrors gpujob_opencl.c lines 279-285 exactly; single device
 * collapses _gpujob_count to 1 + n_workers to 1, but the two-pool
 * (legacy + rules) structure is preserved so the slot_kind discriminator
 * still routes returns correctly. */
struct jobg *GPUWorkHead;
struct jobg **GPUWorkTail;
lock *GPUWorkWaiting;

struct jobg *GPULegacyFreeHead;
struct jobg **GPULegacyFreeTail;
lock *GPULegacyFreeWaiting;

struct jobg *GPURulesFreeHead;
struct jobg **GPURulesFreeTail;
lock *GPURulesFreeWaiting;

static int _gpujob_ready    = 0;
static int _num_legacy_jobg = 0;
static int _num_rules_jobg  = 0;
static int _gpu_batch_max   = GPUBATCH_RULE_MAX;
static thread *_worker      = NULL;

/* Per-device counters (single device — index 0 only). */
static uint64_t _gpu_words         = 0;
static uint64_t _gpu_rules_hashes  = 0;
static uint64_t _gpu_hits          = 0;
static uint64_t _gpu_batches       = 0;
static uint64_t _gpu_busy_us       = 0;
static uint64_t _gpu_first_us      = 0;
static uint64_t _gpu_last_us       = 0;

/* ---- gpujob_overflow_preload_all ----
 *
 * Phase 2a-final: walk JudyL OverflowHash + upload to GPU via
 * gpu_metal_set_overflow. Port of gpujob_opencl.c:582-637 (load_overflow).
 * Single-device on Metal so the per-device loop collapses to one call. */
extern void *OverflowHash;

void gpujob_overflow_preload_all(void) {
    if (gpu_overflow_preloaded) return;
    gpu_overflow_preloaded = 1;
    if (!OverflowHash) return;

    /* First pass: count entries + total padded bytes. Sub-128-bit entries
     * are zero-padded to 16 bytes so the GPU 4xuint32 probe doesn't spill
     * into the next entry — same fix as gpujob_opencl.c:592-606. */
    int ocnt = 0;
    size_t obytes = 0;
    Word_t okey = 0;
    Word_t *OPV = (Word_t *)JudyLFirst(OverflowHash, &okey, NULL);
    while (OPV) {
        struct Hashchain *chain = (struct Hashchain *)(*OPV);
        while (chain) {
            int pad_len = chain->len < 16 ? 16 : chain->len;
            ocnt++;
            obytes += pad_len;
            chain = chain->next;
        }
        OPV = (Word_t *)JudyLNext(OverflowHash, &okey, NULL);
    }
    if (ocnt == 0) return;

    uint64_t      *okeys    = (uint64_t *)malloc(ocnt * sizeof(uint64_t));
    unsigned char *ohashes  = (unsigned char *)malloc(obytes + 16);
    uint32_t      *ooffsets = (uint32_t *)malloc(ocnt * sizeof(uint32_t));
    uint16_t      *olengths = (uint16_t *)malloc(ocnt * sizeof(uint16_t));
    if (!okeys || !ohashes || !ooffsets || !olengths) {
        free(okeys); free(ohashes); free(ooffsets); free(olengths);
        fprintf(stderr, "Metal: overflow preload malloc failed (ocnt=%d)\n", ocnt);
        return;
    }
    memset(ohashes, 0, obytes + 16);

    /* Second pass: pack. */
    int oi = 0;
    uint32_t opos = 0;
    okey = 0;
    OPV = (Word_t *)JudyLFirst(OverflowHash, &okey, NULL);
    while (OPV) {
        struct Hashchain *chain = (struct Hashchain *)(*OPV);
        while (chain) {
            int pad_len = chain->len < 16 ? 16 : chain->len;
            okeys[oi]    = okey;
            ooffsets[oi] = opos;
            olengths[oi] = chain->len;
            memcpy(ohashes + opos, &okey, 8);
            if (chain->len > 8)
                memcpy(ohashes + opos + 8, chain->hash, chain->len - 8);
            opos += pad_len;
            oi++;
            chain = chain->next;
        }
        OPV = (Word_t *)JudyLNext(OverflowHash, &okey, NULL);
    }

    gpu_metal_set_overflow(0, okeys, ohashes, ooffsets, olengths, ocnt);
    GPU_DEBUG_FPRINTF(stderr, "Metal: overflow preload (%d entries, %zu bytes) uploaded\n",
            ocnt, obytes);
    free(okeys); free(ohashes); free(ooffsets); free(olengths);
}

/* ---- Free-pool helpers ---- */

static struct jobg *_gpujob_get_free_kind(char *filename, unsigned long long startline,
                                          enum jobg_kind kind) {
    (void)filename;
    (void)startline;

    if (MDXpause) {
        __sync_fetch_and_add(&MDXpaused_count, 1);
        while (MDXpause) sleep(2);
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
        g->t_acquired = gpu_now_us();
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

void gpujob_submit(struct jobg *g) {
    g->t_added = gpu_now_us();
    g->next = NULL;
    possess(GPUWorkWaiting);
    *GPUWorkTail = g;
    GPUWorkTail = &(g->next);
    twist(GPUWorkWaiting, BY, +1);
}

void gpujob_return_free(struct jobg *g) {
    g->next = NULL;
    g->count = 0;
    g->passbuf_pos = 0;
    g->packed = 0;
    g->packed_count = 0;
    g->packed_pos = 0;
    g->rules_engine = 0;
    g->bf_chunk = 0;
    g->bf_mask_start = 0;
    g->bf_offset_per_word = 0;
    g->bf_num_masks = 0;
    g->bf_inner_iter = 0;
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

/* ---- Worker thread ----
 *
 * Single worker (Metal is single-device Phase 2a). The slot's
 * gpu_metal_dispatch_md5_rules call blocks via
 * `[commandBuffer waitUntilCompleted]` inside the gpu_metal.m
 * implementation (memo §4 OpenCL clFinish analog), so we don't need
 * any extra synchronization at this layer.
 *
 * `slot` argument is unused (single device — mirrors OpenCL twin's
 * my_slot=int parameter passed via launch but kept here for signature
 * compatibility with launch's payload contract). */
static void gpujob_metal_worker(void *arg) {
    (void)arg;
    union HashU curin;
    struct job synthetic_job;
    char *outbuf = (char *)malloc_lock(OUTBUFSIZE + 1024, "gpujob_metal");
    uint64_t hashcnt = 0, found = 0;

    memset(&synthetic_job, 0, sizeof(synthetic_job));
    synthetic_job.outbuf = outbuf;

    /* Phase 2c row 18: per-worker salt-snapshot scratch buffers. Mirrors
     * gpu/gpujob_opencl.c lines 682-694 exactly. Sized to the maximums
     * computed at gpujob_init time (_max_salt_count / _max_salt_bytes).
     * Phase 2c has one worker; multi-worker phases will replicate this
     * block per thread. */
    struct saltentry *saltsnap =
        (struct saltentry *)malloc_lock(
            (size_t)_max_salt_count * sizeof(struct saltentry), "saltentry");
    char *saltpool =
        (char *)malloc_lock(_max_salt_bytes + 16, "saltpool");
    size_t sp_size = _max_salt_bytes + 4096;
    if ((size_t)_max_salt_count * 32 + 4096 > sp_size)
        sp_size = (size_t)_max_salt_count * 32 + 4096;
    char *salts_packed = (char *)malloc_lock(sp_size, "salts_packed");
    uint32_t *soff =
        (uint32_t *)malloc_lock(_max_salt_count * sizeof(uint32_t), "gpujob_metal");
    uint16_t *slen =
        (uint16_t *)malloc_lock(_max_salt_count * sizeof(uint16_t), "gpujob_metal");
    int *pack_map =
        (int *)malloc_lock(_max_salt_count * sizeof(int), "gpujob_metal");
    int nsalts = 0;
    int nsalts_packed = 0;
    int current_op = -1;
    int batch_count = 0;
    /* Phase 2g 2026-05-18: hybrid salt-refresh trigger counters. Refresh
     * fires on (batch_count >= 10) OR (>=5% of salts retired since the
     * last refresh). Counter is bumped by MTL_PV_DEC's 1->0 transition
     * return value at each hit-replay site. Resets to 0 at each refresh
     * along with batch_count. Per-worker scope (single-thread Metal). */
    int _salts_at_last_refresh    = 0;
    int _salts_retired_since_refresh = 0;
    char tsalt[8192];
    tsalt[0] = 0;

    while (1) {
        possess(GPUWorkWaiting);
        wait_for(GPUWorkWaiting, NOT_TO_BE, 0);
        struct jobg *g = GPUWorkHead;
        GPUWorkHead = g->next;
        if (GPUWorkHead == NULL)
            GPUWorkTail = &GPUWorkHead;
        twist(GPUWorkWaiting, BY, -1);

        g->t_dispatched = gpu_now_us();

        /* --- Sentinel handling --- */
        if (g->op == 2000) {
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

        int nhits = 0;
        uint32_t *hits = NULL;

        if (g->count == 0 && !g->packed) goto return_jobg;

        synthetic_job.op = g->op;
        synthetic_job.flags = g->flags;
        synthetic_job.filename = g->filename;
        synthetic_job.doneprint = g->doneprint;
        synthetic_job.found = (unsigned int *)&found;
        synthetic_job.outlen = 0;

        /* Phase 2c row 19: salt-snapshot + pack + upload. Mirrors
         * gpu/gpujob_opencl.c lines 757-847 (with the multi-axis salt-
         * judy / hashsalt / KSALT switch collapsed for Phase 2c -- only
         * JOB_MD5SALT routes through here, default Typesalt arm). The
         * trigger fires when op_cat == GPU_CAT_MASK AND
         * gpu_salt_judy(op) != NULL (mirrors gpujob_opencl.c:768).
         *
         * Refresh policy: rebuild on op change OR after 10 batches
         * (matches OpenCL twin behaviour). For Phase 2c with -i 1
         * (Maxiter=1, no iteration loop) salt_refresh is never set, so
         * the refresh-after-N-batches branch is the active recycler.
         */
        int op_cat = gpu_op_category(g->op);
        int needs_salt_snapshot =
            (op_cat == GPU_CAT_MASK && gpu_salt_judy(g->op) != NULL);

        if (needs_salt_snapshot) {
            batch_count++;
            /* Phase 2g hybrid refresh trigger: fire on op change, every 10
             * batches (existing cadence), OR when >= 5% of the last-uploaded
             * snapshot has retired (MTL_PV_DEC 1->0 transitions counted at
             * hit-replay sites). The retirement test uses int*20 >= total to
             * avoid float; 5% catches late-game salt churn where the 10-batch
             * cadence would compute against 10-20% dead snapshots. */
            int retirement_trigger = (_salts_at_last_refresh > 0 &&
                _salts_retired_since_refresh * 20 >= _salts_at_last_refresh);
            if (g->op != current_op ||
                ((batch_count >= 10 || retirement_trigger) && nsalts_packed > 0)) {
                if (g->op != current_op) {
                    current_op = g->op;
                    gpu_metal_set_op(0, g->op);
                }
                batch_count = 0;
                tsalt[0] = 0;
                nsalts = build_salt_snapshot(saltsnap, saltpool,
                                              gpu_salt_judy(g->op),
                                              tsalt, Printall);
                if (nsalts > 0) {
                    nsalts_packed = gpu_pack_salts_op(saltsnap, nsalts,
                                                      salts_packed, soff, slen,
                                                      pack_map,
                                                      /* use_hashsalt */ 0,
                                                      g->op);
                } else {
                    nsalts_packed = 0;
                }
                if (nsalts_packed > 0)
                    gpu_metal_set_salt(salts_packed, soff, slen, nsalts_packed);
                else
                    Typedone[g->op] = 1;
                /* Reset retirement counter at refresh boundary; bookmark
                 * the just-uploaded snapshot size for next cycle's 5% test. */
                _salts_at_last_refresh    = nsalts_packed;
                _salts_retired_since_refresh = 0;
            }
        } else if (g->op != current_op) {
            current_op = g->op;
            gpu_metal_set_op(0, g->op);
        }

        /* Stale-rebuild fallback (mirrors gpujob_opencl.c:829-847):
         * if needs_salt_snapshot but nsalts_packed dropped to 0
         * (e.g., concurrent Typedone), force one more snapshot before
         * giving up. */
        if (needs_salt_snapshot && nsalts_packed == 0) {
            nsalts = build_salt_snapshot(saltsnap, saltpool,
                                         gpu_salt_judy(g->op),
                                         tsalt, Printall);
            if (nsalts > 0) {
                nsalts_packed = gpu_pack_salts_op(saltsnap, nsalts,
                                                  salts_packed, soff, slen,
                                                  pack_map,
                                                  /* use_hashsalt */ 0,
                                                  g->op);
                if (nsalts_packed > 0)
                    gpu_metal_set_salt(salts_packed, soff, slen, nsalts_packed);
            }
            if (nsalts_packed == 0) {
                goto return_jobg;
            }
            /* Phase 2g: stale-rebuild also resets retirement counters. */
            _salts_at_last_refresh    = nsalts_packed;
            _salts_retired_since_refresh = 0;
        }

        /* --- Packed dispatch (rules-engine only Phase 2a) ---
         *
         * The OpenCL twin asserts `g->packed && !g->rules_engine` is
         * structurally unreachable (post-B7.9). Same invariant here.
         * For Phase 2a non-rules workloads (`-r best64.rule` is the
         * 2a-1 gate workload) the chokepoint pack at mdxfind.c sets
         * g->rules_engine=1.
         *
         * Non-packed slots (g->packed=0): legacy slab path was retired
         * in OpenCL Tranche B; same here. We just return-free. */
        if (g->packed && g->packed_count > 0 && g->rules_engine) {
            _gpu_batches++;
            uint64_t _disp_t0 = gpu_now_us();
            if (_gpu_first_us == 0) _gpu_first_us = _disp_t0;

            hits = gpu_metal_dispatch_md5_rules(
                0 /* dev_idx */,
                g->packed_buf, g->packed_pos,
                g->word_offset, g->packed_count,
                g->op, &nhits,
                0 /* mask_start */,
                0 /* mask_offset_per_word */,
                0 /* bf_num_masks */,
                0 /* inner_iter */,
                0 /* bf_fast_eligible */);

            uint64_t _disp_t1 = gpu_now_us();
            _gpu_words      += g->packed_count;
            if (nhits > 0) _gpu_hits += (uint64_t)nhits;
            _gpu_busy_us    += (_disp_t1 - _disp_t0);
            _gpu_last_us     = _disp_t1;

            /* Simulated candidate count (Totrules_gpu): packed_count *
             * (gpu_rule_count - 1) — minus 1 for the synthetic `:`
             * no-rule pass that mdxfind.c prepends; matches the OpenCL
             * twin at line 993. */
            atomic_fetch_add(&Totrules_gpu,
                (unsigned long long)g->packed_count *
                (unsigned long long)(gpu_rule_count > 1 ? gpu_rule_count - 1 : 0));

            /* --- Hit-replay (Phase 2a: unsalted MD5 only) ---
             *
             * The OpenCL twin (gpu/gpujob_opencl.c lines 1080-2358) has
             * a 3-axis decompose for (salt_idx_global, mask_idx, ridx).
             * Phase 2a collapses to a single axis: mask_size=1,
             * nsalts=1, so combined_ridx == ridx == rule_idx.
             *
             * Thread-local applyrule scratch space + rule-pointer
             * cache mirror the OpenCL twin's pattern (line 1003).
             */
            if (hits && nhits > 0) {
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

                static __thread struct rule_workspace _ws;
                static __thread char _tpass[MAXLINE + 64];

                int stored = nhits;
                if (stored > GPU_PACKED_MAX_HITS) stored = GPU_PACKED_MAX_HITS;
                /* hexlen = ASCII hex length of the digest =
                 * metal_gpu_hash_words(op) * 8 (each uint32 = 8 hex chars).
                 * MD5 (4 words) -> 32, SHA-1 (5 words) -> 40, SHA-224 (7) -> 56,
                 * SHA-256 / BLAKE2S / Keccak-256 / Streebog-256 / HMAC-{BLAKE2S,
                 * STREEBOG256} (8) -> 64, SHA-384/Keccak-384 (12) -> 96, SHA-512/
                 * BLAKE2B / Keccak-512 / Streebog-512 / HMAC-STREEBOG512 (16) ->
                 * 128. Mirrors OpenCL twin gpu/gpujob_opencl.c:1027
                 * (`int hexlen = gpu_hash_words(g->op) * 8;`). Was previously
                 * hardcoded to 32 (MD5-only) -- worked for SHA-2 families because
                 * checkhashsalt/checkhashkey use len only for output formatting
                 * + saltbuf composition (the digest probe via curin.h is byte-
                 * exact regardless of len), but the output format used the wrong
                 * width. Phase 2d.7d HMAC siblings need this corrected so
                 * HMAC-STREEBOG-512 output formats at 128 hex chars, not 32. */
                int hexlen = metal_gpu_hash_words(g->op) * 8;

                /* Phase 2b row 5: mask-active gate. When a mask was uploaded
                 * via gpu_metal_set_mask AND it has more than one combo, the
                 * kernel packed combined_ridx = rule_idx * mask_size +
                 * mask_idx_local. Host divmod recovers both axes; then the
                 * mask bytes are prepended/appended to synthetic_job.line
                 * BEFORE checkhash. Mirrors gpu/gpujob_opencl.c lines
                 * 1066-1610 (WITHOUT BF terms — Metal has no BF in Phase 2b;
                 * bf_mask_start / bf_offset_per_word read zero). */
                int b71_mask_active =
                    (gpu_mask_n_prepend >= 0 && gpu_mask_n_prepend <= 16
                     && gpu_mask_n_append >= 0 && gpu_mask_n_append <= 16
                     && (gpu_mask_n_prepend + gpu_mask_n_append) >= 1
                     && gpu_mask_total > 0);
                uint64_t b71_mask_size = b71_mask_active ? gpu_mask_total : 1u;

                /* Phase 2c: salt-axis decompose. is_salted_op gates the
                 * 3-axis divmod (salt_local, mask_idx, ridx) per
                 * gpu/gpujob_opencl.c:1114-1426. Phase 2c admitted
                 * JOB_MD5SALT only; Phase 2d.2.4 widened to JOB_MD5PASSSALT;
                 * Phase 2d.2.5 widened to JOB_MD5SALTPASS (LAST md5-family
                 * fan-out entry — MD5(salt || pass) PREPEND, mirror of
                 * md5passsalt's APPEND). Phase 2d.3.4 widens to
                 * JOB_SHA1PASSSALT; Phase 2d.3.5 widens to JOB_SHA1SALTPASS
                 * (FIRST SHA-family salted ports on the cl2metal codegen
                 * path). Future 2d siblings extend this list one op at a
                 * time per feedback_metal_is_salted_op_widening.md.
                 *
                 * nsalts_for_decode is nsalts_packed (the full salt count
                 * because num_salts_per_page=nsalts_packed -- one dispatch
                 * covers all salts). For unsalted ops nsalts_for_decode
                 * is 1 and the salt-axis divmod reduces to no-op. */
                int is_salted_op = (g->op == JOB_MD5SALT ||
                                    g->op == JOB_MD5PASSSALT ||
                                    g->op == JOB_MD5SALTPASS ||
                                    g->op == JOB_SHA1PASSSALT ||
                                    g->op == JOB_SHA1SALTPASS ||
                                    /* Forward-stage 2d.4.x SHA-2/224+256 salted family. */
                                    g->op == JOB_SHA224SALTPASS ||
                                    g->op == JOB_SHA256SALTPASS ||
                                    g->op == JOB_SHA256PASSSALT ||
                                    /* Phase 2d.5.5 SHA-2/512 salted family. */
                                    g->op == JOB_SHA512PASSSALT ||
                                    /* Phase 2d.5.6 SHA-2/512 salted PREPEND. */
                                    g->op == JOB_SHA512SALTPASS ||
                                    /* Phase 2d.5.7 SHA-2/384 salted PREPEND (LAST 2d.5). */
                                    g->op == JOB_SHA384SALTPASS ||
                                    /* Phase 2d.7d HMAC siblings (5 ops, 3 carrier
                                     * kernels). All carriers consume the standard
                                     * salt_buf/salt_off/salt_lens trio via the
                                     * salted-template wiring; hit-replay routes
                                     * through checkhashsalt (iter=0 for all
                                     * HMAC ops per CPU semantics). Per
                                     * feedback_metal_is_salted_op_widening.md
                                     * this OR-list gate is structural for the
                                     * Metal salt-snapshot upload + hit-replay
                                     * routing path. */
                                    g->op == JOB_HMAC_BLAKE2S ||
                                    g->op == JOB_HMAC_STREEBOG256_KSALT ||
                                    g->op == JOB_HMAC_STREEBOG256_KPASS ||
                                    g->op == JOB_HMAC_STREEBOG512_KSALT ||
                                    g->op == JOB_HMAC_STREEBOG512_KPASS ||
                                    /* Phase 2d.8a: PHPBB3 + MD5CRYPT both
                                     * salted-only iterated-MD5 ops. Hit-
                                     * replay arms route differently from
                                     * the standard checkhashsalt/key path
                                     * (PHPBB3 -> checkhashbb; MD5CRYPT ->
                                     * hybrid_check + md5crypt_b64encode +
                                     * prfound). is_salted_op MUST admit
                                     * both so the salt-snapshot upload +
                                     * salt-entry resolve fire correctly.
                                     * Per feedback_metal_is_salted_op_-
                                     * widening.md. */
                                    g->op == JOB_PHPBB3 ||
                                    g->op == JOB_MD5CRYPT ||
                                    /* Phase 2d.8b SHACRYPT triple: SHA256-
                                     * CRYPT (op=512), SHA512CRYPT (op=
                                     * 513), and SHA512CRYPTMD5 (op=538).
                                     * All salted-only ops with Typesalt
                                     * carrying the full "$5$..." / "$6$
                                     * ..." line. Hit-replay arms below
                                     * route via hybrid_check + sha-
                                     * {256,512}crypt_b64encode + prfound
                                     * (mirrors gpujob_opencl.c salt-
                                     * snapshot routing). Per feedback_-
                                     * metal_is_salted_op_widening.md
                                     * Phase 2d.8b. */
                                    g->op == JOB_SHA256CRYPT ||
                                    g->op == JOB_SHA512CRYPT ||
                                    g->op == JOB_SHA512CRYPTMD5 ||
                                    /* Phase 2d.9a DESCRYPT (op=500):
                                     * salted-only single-mode (7) DES.
                                     * Salt buffer carries the 2-char
                                     * phpitoa64 salt. Hit-replay arm
                                     * below routes via metal_des_-
                                     * reconstruct + JudyJ[JOB_DESCRYPT]
                                     * + prfound (NOT through checkhash-
                                     * bb/salt/key). Per feedback_metal_-
                                     * is_salted_op_widening.md the is_-
                                     * salted_op gate MUST admit the new
                                     * op or the salt-snapshot upload +
                                     * hit-replay salt-entry resolve will
                                     * not fire. Phase 5 of Unix-crypt
                                     * ladder on Metal -- LAST Unix-crypt
                                     * op to migrate. */
                                    g->op == JOB_DESCRYPT ||
                                    /* Phase 2d.9b BCRYPT (op=450):
                                     * salted-only single-mode (8)
                                     * Eksblowfish. Salt buffer carries
                                     * the full 60-char $2[abxy]$NN$
                                     * crypt(3) line; kernel decodes cost
                                     * + raw 16-byte salt inside
                                     * template_finalize. Hit-replay arm
                                     * below routes via bf_encode_23 +
                                     * JudyJ[JOB_BCRYPT] + prfound (NOT
                                     * through checkhashbb/salt/key).
                                     * FINAL Phase 2d sub-phase. */
                                    g->op == JOB_BCRYPT);
                uint32_t nsalts_for_decode =
                    is_salted_op ? (uint32_t)nsalts_packed : 1u;
                if (nsalts_for_decode == 0u) nsalts_for_decode = 1u;

                /* Phase 2b: BF chunk-as-job is OUT OF SCOPE on Metal. The
                 * OpenCL twin's b71_mask_size = g->bf_num_masks branch
                 * (gpujob_opencl.c:1073-1075) does NOT apply here. Guard
                 * the BF chunk flag for symmetry with the OpenCL twin so
                 * a stray Phase 2c+ enablement is caught early. */
                if (g->bf_chunk) {
                    /* UNREACHABLE in Phase 2b — production gate at the
                     * chokepoint pack rejects BF for the Metal arm. If we
                     * ever land a slot with bf_chunk=1 it's a Phase 2c+
                     * regression; log once and proceed with mask_size=1
                     * decode (safe fallback). */
                    static int _bf_unexpected = 0;
                    if (!_bf_unexpected) {
                        _bf_unexpected = 1;
                        fprintf(stderr, "Metal: gpujob_metal: bf_chunk=1 "
                                "slot received but BF is out-of-scope in "
                                "Phase 2b — falling back to mask_size=1 "
                                "decode (no BF re-add to mask_idx_abs).\n");
                    }
                }

                for (int h = 0; h < stored; h++) {
                    uint32_t *entry = hits + h * GPU_HIT_STRIDE;
                    uint32_t widx          = entry[0];
                    uint32_t combined_ridx = entry[1];
                    int      iter_num      = (int)entry[2];

                    /* Phase 2c three-axis decompose. Kernel packed:
                     *   combined_ridx = ((rule_idx * mask_size +
                     *                     mask_idx_local) *
                     *                    num_salts_per_page) + salt_local
                     * where num_salts_per_page = nsalts_packed in Phase 2c
                     * (one dispatch covers the whole list). Recover
                     * salt_local first (mod nsalts_for_decode), then
                     * mask_idx (mod mask_size), then ridx (remainder).
                     * For unsalted ops nsalts_for_decode==1 so the
                     * salt mod is always 0 and ridx/mask collapse to
                     * Phase 2b layout bit-identical. */
                    uint32_t salt_idx_global =
                        (uint32_t)((uint64_t)combined_ridx % nsalts_for_decode);
                    uint32_t tmp =
                        (uint32_t)((uint64_t)combined_ridx / nsalts_for_decode);
                    uint32_t mask_idx = (uint32_t)((uint64_t)tmp % b71_mask_size);
                    uint32_t ridx     = (uint32_t)((uint64_t)tmp / b71_mask_size);

                    if (widx >= g->packed_count) continue;
                    if ((int)ridx >= gpu_rule_count) continue;
                    if (b71_mask_active && (uint64_t)mask_idx >= b71_mask_size) continue;
                    if (is_salted_op &&
                        (int)salt_idx_global >= nsalts_packed) continue;

                    /* Phase 2c salt-entry resolve. pack_map[salt_idx_global]
                     * -> snap_idx -> saltsnap[snap_idx] yields the salt
                     * bytes for checkhashsalt. Mirrors gpu/gpujob_opencl.c
                     * lines 1465-1475. */
                    char *salt_bytes = NULL;
                    int   salt_len_b = 0;
                    struct saltentry *salt_snap_entry = NULL;
                    if (is_salted_op) {
                        int snap_idx = pack_map[salt_idx_global];
                        if (snap_idx < 0) continue;
                        salt_snap_entry = &saltsnap[snap_idx];
                        salt_bytes      = salt_snap_entry->salt;
                        salt_len_b      = salt_snap_entry->saltlen;
                    }

                    /* Decode the candidate hash from the hit entry. Width
                     * depends on op: MD5/MD4 family = 4 words; SHA-1 = 5
                     * words; future SHA families up to 16 (SHA-512). Per
                     * metal_gpu_hash_words() above; mirrors the OpenCL
                     * twin gpu/gpujob_opencl.c:1478-1481. The kernel
                     * EMIT_HIT_N variant emits exactly N digest words at
                     * hits[_base+3..+3+N-1]; reading the wrong width
                     * leaves curin.i[N..16] uninitialized and
                     * hybrid_check's full-digest memcmp fails — that was
                     * the Phase 2d.3.1 SHA-1 canary bug. */
                    {
                        int hw = metal_gpu_hash_words(g->op);
                        if (hw > 16) hw = 16;  /* defensive: curin.i has 16 uint32 */
                        for (int w = 0; w < hw; w++) curin.i[w] = entry[3 + w];
                    }

                    /* Recover original word from packed_buf. */
                    uint32_t pos = g->word_offset[widx];
                    if (pos >= g->packed_pos) continue;
                    uint8_t plen = (uint8_t)g->packed_buf[pos];
                    if (pos + 1 + plen > g->packed_pos) continue;
                    char *pword = g->packed_buf + pos + 1;

                    /* Map GPU rule index -> original Rules[] index.
                     * orig_idx == -1 sentinel: synthetic `:` no-rule pass —
                     * skip applyrule replay and use word directly. */
                    int orig_idx = gpu_rule_origin[ridx];
                    int out_len;
                    if (orig_idx == -1) {
                        memcpy(synthetic_job.line, pword, plen);
                        synthetic_job.line[plen] = 0;
                        out_len = (int)plen;
                        synthetic_job.Ruleindex = 0;
                    } else {
                        if (orig_idx < 0 || orig_idx >= (int)Numrules) continue;

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
                        } else if (new_len < 0) {
                            continue;
                        } else {
                            memcpy(synthetic_job.line, _tpass, new_len);
                            synthetic_job.line[new_len] = 0;
                            out_len = new_len;
                        }
                        synthetic_job.Ruleindex = orig_idx;
                    }

                    /* Phase 2b row 5: prepend+append the mask characters to
                     * the candidate plaintext. Mirrors gpu/gpujob_opencl.c
                     * lines 1555-1663 (WITHOUT the BF mask_idx_abs re-add —
                     * Phase 2b has no BF on Metal so mask_idx_abs ==
                     * mask_idx_local). */
                    if (b71_mask_active) {
                        int npre = gpu_mask_n_prepend;
                        int napp = gpu_mask_n_append;
                        if (npre > 16) npre = 16;
                        if (napp > 16) napp = 16;
                        /* append_combos = product(sizes[npre..npre+napp)). */
                        uint64_t append_combos = 1u;
                        for (int j = 0; j < napp; j++) {
                            int sz = gpu_mask_sizes[npre + j];
                            if (sz <= 0) sz = 1;
                            append_combos *= (uint64_t)sz;
                        }
                        if (append_combos == 0u) append_combos = 1u;
                        /* Phase 2b: mask_idx_abs == mask_idx_local (no BF). */
                        uint64_t mask_idx_abs = (uint64_t)mask_idx;
                        uint64_t prepend_idx  = mask_idx_abs / append_combos;
                        uint64_t append_idx   = mask_idx_abs % append_combos;

                        char prepend_chars[16];
                        char append_chars[16];
                        /* Decode prepend chars (positions [0..npre) in
                         * mask_charsets layout). Last position innermost. */
                        {
                            uint64_t remaining = prepend_idx;
                            for (int k = 0; k < npre; k++) {
                                int i = npre - 1 - k;
                                int sz = gpu_mask_sizes[i];
                                if (sz <= 0) sz = 1;
                                int pidx = (int)(remaining % (uint64_t)sz);
                                remaining /= (uint64_t)sz;
                                prepend_chars[i] = (char)gpu_mask_charsets_host[i][pidx];
                            }
                        }
                        /* Decode append chars (positions [npre..npre+napp)). */
                        {
                            uint64_t remaining = append_idx;
                            for (int k = 0; k < napp; k++) {
                                int i = napp - 1 - k;
                                int row = npre + i;
                                int sz = gpu_mask_sizes[row];
                                if (sz <= 0) sz = 1;
                                int pidx = (int)(remaining % (uint64_t)sz);
                                remaining /= (uint64_t)sz;
                                append_chars[i] = (char)gpu_mask_charsets_host[row][pidx];
                            }
                        }
                        /* Assemble [prepend][rule_output][append] in
                         * synthetic_job.line. Shift rule output right by
                         * npre via memmove (handles overlap), then write
                         * prepend at the front and append at the end. */
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

                    synthetic_job.clen = out_len;
                    synthetic_job.pass = synthetic_job.line;

                    /* Phase 2c: salted dispatch routing. Mirrors
                     * gpu/gpujob_opencl.c:1700-2358 with the iter-aware
                     * switch collapsed.
                     *   JOB_MD5SALT iter==1 -> checkhashkey (no iter
                     *     suffix in label; matches CPU MD5SALT path at
                     *     mdxfind.c:22198).
                     *   JOB_MD5SALT iter > 1 -> checkhashsalt (xNN
                     *     suffix; matches CPU MD5SALT path at
                     *     mdxfind.c:22207).
                     *   JOB_MD5PASSSALT (Phase 2d.2.4) -> always
                     *     checkhashsalt — no iter==1 special case (mirrors
                     *     gpujob_opencl.c:1114-1700 routing; CPU path at
                     *     mdxfind.c:16655 always emits MD5PASSSALTxNN).
                     *   JOB_MD5SALTPASS (Phase 2d.2.5) -> always
                     *     checkhashsalt — same routing as MD5PASSSALT
                     *     (CPU path at mdxfind.c:16884-16914 always emits
                     *     MD5SALTPASSxNN via checkhashsalt, no iter==1
                     *     special case).
                     * On successful match, MTL_PV_DEC the salt's pending
                     * counter so subsequent dispatches skip it. */
                    if (is_salted_op) {
                        int hit = 0;
                        if (g->op == JOB_MD5SALT && iter_num == 1) {
                            hit = checkhashkey(&curin, hexlen,
                                               salt_bytes,
                                               &synthetic_job);
                        } else if (g->op == JOB_HMAC_BLAKE2S ||
                                   g->op == JOB_HMAC_STREEBOG256_KSALT ||
                                   g->op == JOB_HMAC_STREEBOG256_KPASS ||
                                   g->op == JOB_HMAC_STREEBOG512_KSALT ||
                                   g->op == JOB_HMAC_STREEBOG512_KPASS) {
                            /* Phase 2d.7d: HMAC ops route via checkhashsalt
                             * with iter=0 (NOT iter_num). max_iter is forced
                             * to 1 host-side; the HMAC body runs inside
                             * template_finalize, kernel emits iter_num=1, but
                             * the CPU output convention is iter=0 (no `xNN`
                             * suffix). Mirrors gpu/gpujob_opencl.c:1903-1947
                             * routing for the 5 HMAC ops. */
                            hit = checkhashsalt(&curin, hexlen,
                                                salt_bytes, salt_len_b,
                                                0,
                                                &synthetic_job);
                        } else if (g->op == JOB_PHPBB3) {
                            /* Phase 2d.8a.1: PHPBB3 routing. CPU semantics
                             * at mdxfind.c:13620 calls checkhashbb(curin,
                             * 32, s1, job) where s1 is the full 12-byte
                             * "$H$<cost><8>" salt prefix from
                             * saltsnap[si].salt. The hit-replay arm
                             * mirrors gpu/gpujob_opencl.c:1964-1967 byte-
                             * for-byte (the slab-path arm at line 1682
                             * also used checkhashbb). hexlen=32 = 32 hex
                             * chars / 16 bytes (4 LE uint32 = HASH_WORDS=4
                             * = MD5 width, which is the metal_gpu_hash_-
                             * words default arm return value). max_iter
                             * is forced to 1 host-side; the algorithm's
                             * internal iter count is decoded from
                             * salt_bytes[3] inside template_finalize. NOT
                             * routed through checkhashkey/checkhashsalt
                             * because PHPBB3 has its own bb-specific
                             * output format (phpitoa64-encoded 22-char
                             * hash + the salt prefix), distinct from the
                             * HMAC families' hex-encoded outputs. */
                            hit = checkhashbb(&curin, hexlen,
                                              salt_bytes,
                                              &synthetic_job);
                        } else if (g->op == JOB_MD5CRYPT) {
                            /* Phase 2d.8a.2: MD5CRYPT routing. CPU
                             * semantics at mdxfind.c:13071 calls
                             * hybrid_check(curin.h, 16, &match_len,
                             * &match_flags) on the 16-byte binary MD5
                             * digest, then reconstructs "$1$<salt>$
                             * <22-char-phpitoa64>" via the
                             * md5crypt_b64encode helper. We mirror the
                             * OpenCL twin arm at gpujob_opencl.c:1986-
                             * 2008 byte-for-byte (probe + reconstruct +
                             * prfound). NOT routed through checkhashbb /
                             * checkhashkey / checkhashsalt because
                             * MD5CRYPT has its own bespoke output format
                             * (phpitoa64-encoded 22-char hash with custom
                             * byte permutation distinct from PHPBB3's).
                             * hit stays 0: this arm does its own
                             * MTL_PV_DEC + prfound inline (mirroring the
                             * OpenCL twin's slab arm); the outer salt-
                             * snap MTL_PV_DEC at "if (hit &&
                             * salt_snap_entry)" is gated by hit==0, so
                             * leaving hit=0 keeps PV accounting
                             * single-path. */
                            hit = 0;
                            if (salt_snap_entry) {
                                int match_len;
                                unsigned short *match_flags;
                                int hf = hybrid_check((const unsigned char *)curin.h,
                                                      16,
                                                      &match_len, &match_flags);
                                if (hf && *match_flags != (unsigned short)g->op) {
                                    *match_flags = g->op;
                                    if (MTL_PV_DEC(salt_snap_entry->PV)) _salts_retired_since_refresh++;
                                    char *sp = salt_snap_entry->salt;
                                    int splen = salt_snap_entry->saltlen;
                                    char mdbuf[128];
                                    memcpy(mdbuf, sp, splen);
                                    md5crypt_b64encode((const unsigned char *)curin.h,
                                                       mdbuf + splen);
                                    prfound(&synthetic_job, mdbuf);
                                }
                            }
                        } else if (g->op == JOB_SHA256CRYPT) {
                            /* Phase 2d.8b: SHA256CRYPT routing. CPU
                             * semantics at mdxfind.c:12290 (cryptlen=32
                             * branch) computes 32-byte SHA-256 digest
                             * in curin.h, then reconstructs "$5$[rounds
                             * =N$]<salt>$<43-base64>" via the cryptlen
                             * =32 b64 byte-permutation table at mdxfind
                             * .c:12753-12980 (wrapped in sha256crypt_-
                             * b64encode helper above). Mirrors gpujob_-
                             * opencl.c:2031-2073 byte-for-byte. NOT
                             * routed through checkhashbb / checkhashkey
                             * / checkhashsalt because SHA256CRYPT has
                             * its own bespoke output format (43-char
                             * base64 with the 32-byte permutation
                             * distinct from MD5CRYPT's 22-char or
                             * PHPBB3's 22-char encodings). hit stays 0:
                             * this arm does its own MTL_PV_DEC + prfound
                             * inline; outer salt-snap MTL_PV_DEC gate is
                             * `hit && salt_snap_entry`, leaving hit=0
                             * keeps PV accounting single-path. The last-
                             * `$` prefix-len scan handles "$5$[rounds=N
                             * $]<salt>$<43-b64>" -- phpitoa64 alphabet
                             * (./0-9A-Za-z) excludes '$', so the final
                             * '$' reliably terminates the salt prefix.
                             * Output buffer: salt prefix (up to ~30
                             * bytes incl rounds=N$) + 43-char b64 + NUL
                             * -- mdbuf[128] is comfortably oversized.
                             * Phase 2 of Unix-crypt ladder on Metal. */
                            hit = 0;
                            if (salt_snap_entry) {
                                int match_len;
                                unsigned short *match_flags;
                                int hf = hybrid_check((const unsigned char *)curin.h,
                                                      32,
                                                      &match_len, &match_flags);
                                if (hf && *match_flags != (unsigned short)g->op) {
                                    *match_flags = g->op;
                                    if (MTL_PV_DEC(salt_snap_entry->PV)) _salts_retired_since_refresh++;
                                    char *sp = salt_snap_entry->salt;
                                    int splen = salt_snap_entry->saltlen;
                                    char mdbuf[128];
                                    int prefix_len = splen;
                                    while (prefix_len > 0 && sp[prefix_len - 1] != '$')
                                        prefix_len--;
                                    memcpy(mdbuf, sp, prefix_len);
                                    sha256crypt_b64encode((const unsigned char *)curin.h,
                                                          mdbuf + prefix_len);
                                    prfound(&synthetic_job, mdbuf);
                                }
                            }
                        } else if (g->op == JOB_SHA512CRYPT ||
                                   g->op == JOB_SHA512CRYPTMD5) {
                            /* Phase 2d.8b: SHA512CRYPT + SHA512CRYPTMD5
                             * routing. CPU semantics at mdxfind.c:12290
                             * (cryptlen=64 branch) computes 64-byte SHA-
                             * 512 digest in curin.h, then reconstructs
                             * "$6$[rounds=N$]<salt>$<86-base64>" via the
                             * cryptlen=64 b64 byte-permutation table at
                             * mdxfind.c:12361-12780 (wrapped in
                             * sha512crypt_b64encode helper above).
                             * SHA512CRYPTMD5 shares this arm verbatim
                             * because BOTH ops produce the same $6$
                             * output format and Typesalt entries
                             * (mdxfind.c:47049-47087 inserts the SAME
                             * line into BOTH Judy arrays). The kernel-
                             * side MD5-preprocess (algo_mode=1u) is
                             * upstream of the SHA-crypt chain and does
                             * NOT change the output format. Mirrors
                             * gpujob_opencl.c:2127-2159 byte-for-byte.
                             * NOT routed through checkhashbb /
                             * checkhashkey / checkhashsalt because
                             * SHA512CRYPT has its own bespoke output
                             * format (86-char base64 with the 64-byte
                             * permutation distinct from SHA256CRYPT's
                             * 43-char or MD5CRYPT's 22-char encodings).
                             * hit stays 0: this arm does its own
                             * MTL_PV_DEC + prfound inline; outer salt-
                             * snap MTL_PV_DEC gate is `hit && salt_-
                             * snap_entry`, leaving hit=0 keeps PV
                             * accounting single-path. *match_flags =
                             * g->op preserves per-op deduplication (a
                             * $6$ line matched as JOB_SHA512CRYPT is
                             * not re-emitted as JOB_SHA512CRYPTMD5
                             * within the same hit record, and vice
                             * versa). Output buffer: salt prefix (up to
                             * ~30 bytes incl rounds=N$) + 86-char b64 +
                             * NUL = up to 117 bytes -- mdbuf[128] is
                             * comfortably oversized. Phases 3 + 4 of
                             * Unix-crypt ladder on Metal (FINAL
                             * phase). */
                            hit = 0;
                            if (salt_snap_entry) {
                                int match_len;
                                unsigned short *match_flags;
                                int hf = hybrid_check((const unsigned char *)curin.h,
                                                      64,
                                                      &match_len, &match_flags);
                                if (hf && *match_flags != (unsigned short)g->op) {
                                    *match_flags = g->op;
                                    if (MTL_PV_DEC(salt_snap_entry->PV)) _salts_retired_since_refresh++;
                                    char *sp = salt_snap_entry->salt;
                                    int splen = salt_snap_entry->saltlen;
                                    char mdbuf[128];
                                    int prefix_len = splen;
                                    while (prefix_len > 0 && sp[prefix_len - 1] != '$')
                                        prefix_len--;
                                    memcpy(mdbuf, sp, prefix_len);
                                    sha512crypt_b64encode((const unsigned char *)curin.h,
                                                          mdbuf + prefix_len);
                                    prfound(&synthetic_job, mdbuf);
                                }
                            }
                        } else if (g->op == JOB_BCRYPT) {
                            /* Phase 2d.9b: BCRYPT routing. The GPU
                             * emits 6 LE uint32 words = 24 bytes in
                             * curin.i[0..5] (raw byte stream after the
                             * kernel's BE->LE swap at metal_bcrypt_core
                             * step 8). Hit-replay reconstructs the
                             * 60-char "$2[abxy]$NN$<22-b64-salt>
                             * <31-b64-hash>" crypt(3) hash via metal_-
                             * bf_encode_23 (NEW helper above; duplicated
                             * with citation from gpu/gpujob_opencl.c
                             * bf_encode_23 per project_hx_algo_dedup.md
                             * discipline since the OpenCL helper is
                             * static and TU-local). Probes JudyJ[JOB_-
                             * BCRYPT] for the full 60-char string and
                             * uses CAS dedup (atomic compare-and-swap
                             * 0 -> 1 on the Judy value), mirroring the
                             * OpenCL twin's JOB_BCRYPT arm at gpu/
                             * gpujob_opencl.c:2284-2347 byte-for-byte.
                             * NOT routed through checkhashbb / check-
                             * hashkey / checkhashsalt because BCRYPT
                             * has its own bespoke output format (60-
                             * char crypt(3) hash, distinct from
                             * MD5CRYPT's $1$/PHPBB3's $H$ shapes).
                             * hit stays 0 (this arm does its own MTL_-
                             * PV_DEC + prfound inline; outer salt-snap
                             * MTL_PV_DEC gate is `hit && salt_snap_-
                             * entry`, leaving hit=0 keeps PV accounting
                             * single-path). Uses salt_snap_entry +
                             * salt_bytes + salt_len_b per feedback_-
                             * rules_engine_hit_replay_vars.md (rules-
                             * engine context, NOT slab `sidx`).
                             *
                             * Display password: FULL post-rule plaintext
                             * (NO clamp). CPU does NOT clamp display
                             * for BCRYPT -- the 72-byte truncation is
                             * INSIDE BF_set_key, not at display. Q1
                             * user decision 2026-05-08 (DIFFERENT from
                             * DESCRYPT's 8-byte clamp). The post-rule
                             * plaintext at synthetic_job.line is
                             * already at length out_len (populated
                             * above by either the synthetic-`:` arm or
                             * the applyrule replay arm + optional mask
                             * prepend/append); we re-emit it AS-IS
                             * with synthetic_job.clen = out_len. FINAL
                             * Phase 2d sub-phase -- 52nd Metal family. */
                            hit = 0;
                            if (salt_snap_entry) {
                                /* GPU emits 6 LE uint32 = 24 bytes in
                                 * curin.i[0..5]; cast to uchar for
                                 * byte-stream encoding. metal_bf_-
                                 * encode_23 reads first 23 bytes (24th
                                 * is zero pad from BE->LE swap tail
                                 * that BF_encode discards). */
                                unsigned char *raw = (unsigned char *)&curin.i[0];
                                char hashb64[32];
                                metal_bf_encode_23(raw, hashb64);
                                /* Build full 60-char hash: salt prefix
                                 * (28 or 29 chars) + 31-char b64 hash.
                                 * fullhash[80] is comfortably oversized
                                 * for the 60-char standard / 59-char
                                 * $2k variant. */
                                char fullhash[80];
                                int splen = salt_len_b;
                                /* salt_bytes in Typesalt[JOB_BCRYPT]
                                 * is the full 60-char hash line per
                                 * mdxfind.c BCRYPT loader; for
                                 * reconstruction we keep only the
                                 * first 28 (or 29 for $2k$) chars --
                                 * the part BEFORE the 31-char b64
                                 * hash. Mirrors gpu/gpujob_opencl.c
                                 * lines 2320-2325 byte-for-byte. */
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
                                    if (MTL_PV_DEC(salt_snap_entry->PV)) _salts_retired_since_refresh++;
                                    /* Q1 (2026-05-08): NO display clamp
                                     * (DIFFERENT from DESCRYPT's
                                     * 8-byte cap). Render full post-
                                     * rule plaintext as-is; clen is
                                     * already out_len. */
                                    synthetic_job.line[out_len] = 0;
                                    synthetic_job.clen = out_len;
                                    prfound(&synthetic_job, fullhash);
                                }
                            }
                        } else if (g->op == JOB_DESCRYPT) {
                            /* Phase 2d.9a: DESCRYPT routing. The GPU
                             * emits pre-FP (l, r) in curin.i[0..1]
                             * (h[2..3] are zero-padded by the kernel to
                             * match the host compact-table layout 4 il +
                             * 4 ir + 8 zero pad). Hit-replay reconstructs
                             * the 13-char crypt(3) hash via metal_des_-
                             * reconstruct (NEW helper above; ports
                             * gpu/gpujob_opencl.c des_reconstruct byte-
                             * for-byte via project_hx_algo_dedup.md
                             * citation since the OpenCL helper is static).
                             * Probes JudyJ[JOB_DESCRYPT] for the 13-char
                             * string and uses CAS dedup (atomic compare-
                             * and-swap 0 -> 1 on the Judy value),
                             * mirroring gpu/gpujob_opencl.c JOB_DESCRYPT
                             * arm at line 2204-2232 byte-for-byte. NOT
                             * routed through checkhashbb / checkhashkey
                             * / checkhashsalt because DESCRYPT has its
                             * own bespoke output format (13-char salt+
                             * 11-base64 crypt(3) hash, distinct from
                             * MD5CRYPT's 22-char b64 or PHPBB3's 22-char
                             * b64). hit stays 0 (this arm does its own
                             * MTL_PV_DEC + prfound inline; outer salt-
                             * snap MTL_PV_DEC gate is `hit && salt_snap_-
                             * entry`, leaving hit=0 keeps PV accounting
                             * single-path). Uses salt_snap_entry +
                             * salt_bytes per feedback_rules_engine_hit_-
                             * replay_vars.md (rules-engine context, NOT
                             * slab `sidx`). Display password CLAMPED TO
                             * 8 BYTES per CPU parity (mirrors mdxfind.c
                             * :23676-23677 `i = min(len, 8)` for non-
                             * extended salts; Q1 user decision 2026-05-
                             * 08). The post-rule plaintext lives in
                             * synthetic_job.line (already populated above
                             * by the synthetic-`:` arm or the applyrule
                             * replay arm + optional mask prepend/append);
                             * we re-clamp the LENGTH for display while
                             * preserving the underlying buffer (the
                             * kernel only saw the first 8 bytes anyway
                             * via the host-side rules-engine pack-site
                             * clamp at mdxfind.c:11780 + the kernel-side
                             * `if (plen > 8) plen = 8;` defensive cap in
                             * template_finalize). Phase 5 of Unix-crypt
                             * ladder on Metal -- LAST Unix-crypt op. */
                            hit = 0;
                            if (salt_snap_entry) {
                                char desbuf[64];
                                metal_des_reconstruct(curin.i[0], curin.i[1],
                                                      salt_bytes, desbuf);
                                Word_t *HPV;
                                JSLG(HPV, JudyJ[JOB_DESCRYPT],
                                     (unsigned char *)desbuf);
                                if (HPV && __sync_bool_compare_and_swap(HPV, 0, 1)) {
                                    if (MTL_PV_DEC(salt_snap_entry->PV)) _salts_retired_since_refresh++;
                                    /* Q1 (2026-05-08): clamp display
                                     * password to 8 bytes for CPU parity.
                                     * out_len is the post-rule + post-
                                     * mask length already set above; we
                                     * truncate via synthetic_job.clen so
                                     * prfound + downstream printers honor
                                     * the 8-byte cap. The line buffer
                                     * itself stays intact (no mutation
                                     * of bytes 9+). */
                                    int cplen = (out_len > 8) ? 8 : out_len;
                                    synthetic_job.line[cplen] = 0;
                                    synthetic_job.clen = cplen;
                                    prfound(&synthetic_job, desbuf);
                                }
                            }
                        } else {
                            hit = checkhashsalt(&curin, hexlen,
                                                salt_bytes, salt_len_b,
                                                iter_num,
                                                &synthetic_job);
                        }
                        if (hit && salt_snap_entry) {
                            if (MTL_PV_DEC(salt_snap_entry->PV)) _salts_retired_since_refresh++;
                        }
                    } else {
                        /* Phase 2a/2b: unsalted MD5 -> checkhash. */
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

            /* Hash accounting: packed_count * gpu_rule_count * mask_size
             * * iter_sum * Maxiter. Phase 2c adds the nsalts fan-out.
             * 2026-05-19: replace bare _nsalt_acct with _iter_sum_acct
             * (gpu_compute_iter_sum) so iterated GPU types (BCRYPT/
             * PHPBB3/DESCRYPT/MD5CRYPT/SHACRYPT/SHA1DRU) multiply by
             * actual rounds rather than salt count alone. For non-iterated
             * types the function returns nsalts_packed*1, bit-identical
             * to the prior accounting. */
            uint64_t _mask_size_acct =
                (gpu_mask_total > 1) ? gpu_mask_total : 1ull;
            uint64_t _iter_sum_acct =
                (nsalts_packed > 0)
                ? gpu_compute_iter_sum(g->op, saltsnap, pack_map,
                                       nsalts_packed)
                : 1ULL;
            uint64_t _per_dispatch_hashes =
                (uint64_t)g->packed_count *
                (uint64_t)(gpu_rule_count > 0 ? gpu_rule_count : 1) *
                _mask_size_acct *
                _iter_sum_acct *
                (uint64_t)Maxiter;
            hashcnt += _per_dispatch_hashes;
            _gpu_rules_hashes += _per_dispatch_hashes;

            if (hashcnt > 10000000 || found > 0) {
                atomic_fetch_add(&Tothash, hashcnt);
                atomic_fetch_add(&Totfound, found);
                hashcnt = 0;
                found = 0;
            }
            goto return_jobg;
        }

        /* Defensive: any packed slot without rules_engine is a caller
         * bug (mirrors OpenCL twin B7.9 invariant). Phase 2a non-packed
         * slots fall through to return_jobg (no slab path). */
        if (g->packed && g->packed_count > 0 && !g->rules_engine) {
            fprintf(stderr,
                "BUG: gpujob_metal_worker received packed slot with "
                "rules_engine=0 (op=%d packed_count=%u) — packed path "
                "requires rules_engine=1 in Phase 2a. Returning slot.\n",
                g->op, g->packed_count);
        }

return_jobg:
        g->next = NULL;
        g->count = 0;
        g->passbuf_pos = 0;
        g->word_stride = 0;
        g->packed = 0;
        g->packed_count = 0;
        g->packed_pos = 0;
        g->rules_engine = 0;
        g->bf_chunk = 0;
        g->bf_mask_start = 0;
        g->bf_offset_per_word = 0;
        g->bf_num_masks = 0;
        g->bf_inner_iter = 0;
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

    if (hashcnt || found) {
        atomic_fetch_add(&Tothash, hashcnt);
        atomic_fetch_add(&Totfound, found);
    }

    free(outbuf);
}

/* ---- Init / shutdown ---- */

int gpujob_init(int num_jobg) {
    if (!gpu_metal_available()) return -1;

    /* Phase 2c: compute _max_salt_count / _max_salt_bytes from
     * Typesaltcnt[] / Typesaltbytes[] (populated by mdxfind.c load
     * loop). Mirrors gpujob_opencl.c lines 2534-2543 exactly. The
     * worker thread allocates per-thread salt-snapshot scratch sized
     * to these constants. */
    _max_salt_count = 0;
    _max_salt_bytes = 0;
    for (int sti = 0; sti < 2000; sti++) {
        if (Typesaltcnt && Typesaltcnt[sti] > _max_salt_count)
            _max_salt_count = Typesaltcnt[sti];
        if (Typesaltbytes && Typesaltbytes[sti] > _max_salt_bytes)
            _max_salt_bytes = Typesaltbytes[sti];
    }
    if (_max_salt_count < 1024) _max_salt_count = 1024;
    if (_max_salt_bytes < 8192) _max_salt_bytes = 8192;

    gpu_metal_set_max_iter(0, Maxiter);

    _gpu_batch_max = GPU_RULES_MAX_WORDS_PER_BATCH;

    int n_legacy = num_jobg;
    int n_rules  = (gpu_rule_count > 0) ? num_jobg : 0;
    _num_legacy_jobg = n_legacy;
    _num_rules_jobg  = n_rules;

    GPUWorkWaiting       = new_lock(0);
    GPULegacyFreeWaiting = new_lock(n_legacy);
    GPURulesFreeWaiting  = new_lock(n_rules);
    GPUWorkTail          = &GPUWorkHead;

    for (int i = 0; i < n_legacy; i++) {
        struct jobg *g = (struct jobg *)malloc_lock(sizeof(struct jobg), "jobg");
        g->packed_buf = NULL;
        g->word_offset = NULL;
        g->packed = 0;
        g->packed_count = 0;
        g->packed_pos = 0;
        g->rules_engine = 0;
        g->bf_chunk = 0;
        g->bf_mask_start = 0;
        g->bf_offset_per_word = 0;
        g->bf_num_masks = 0;
        g->bf_inner_iter = 0;
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

    for (int i = 0; i < n_rules; i++) {
        struct jobg *g = (struct jobg *)malloc_lock(sizeof(struct jobg), "jobg");
        g->packed_buf = NULL;
        g->word_offset = NULL;
        g->packed = 0;
        g->packed_count = 0;
        g->packed_pos = 0;
        g->rules_engine = 0;
        g->bf_chunk = 0;
        g->bf_mask_start = 0;
        g->bf_offset_per_word = 0;
        g->bf_num_masks = 0;
        g->bf_inner_iter = 0;
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

    /* Spawn the single worker. */
    _worker = launch(gpujob_metal_worker, NULL);
    _gpujob_ready = 1;

    if (n_rules > 0)
        GPU_DEBUG_FPRINTF(stderr,
                "Metal GPU: 1 gpujob thread started (%d legacy + %d rules batch buffers)\n",
                n_legacy, n_rules);
    else
        GPU_DEBUG_FPRINTF(stderr,
                "Metal GPU: 1 gpujob thread started (%d batch buffers)\n",
                n_legacy);
    return 0;
}

void gpujob_shutdown(void) {
    if (!_gpujob_ready) return;

    /* Wait for queues to drain. */
    possess(GPULegacyFreeWaiting);
    wait_for(GPULegacyFreeWaiting, TO_BE, _num_legacy_jobg);
    release(GPULegacyFreeWaiting);
    if (_num_rules_jobg > 0) {
        possess(GPURulesFreeWaiting);
        wait_for(GPURulesFreeWaiting, TO_BE, _num_rules_jobg);
        release(GPURulesFreeWaiting);
    }

    /* Send one sentinel (Phase 2a: 1 worker). */
    struct jobg *sentinel = gpujob_get_free(NULL, 0);
    sentinel->op = 2000;
    sentinel->count = 0;
    gpujob_submit(sentinel);

    /* Join the worker so its hashcnt/found flush completes before
     * we tear down. */
    if (_worker) {
        join(_worker);
        _worker = NULL;
    }

    _gpujob_ready = 0;

    /* End-of-run summary (single device). */
    uint64_t units = _gpu_words;
    if (_gpu_batches > 0 || units > 0) {
        uint64_t wall_us = (_gpu_last_us && _gpu_first_us &&
                            _gpu_last_us >= _gpu_first_us)
                          ? (_gpu_last_us - _gpu_first_us) : 0;
        uint64_t busy_us = _gpu_busy_us;
        uint64_t idle_us = (wall_us > busy_us) ? (wall_us - busy_us) : 0;
        double idle_pct = wall_us > 0 ? (100.0 * (double)idle_us / (double)wall_us) : 0.0;
        uint64_t hashes = _gpu_rules_hashes;
        double word_mhps = busy_us > 0 ? ((double)units / (double)busy_us) : 0.0;
        double hash_ghps = busy_us > 0 ? ((double)hashes / (double)busy_us / 1e3) : 0.0;
        fprintf(stderr,
                "Metal GPU[0]: %llu batches | %llu words | %llu hashes | %llu hits\n"
                "              wall=%.2fs busy=%.2fs idle=%.2fs (%.0f%%) | hash_Gh/s=%.3f  unit_Mh/s=%.2f\n",
                (unsigned long long)_gpu_batches,
                (unsigned long long)_gpu_words,
                (unsigned long long)hashes,
                (unsigned long long)_gpu_hits,
                wall_us / 1e6, busy_us / 1e6, idle_us / 1e6, idle_pct,
                hash_ghps, word_mhps);
    }
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

void gpujob_print_share_line(FILE *fp) {
    if (!fp) return;
    /* Single-device: no share to print (>1 active devices is the
     * minimum threshold; matches OpenCL twin behavior at line 2849). */
}

int gpu_op_category(int op) {
    /* Phase 2d.2.1a: op_category is now sourced from the registered
     * family in gpu_metal.m. Each family advertises its op_category at
     * registration time; the load-bearing chokepoint gate at
     * mdxfind.c (per feedback_architect_host_wiring_reflex.md) sees the
     * correct category for every registered family without further
     * editing here.
     *
     * Phase 2c precedent: JOB_MD5 -> GPU_CAT_UNSALTED, JOB_MD5SALT ->
     * GPU_CAT_MASK (mirrors gpu/gpujob_opencl.c:3411 post-B6.6
     * placement; MD5SALT is template-routed, not slab-routed). Future
     * families register their op_category directly; no edit needed
     * here when adding md4, md5raw, etc.
     *
     * Unregistered ops fall through to GPU_CAT_NONE so the chokepoint
     * admit gate rejects them and they route through the CPU path. */
    struct gpu_metal_family *fam = gpu_metal_lookup_family(op);
    if (fam == NULL) return GPU_CAT_NONE;
    return fam->op_category;
}

#endif /* METAL_GPU */
