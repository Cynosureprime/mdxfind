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
extern lock *FreeWaiting;
extern unsigned long long Tothash, Totfound;
extern int checkhash(union HashU *curin, int len, int x, struct job *job);
extern int checkhashkey(union HashU *curin, int len, char *key, struct job *job);
extern int checkhashsalt(union HashU *curin, int len, char *salt, int saltlen, int x, struct job *job);
extern int build_salt_snapshot(void *snap, char *pool,
                void *judy, char *keybuf, int printall);
extern int *Typesaltcnt;
extern int *Typesaltbytes;

/* Return the correct salt/user Judy for a given op.
 * HMAC key=$salt types store their "salt" (the HMAC key) in Typeuser. */
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
    case JOB_HMAC_STREEBOG256_KSALT:
    case JOB_HMAC_STREEBOG512_KSALT:
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
struct jobg *GPUFreeHead, **GPUFreeTail;
lock *GPUWorkWaiting, *GPUFreeWaiting;
static int _gpujob_ready = 0;
static int _gpujob_count = 1;
static int _num_jobg_buffers = 0;
static int _max_salt_count = 0;
static int _max_salt_bytes = 0;
static int overflow_loaded = 0;
static int _gpu_batch_max = GPUBATCH_MAX; /* min across all devices */

/* ---- GPU scheduling — priority buffer allocation ---- */
struct gpu_waiter {
    struct gpu_waiter *next;
    char *filename;
    unsigned int startline;
    lock *wake;
};

static lock *GPUSchedLock;
static char *gpu_sched_filename = NULL;
static unsigned int gpu_sched_curline = 0;
static int gpu_sched_active = 0;
static struct gpu_waiter *gpu_waiter_head = NULL;
static struct gpu_waiter *gpu_waiter_pool = NULL; /* pre-allocated pool */
static int _gpu_waiter_count = 0;
static int gpu_sched_active_count = 0; /* threads granted but not yet submitted */

static void gpu_sched_wake_best(void) {
    /* Called with GPUSchedLock possessed */
    if (!gpu_waiter_head) return;

    struct gpu_waiter *best = NULL;
    struct gpu_waiter **best_pp = NULL;

    /* Priority 1: same filename, lower-than-current line number */
    if (gpu_sched_active) {
        struct gpu_waiter **pp = &gpu_waiter_head;
        while (*pp) {
            struct gpu_waiter *w = *pp;
            if (w->filename == gpu_sched_filename &&
                w->startline <= gpu_sched_curline) {
                if (!best || w->startline < best->startline) {
                    best = w;
                    best_pp = pp;
                }
            }
            pp = &w->next;
        }
    }

    /* Priority 2: no eligible same-file waiter found.
     * If no active threads remain, wake ALL waiters to prevent deadlock.
     * Each woken thread will pack its pending word and submit, keeping
     * the system alive. Without this, small wordlists hang. */
    if (!best && gpu_waiter_head && gpu_sched_active_count == 0) {
        while (gpu_waiter_head) {
            struct gpu_waiter *w = gpu_waiter_head;
            gpu_waiter_head = w->next;
            w->next = NULL;
            gpu_sched_active_count++;
            possess(w->wake);
            twist(w->wake, TO, 1);
        }
        return;
    }

    if (best) {
        *best_pp = best->next;
        best->next = NULL;
        gpu_sched_filename = best->filename;
        gpu_sched_curline = best->startline;
        gpu_sched_active = 1;
        gpu_sched_active_count++;
        possess(best->wake);
        twist(best->wake, TO, 1);
    }
}


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
#define GPU_MAX_RETURN 32768
#define OUTBUFSIZE (1024 * 1024)

static int gpu_pack_salts(struct saltentry *saltsnap, int nsalts,
                          char *salts_packed, uint32_t *soff, uint16_t *slen,
                          int *pack_map, int use_hashsalt) {
    int packed = 0;
    uint32_t gsp = 0;
    for (int i = 0; i < nsalts; i++) {
        if (!Printall && *saltsnap[i].PV == 0) continue;
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

static void load_overflow(int dev_idx) {
    if (!OverflowHash) return;
    int ocnt = 0;
    size_t obytes = 0;
    Word_t okey = 0;
    Word_t *OPV;
    OPV = (Word_t *)JudyLFirst(OverflowHash, &okey, NULL);
    while (OPV) {
        struct Hashchain *chain = (struct Hashchain *)(*OPV);
        while (chain) { ocnt++; obytes += chain->len; chain = chain->next; }
        OPV = (Word_t *)JudyLNext(OverflowHash, &okey, NULL);
    }
    if (ocnt > 0) {
        uint64_t *okeys = (uint64_t *)malloc_lock(ocnt * sizeof(uint64_t),"load_overflow");
        unsigned char *ohashes = (unsigned char *)malloc_lock(obytes + ocnt * 8,"load_overflow");
        uint32_t *ooffsets = (uint32_t *)malloc_lock(ocnt * sizeof(uint32_t),"load_overflow");
        uint16_t *olengths = (uint16_t *)malloc_lock(ocnt * sizeof(uint16_t),"load_overflow");
        int oi = 0;
        uint32_t opos = 0;
        okey = 0;
        OPV = (Word_t *)JudyLFirst(OverflowHash, &okey, NULL);
        while (OPV) {
            struct Hashchain *chain = (struct Hashchain *)(*OPV);
            while (chain) {
                okeys[oi] = okey;
                ooffsets[oi] = opos;
                olengths[oi] = chain->len;
                memcpy(ohashes + opos, &okey, 8);
                if (chain->len > 8)
                    memcpy(ohashes + opos + 8, chain->hash, chain->len - 8);
                opos += chain->len;
                oi++;
                chain = chain->next;
            }
            OPV = (Word_t *)JudyLNext(OverflowHash, &okey, NULL);
        }
        gpu_opencl_set_overflow(dev_idx, okeys, ohashes, ooffsets, olengths, ocnt);
        free(okeys); free(ohashes); free(ooffsets); free(olengths);
    }
}

void gpujob(void *arg) {
    int my_slot = (int)(intptr_t)arg;
    union HashU curin;
    struct job synthetic_job;
    char *outbuf = (char *)malloc_lock(OUTBUFSIZE,"gpujob");
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

        if (g->op == 2000) {
            g->next = NULL;
            possess(GPUFreeWaiting);
            if (GPUFreeTail) {
                *GPUFreeTail = g;
                GPUFreeTail = &(g->next);
            } else {
                GPUFreeHead = g;
                GPUFreeTail = &(g->next);
            }
            twist(GPUFreeWaiting, BY, +1);
            break;
        }

        /* Each device loads overflow to its own GPU memory */
        if (!my_overflow_loaded && OverflowHash) {
            my_overflow_loaded = 1;
            load_overflow(my_slot);
        }

        int op_cat = gpu_op_category(g->op);

        /* Rebuild salt snapshot on op change, periodically, or after hits found */
        if (op_cat == GPU_CAT_SALTED || op_cat == GPU_CAT_SALTPASS) {
            batch_count++;
            if (g->op != current_op ||
                (salt_refresh && op_cat == GPU_CAT_SALTPASS) ||
                (batch_count >= 10 && nsalts_packed > 0)) {
                salt_refresh = 0;
                if (g->op != current_op) {
                    current_op = g->op;
                    gpu_opencl_set_op(g->op);
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
                    nsalts_packed = gpu_pack_salts(saltsnap, nsalts,
                                                   salts_packed, soff, slen, pack_map, use_hs);
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
            gpu_opencl_set_op(g->op);
        }

        int nhits = 0;
        uint32_t *hits = NULL;

        if (g->count == 0) goto return_jobg;
        if ((op_cat == GPU_CAT_SALTED || op_cat == GPU_CAT_SALTPASS) && nsalts_packed == 0) {
            /* Stale nsalts_packed — force a fresh rebuild before giving up.
             * Words in this batch were packed when salts were still active. */
            int nsalts = build_salt_snapshot(saltsnap, saltpool,
                            gpu_salt_judy(g->op), tsalt, Printall);
            if (nsalts > 0) {
                nsalts_packed = gpu_pack_salts(saltsnap, nsalts,
                                               salts_packed, soff, slen, pack_map, 0);
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
            gpu_opencl_set_op(g->op);
        }

        synthetic_job.op = g->op;
        synthetic_job.flags = g->flags;
        synthetic_job.filename = g->filename;
        synthetic_job.doneprint = g->doneprint;
        synthetic_job.found = (unsigned int *)&found;
        synthetic_job.outlen = 0;

        /* Mask overflow retry loop: if GPU hit buffer overflows, process stored hits
         * then re-dispatch starting from the highest mask_idx seen. */
        int dispatch_retry;
        uint32_t max_mask_idx = 0;
        do {
        dispatch_retry = 0;
        max_mask_idx = 0;
        hits = gpu_opencl_dispatch_batch(my_slot,
            g->word_stride ? g->raw : g->hexhash[0],
            (const uint16_t *)g->hexlen,
            g->count, &nhits);

        if (hits && nhits > 0) {
            salt_refresh = 1;
            salt_hits_pending += nhits;
            int stored = nhits > GPU_MAX_RETURN ? GPU_MAX_RETURN : nhits;
            int is_md5crypt = (g->op == JOB_MD5CRYPT || g->op == JOB_PHPBB3);
            int is_bcrypt = (g->op == JOB_BCRYPT);
            int hash_words = is_bcrypt ? 6 : gpu_hash_words(g->op);
            int hexlen = hash_words * 8;
            /* hit_stride = 2 (word_idx, mask_idx/salt_idx) + hash_words
             * + 1 for iter when iteration/saltpass modes are active.
             * md5crypt/phpbb3: fixed stride 6, bcrypt: fixed stride 9. */
            int has_iter = (Maxiter > 1 || op_cat == GPU_CAT_ITER || op_cat == GPU_CAT_SALTPASS);
            int hit_stride = is_md5crypt ? 6
                : is_bcrypt ? 9
                : has_iter ? (3 + hash_words) : (2 + hash_words);
            int overflow = (nhits > GPU_MAX_RETURN);

            /* Track highest mask/salt index among stored hits for overflow retry.
             * On overflow, resume from this point on the next dispatch.
             * Up to num_words duplicates at the boundary are filtered by checkhash. */
            uint32_t max_salt_idx = 0;

            for (int h = 0; h < stored; h++) {
                uint32_t *entry = hits + h * hit_stride;
                int widx = entry[0];
                int sidx = entry[1];
                if ((unsigned)widx >= (unsigned)g->count) continue;
                if (op_cat != GPU_CAT_MASK && op_cat != GPU_CAT_UNSALTED && sidx < 0) continue;


                int iter_num;
                if (is_md5crypt) {
                    /* md5crypt/phpbb3: fixed stride 6, no iter field */
                    iter_num = 1;
                    for (int w = 0; w < 4; w++)
                        curin.i[w] = entry[2 + w];
                } else if (has_iter) {
                    iter_num = entry[2];
                    for (int w = 0; w < hash_words; w++)
                        curin.i[w] = entry[3 + w];
                } else {
                    iter_num = 1;
                    for (int w = 0; w < hash_words; w++)
                        curin.i[w] = entry[2 + w];
                }

                if (op_cat == GPU_CAT_MASK || op_cat == GPU_CAT_UNSALTED) {
                    /* Mask/unsalted: reconstruct candidate from base word + mask index.
                     * Recover full 64-bit mask_idx from truncated uint32 + mask_start. */
                    synthetic_job.Ruleindex = (widx < GPUBATCH_MAX) ? g->ruleindex[widx] : 0;
                    uint32_t midx32 = entry[1];
                    if (midx32 > max_mask_idx) max_mask_idx = midx32;
                    if ((uint32_t)sidx > max_salt_idx) max_salt_idx = (uint32_t)sidx;
                    extern uint64_t gpu_opencl_last_mask_start(void);
                    uint64_t chunk_ms = gpu_opencl_last_mask_start();
                    uint64_t midx = chunk_ms + (uint64_t)(midx32 - (uint32_t)chunk_ms);
                    char *base_word;
                    int blen;
                    if (g->word_stride && widx < 4096 &&
                        (gpu_mask_n_prepend > 0 || gpu_mask_n_append > 0)) {
                        char *slot = g->raw + widx * g->word_stride;
                        int total_len = ((uint32_t *)slot)[14] >> 3;
                        blen = total_len - gpu_mask_n_prepend - gpu_mask_n_append;
                        base_word = slot + gpu_mask_n_prepend;
                    } else {
                        base_word = &g->passbuf[g->passoff[widx]];
                        blen = g->passlen[widx];
                    }
                    char cand[256];
                    int clen = 0;
                    if (gpu_mask_n_prepend > 0) {
                        uint64_t append_combos = 1;
                        for (int mi = 0; mi < gpu_mask_n_append; mi++)
                            append_combos *= gpu_mask_sizes[gpu_mask_n_prepend + mi];
                        uint64_t prepend_idx = midx / append_combos;
                        uint64_t append_idx = midx % append_combos;
                        clen += mask_decode(prepend_idx, 0, gpu_mask_n_prepend, cand);
                        memcpy(cand + clen, base_word, blen);
                        clen += blen;
                        if (gpu_mask_n_append > 0)
                            clen += mask_decode(append_idx, gpu_mask_n_prepend,
                                                gpu_mask_n_append, cand + clen);
                    } else {
                        memcpy(cand, base_word, blen);
                        clen = blen;
                        clen += mask_decode(midx, 0, gpu_mask_n_append, cand + clen);
                    }
                    cand[clen] = 0;
                    memcpy(synthetic_job.line, cand, clen + 1);
                    synthetic_job.clen = clen;
                    synthetic_job.pass = synthetic_job.line;
                    checkhash(&curin, hexlen, iter_num, &synthetic_job);
                } else {
                    /* Non-mask path: set up synthetic_job from per-word metadata */
                    if ((uint32_t)sidx > max_salt_idx) max_salt_idx = (uint32_t)sidx;
                    char *pass = &g->passbuf[g->passoff[widx]];
                    int plen = g->passlen[widx];
                    memcpy(synthetic_job.line, pass, plen);
                    synthetic_job.line[plen] = 0;
                    synthetic_job.clen = g->clen[widx];
                    synthetic_job.pass = synthetic_job.line;
                    synthetic_job.Ruleindex = g->ruleindex[widx];
                if (op_cat == GPU_CAT_ITER) {
                    checkhash(&curin, hexlen, iter_num, &synthetic_job);
                } else if (g->op == JOB_PHPBB3) {
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    if (checkhashbb(&curin, 32, saltsnap[snap_idx].salt, &synthetic_job))
                        PV_DEC(saltsnap[snap_idx].PV);
                } else if (g->op == JOB_MD5CRYPT) {
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    int match_len; unsigned short *match_flags;
                    int hf = hybrid_check(curin.h, 16, &match_len, &match_flags);
                    if (hf && *match_flags != (unsigned short)g->op) {
                        *match_flags = g->op;
                        PV_DEC(saltsnap[snap_idx].PV);
                        char *sp = saltsnap[snap_idx].salt;
                        int splen = saltsnap[snap_idx].saltlen;
                        char mdbuf[128];
                        memcpy(mdbuf, sp, splen);
                        md5crypt_b64encode(curin.h, mdbuf + splen);
                        prfound(&synthetic_job, mdbuf);
                    }
                } else if (g->op == JOB_DESCRYPT) {
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    char *s1 = saltsnap[snap_idx].salt;
                    des_reconstruct(curin.i[0], curin.i[1], s1, desbuf);
                    Word_t *HPV;
                    JSLG(HPV, JudyJ[JOB_DESCRYPT], (unsigned char *)desbuf);
                    if (HPV && __sync_bool_compare_and_swap(HPV, 0, 1)) {
                        PV_DEC(saltsnap[snap_idx].PV);
                        char *pass = &g->passbuf[g->passoff[widx]];
                        int plen = g->passlen[widx];
                        int cplen = (plen > 8) ? 8 : plen;
                        memcpy(synthetic_job.line, pass, cplen);
                        synthetic_job.line[cplen] = 0;
                        synthetic_job.pass = synthetic_job.line;
                        synthetic_job.clen = cplen;
                        prfound(&synthetic_job, desbuf);
                    }
                } else if (g->op == JOB_BCRYPT) {
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    /* Reconstruct bcrypt hash string from GPU output.
                     * GPU hit gives 6 LE uint32 words = 24 bytes (23 meaningful).
                     * BF_encode expects BE byte stream, so swap each word back. */
                    unsigned char *raw = (unsigned char *)&curin.i[0];
                    /* BF_encode: 23 bytes -> 31 base64 chars */
                    static const char bf_itoa64[] =
                        "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
                    char hashb64[32];
                    { const unsigned char *sp = raw;
                      char *dp = hashb64;
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
                    /* Build full hash: salt_setting + hash_base64 */
                    char *sp2 = saltsnap[snap_idx].salt;
                    int splen = saltsnap[snap_idx].saltlen;
                    char fullhash[128];
                    memcpy(fullhash, sp2, splen);
                    memcpy(fullhash + splen, hashb64, 32);
                    Word_t *HPV;
                    JSLG(HPV, JudyJ[JOB_BCRYPT], (unsigned char *)fullhash);
                    if (HPV && __sync_bool_compare_and_swap(HPV, 0, 1)) {
                        PV_DEC(saltsnap[snap_idx].PV);
                        prfound(&synthetic_job, fullhash);
                    }
                } else {
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    char *s1 = saltsnap[snap_idx].salt;
                    int saltlen = saltsnap[snap_idx].saltlen;

                    if (iter_num <= 1 && (op_cat == GPU_CAT_SALTED ||
                        g->op == JOB_HMAC_MD5 || g->op == JOB_HMAC_SHA1 ||
                        g->op == JOB_HMAC_SHA224 || g->op == JOB_HMAC_SHA256 ||
                        g->op == JOB_HMAC_SHA384 || g->op == JOB_HMAC_SHA512 ||
                        g->op == JOB_HMAC_RMD160 || g->op == JOB_HMAC_RMD320 ||
                        g->op == JOB_HMAC_STREEBOG256_KSALT || g->op == JOB_HMAC_STREEBOG512_KSALT)) {
                        if (checkhashkey(&curin, hexlen, s1, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    } else {
                        if (checkhashsalt(&curin, hexlen, s1, saltlen, iter_num, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    }
                }
                } /* end non-mask else */
            }

            /* Flush thread-local found count to Totfound */
            if (*synthetic_job.found) {
                possess(FreeWaiting);
                Totfound += *synthetic_job.found;
                release(FreeWaiting);
                *synthetic_job.found = 0;
            }

            /* Hit buffer overflow: re-dispatch same batch to GPU starting from
             * the highest index seen.  Up to 4096 duplicates at the boundary
             * are acceptable — filtered by checksalthash / checkhash. */
            if (overflow) {
                dispatch_retry = 1;
                if (op_cat == GPU_CAT_MASK || op_cat == GPU_CAT_UNSALTED) {
                    gpu_opencl_set_mask_resume(max_mask_idx);
                } else if (op_cat == GPU_CAT_SALTED || op_cat == GPU_CAT_SALTPASS) {
                    gpu_opencl_set_salt_resume(max_salt_idx);
                }
            }
            /* Per-chunk hit return: if dispatch_batch returned early with
             * hits and more chunks remain, retry to continue processing. */
            { extern int gpu_opencl_has_resume(void);
              if (!dispatch_retry && gpu_opencl_has_resume())
                dispatch_retry = 1;
            }

            if (0) {  /* --- dead CPU fallback (replaced by GPU retry above) --- */
                int wi, si;
                for (wi = 0; wi < g->count; wi++) {
                    synthetic_job.clen = g->clen[wi];
                    synthetic_job.Ruleindex = g->ruleindex[wi];

                    for (si = 0; si < nsalts_packed; si++) {
                        int snap_idx = pack_map[si];
                        char *s1 = saltsnap[snap_idx].salt;
                        int saltlen = saltsnap[snap_idx].saltlen;
                        char tmpbuf[1024];
                        int tlen;

                        char *pass; int plen;
                        switch (g->op) {
                        case JOB_MD5SALT: case JOB_MD5UCSALT: case JOB_MD5revMD5SALT:
                        case JOB_MD5sub8_24SALT:
                            memcpy(tmpbuf, g->hexhash[wi], g->hexlen[wi]);
                            memcpy(tmpbuf + g->hexlen[wi], s1, saltlen);
                            mymd5(tmpbuf, g->hexlen[wi] + saltlen, curin.h);
                            memcpy(synthetic_job.line, g->hexhash[wi], g->hexlen[wi]);
                            synthetic_job.line[g->hexlen[wi]] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashkey(&curin, 32, s1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_MD5_MD5SALTMD5PASS:
                            /* Recompute: MD5(hex(MD5(salt)) + hex(MD5(pass))) */
                            { char *hs = saltsnap[snap_idx].hashsalt;
                              if (!hs) break;
                              memcpy(tmpbuf, hs, 32);
                              memcpy(tmpbuf + 32, g->hexhash[wi], g->hexlen[wi]);
                              mymd5(tmpbuf, 32 + g->hexlen[wi], curin.h);
                            }
                            memcpy(synthetic_job.line, g->hexhash[wi], g->hexlen[wi]);
                            synthetic_job.line[g->hexlen[wi]] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 32, saltsnap[snap_idx].salt,
                                    saltsnap[snap_idx].saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_MD5SALTPASS:
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            memcpy(tmpbuf, s1, saltlen);
                            memcpy(tmpbuf + saltlen, pass, plen);
                            mymd5(tmpbuf, saltlen + plen, curin.h);
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 32, s1, saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_MD5PASSSALT:
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            memcpy(tmpbuf, pass, plen);
                            memcpy(tmpbuf + plen, s1, saltlen);
                            mymd5(tmpbuf, plen + saltlen, curin.h);
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 32, s1, saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_SHA256SALTPASS:
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            memcpy(tmpbuf, s1, saltlen);
                            memcpy(tmpbuf + saltlen, pass, plen);
                            mysha256(tmpbuf, saltlen + plen, curin.h);
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 64, s1, saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_SHA256PASSSALT:
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            memcpy(tmpbuf, pass, plen);
                            memcpy(tmpbuf + plen, s1, saltlen);
                            mysha256(tmpbuf, plen + saltlen, curin.h);
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 64, s1, saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_SHA1SALTPASS:
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            memcpy(tmpbuf, s1, saltlen);
                            memcpy(tmpbuf + saltlen, pass, plen);
                            mysha1(tmpbuf, saltlen + plen, curin.h);
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 40, s1, saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_SHA1PASSSALT:
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            memcpy(tmpbuf, pass, plen);
                            memcpy(tmpbuf + plen, s1, saltlen);
                            mysha1(tmpbuf, plen + saltlen, curin.h);
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashsalt(&curin, 40, s1, saltlen, 1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        case JOB_MD5CRYPT: {  /* Full MD5CRYPT compute for overflow retry */
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            /* s1 = "$1$salt$", extract raw salt */
                            char *rawsalt = s1 + 3;
                            int rslen = saltlen - 4;
                            if (rslen < 0) rslen = 0;
                            if (rslen > 8) rslen = 8;
                            /* Step 1 */
                            int blen = 0;
                            memcpy(tmpbuf, pass, plen); blen = plen;
                            memcpy(tmpbuf+blen, rawsalt, rslen); blen += rslen;
                            memcpy(tmpbuf+blen, pass, plen); blen += plen;
                            mymd5(tmpbuf, blen, curin.h);
                            /* Step 2 */
                            blen = 0;
                            memcpy(tmpbuf, pass, plen); blen = plen;
                            memcpy(tmpbuf+blen, s1, rslen+3); blen += rslen+3;
                            { int xx;
                            for (xx = plen; xx > 0; xx -= 16) {
                                int n = (xx > 16) ? 16 : xx;
                                memcpy(tmpbuf+blen, curin.h, n); blen += n;
                            }
                            for (xx = plen; xx != 0; xx >>= 1)
                                tmpbuf[blen++] = (xx & 1) ? 0 : pass[0];
                            }
                            mymd5(tmpbuf, blen, curin.h);
                            /* Step 3: 1000 iterations */
                            { int xx;
                            for (xx = 0; xx < 1000; xx++) {
                                blen = 0;
                                if (xx & 1) { memcpy(tmpbuf, pass, plen); blen = plen; }
                                else { memcpy(tmpbuf, curin.h, 16); blen = 16; }
                                if (xx % 3) { memcpy(tmpbuf+blen, rawsalt, rslen); blen += rslen; }
                                if (xx % 7) { memcpy(tmpbuf+blen, pass, plen); blen += plen; }
                                if (xx & 1) { memcpy(tmpbuf+blen, curin.h, 16); blen += 16; }
                                else { memcpy(tmpbuf+blen, pass, plen); blen += plen; }
                                mymd5(tmpbuf, blen, curin.h);
                            }
                            }
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            { int ml; unsigned short *mf;
                              int hf = hybrid_check(curin.h, 16, &ml, &mf);
                              if (hf && *mf != (unsigned short)g->op) {
                                  *mf = g->op;
                                  PV_DEC(saltsnap[snap_idx].PV);
                                  char mdbuf2[128];
                                  memcpy(mdbuf2, s1, saltlen);
                                  md5crypt_b64encode(curin.h, mdbuf2 + saltlen);
                                  prfound(&synthetic_job, mdbuf2);
                              }
                            }
                            break;
                        }
                        case JOB_DESCRYPT: {
                            /* Overflow retry: recompute DES for this word+salt on CPU */
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            int cplen = (plen > 8) ? 8 : plen;
                            /* Use bsd_crypt_des for overflow (desblk allocated once) */
                            static void *_overflow_desblk;
                            if (!_overflow_desblk) _overflow_desblk = malloc_lock(102400, "desblk");
                            if (_overflow_desblk) {
                                char passz[16], dbuf[64];
                                memcpy(passz, pass, cplen);
                                passz[cplen] = 0;
                                extern char *bsd_crypt_des(char *, char *, char *, void *);
                                char *crypted = bsd_crypt_des(passz, s1, dbuf, _overflow_desblk);
                                if (crypted) {
                                    Word_t *HPV;
                                    JSLG(HPV, JudyJ[JOB_DESCRYPT], (unsigned char *)dbuf);
                                    if (HPV && *HPV == 0) {
                                        *HPV = 1;
                                        PV_DEC(saltsnap[snap_idx].PV);
                                        memcpy(synthetic_job.line, pass, cplen);
                                        synthetic_job.line[cplen] = 0;
                                        synthetic_job.pass = synthetic_job.line;
                                        synthetic_job.clen = cplen;
                                        prfound(&synthetic_job, dbuf);
                                    }
                                }
                            }
                            break;
                        }
                        }
                    }
                }
            }
        }

        if (synthetic_job.outlen > 0) {
            fwrite(outbuf, synthetic_job.outlen, 1, stdout);
            fflush(stdout);
            synthetic_job.outlen = 0;
        }

        /* Overflow retry already set up above (mask_resume or salt_resume) */

        } while (dispatch_retry);  /* retry mask dispatch on overflow */

        /* Hash count: mask/unsalted types are tracked per-chunk inside
         * dispatch_batch via __sync_fetch_and_add(&Tothash). Only count
         * here for non-mask types. */
        if (op_cat == GPU_CAT_MASK || op_cat == GPU_CAT_UNSALTED) {
            /* already counted per-chunk in dispatch_batch */
        }
        else if (op_cat == GPU_CAT_ITER)
            hashcnt += (uint64_t)g->count * (Maxiter - 1);
        else if (g->op == JOB_PHPBB3) {
            uint64_t total_iter = 0;
            for (int si = 0; si < nsalts_packed; si++)
                total_iter += phpbb3_iter_count(salts_packed + soff[si], slen[si]);
            hashcnt += (uint64_t)g->count * total_iter;
        } else
            hashcnt += (uint64_t)g->count * nsalts_packed * Maxiter;

        if (hashcnt > 10000000 || found > 0) {
            possess(FreeWaiting);
            Tothash += hashcnt;
            Totfound += found;
            release(FreeWaiting);
            hashcnt = 0;
            found = 0;
        }

return_jobg:
        g->next = NULL;
        g->count = 0;
        g->passbuf_pos = 0;
        possess(GPUFreeWaiting);
        if (GPUFreeTail) {
            *GPUFreeTail = g;
            GPUFreeTail = &(g->next);
        } else {
            GPUFreeHead = g;
            GPUFreeTail = &(g->next);
        }
        twist(GPUFreeWaiting, BY, +1);
    }

    if (hashcnt || found) {
        possess(FreeWaiting);
        Tothash += hashcnt;
        Totfound += found;
        release(FreeWaiting);
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

    gpu_opencl_set_max_iter(Maxiter);

    /* One gpujob thread per GPU device; compute min batch limit */
    _gpujob_count = gpu_opencl_num_devices();
    if (_gpujob_count < 1) _gpujob_count = 1;

    _gpu_batch_max = GPUBATCH_MAX;
    for (int i = 0; i < _gpujob_count; i++) {
        int mb = gpu_opencl_max_batch(i);
        if (mb < _gpu_batch_max) _gpu_batch_max = mb;
    }

    _num_jobg_buffers = num_jobg;
    GPUWorkWaiting = new_lock(0);
    GPUFreeWaiting = new_lock(num_jobg);
    GPUWorkTail = &GPUWorkHead;

    /* Scheduler: pre-allocate maxt+1 waiter slots */
    GPUSchedLock = new_lock(0);
    release(GPUSchedLock);
    _gpu_waiter_count = num_jobg;  /* at least maxt+1 */
    for (int i = 0; i < _gpu_waiter_count; i++) {
        struct gpu_waiter *w = (struct gpu_waiter *)malloc_lock(sizeof(*w), "gpu_waiter");
        w->wake = new_lock(0);
        w->next = gpu_waiter_pool;
        gpu_waiter_pool = w;
    }

    for (int i = 0; i < num_jobg; i++) {
        struct jobg *g = (struct jobg *)malloc_lock(sizeof(struct jobg), "jobg");
        if (GPUFreeTail) {
            *GPUFreeTail = g;
            GPUFreeTail = &(g->next);
        } else {
            GPUFreeHead = g;
            GPUFreeTail = &(g->next);
        }
    }

    for (int i = 0; i < _gpujob_count; i++)
        launch(gpujob, (void *)(intptr_t)i);
    _gpujob_ready = 1;
    fprintf(stderr, "OpenCL GPU: %d gpujob thread%s started (%d batch buffers)\n",
            _gpujob_count, _gpujob_count > 1 ? "s" : "", num_jobg);
    return 0;
}

void gpujob_shutdown(void) {
    if (!_gpujob_ready) return;

    /* Wake all threads blocked in the priority scheduler so they can
     * finish their work and reach JOB_DONE. Without this, threads
     * waiting on the scheduler with small wordlists deadlock. */
    possess(GPUSchedLock);
    gpu_sched_active = 0;
    while (gpu_waiter_head) {
        struct gpu_waiter *w = gpu_waiter_head;
        gpu_waiter_head = w->next;
        w->next = NULL;
        gpu_sched_active_count++;
        possess(w->wake);
        twist(w->wake, TO, 1);
    }
    release(GPUSchedLock);

    /* Wait for GPU work queue to drain — procjob threads may still be
     * flushing partial JOBGs. Wait until all batch buffers are returned
     * to the free list (meaning no work is in flight). */
    possess(GPUFreeWaiting);
    wait_for(GPUFreeWaiting, TO_BE, _num_jobg_buffers);
    release(GPUFreeWaiting);

    for (int i = 0; i < _gpujob_count; i++) {
        struct jobg *sentinel = gpujob_get_free(NULL, 0);
        sentinel->op = 2000;
        sentinel->count = 0;
        sentinel->line_num = 0;
        gpujob_submit(sentinel);
    }
    _gpujob_ready = 0;
}

struct jobg *gpujob_get_free(char *filename, unsigned int startline) {
    /* NULL filename = shutdown sentinel, skip scheduling */
    if (!filename) goto get_buffer;

    possess(GPUSchedLock);

    if (!gpu_sched_active) {
        /* No current file — take over */
        gpu_sched_filename = filename;
        gpu_sched_curline = startline;
        gpu_sched_active = 1;
        gpu_sched_active_count++;
        release(GPUSchedLock);
    } else if (filename == gpu_sched_filename && startline <= gpu_sched_curline) {
        /* Same file, equal or lower line — grant immediately */
        if (startline < gpu_sched_curline)
            gpu_sched_curline = startline;
        gpu_sched_active_count++;
        release(GPUSchedLock);
    } else if (filename == gpu_sched_filename) {
        /* Same file, higher line — grant but don't lower curline.
         * Full blocking was causing deadlock with small wordlists. */
        gpu_sched_active_count++;
        release(GPUSchedLock);
    } else {
        /* Different file — wait for current file to finish */
        struct gpu_waiter *w = gpu_waiter_pool;
        if (w) gpu_waiter_pool = w->next;
        else w = (struct gpu_waiter *)malloc_lock(sizeof(*w), "gpu_waiter");
        if (!w->wake) w->wake = new_lock(0);
        w->filename = filename;
        w->startline = startline;
        w->next = gpu_waiter_head;
        gpu_waiter_head = w;
        lock *my_wake = w->wake;
        release(GPUSchedLock);

        /* Sleep until woken by scheduler */
        possess(my_wake);
        wait_for(my_wake, TO_BE, 1);
        twist(my_wake, TO, 0);

        /* Return waiter to pool */
        possess(GPUSchedLock);
        w->next = gpu_waiter_pool;
        gpu_waiter_pool = w;
        release(GPUSchedLock);
    }

get_buffer:
    if (MDXpause) {
        __sync_fetch_and_add(&MDXpaused_count, 1);
#ifdef _WIN32
        while (MDXpause) Sleep(2000);
#else
        while (MDXpause) sleep(2);
#endif
        __sync_fetch_and_sub(&MDXpaused_count, 1);
    }
    possess(GPUFreeWaiting);
    __sync_fetch_and_add(&MDXpaused_count, 1);
    wait_for(GPUFreeWaiting, NOT_TO_BE, 0);
    __sync_fetch_and_sub(&MDXpaused_count, 1);
    struct jobg *g = GPUFreeHead;
    GPUFreeHead = g->next;
    g->next = NULL;
    if (GPUFreeHead == NULL)
        GPUFreeTail = &GPUFreeHead;
    twist(GPUFreeWaiting, BY, -1);
    g->count = 0;
    g->passbuf_pos = 0;
    return g;
}

void gpujob_submit(struct jobg *g) {
    g->next = NULL;
    possess(GPUWorkWaiting);
    *GPUWorkTail = g;
    GPUWorkTail = &(g->next);
    twist(GPUWorkWaiting, BY, +1);

    /* Wake the best waiting procjob thread */
    possess(GPUSchedLock);
    gpu_sched_active_count--;
    gpu_sched_wake_best();
    release(GPUSchedLock);
}

/* Return a jobg buffer to the free list without submitting it for work. */
void gpujob_return_free(struct jobg *g) {
    g->next = NULL;
    g->count = 0;
    g->passbuf_pos = 0;
    possess(GPUFreeWaiting);
    if (GPUFreeTail) {
        *GPUFreeTail = g;
        GPUFreeTail = &(g->next);
    } else {
        GPUFreeHead = g;
        GPUFreeTail = &(g->next);
    }
    twist(GPUFreeWaiting, BY, +1);
}

/* Non-blocking version: returns NULL immediately if no free buffer. */
struct jobg *gpujob_try_get_free(void) {
    if (!_gpujob_ready) return NULL;
    possess(GPUFreeWaiting);
    if (peek_lock(GPUFreeWaiting) == 0) {
        release(GPUFreeWaiting);
        return NULL;
    }
    struct jobg *g = GPUFreeHead;
    GPUFreeHead = g->next;
    g->next = NULL;
    if (GPUFreeHead == NULL)
        GPUFreeTail = &GPUFreeHead;
    twist(GPUFreeWaiting, BY, -1);
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
    return (int)peek_lock(GPUFreeWaiting);
}

int gpu_op_category(int op) {
    switch (op) {
    /* Salted MD5 variants -- GPU iterates over salts */
    case JOB_MD5SALT:
    case JOB_MD5UCSALT:
    case JOB_MD5revMD5SALT:
    case JOB_MD5sub8_24SALT:
        return GPU_CAT_SALTED;
    /* Salt + raw password */
    case JOB_MD5SALTPASS:
    case JOB_MD5PASSSALT:
    case JOB_SHA256SALTPASS:
    case JOB_SHA256PASSSALT:
    case JOB_SHA1SALTPASS:
    case JOB_SHA1PASSSALT:
    case JOB_MD5CRYPT:
    case JOB_PHPBB3:
    case JOB_DESCRYPT:
    case JOB_HMAC_MD5: case JOB_HMAC_MD5_KPASS:
    case JOB_HMAC_SHA1: case JOB_HMAC_SHA1_KPASS:
    case JOB_HMAC_SHA224: case JOB_HMAC_SHA224_KPASS:
    case JOB_HMAC_SHA256: case JOB_HMAC_SHA256_KPASS:
    case JOB_SHA512PASSSALT: case JOB_SHA512SALTPASS:
    case JOB_HMAC_SHA384: case JOB_HMAC_SHA384_KPASS:
    case JOB_HMAC_SHA512: case JOB_HMAC_SHA512_KPASS:
    case JOB_HMAC_RMD160: case JOB_HMAC_RMD160_KPASS:
    case JOB_HMAC_RMD320: case JOB_HMAC_RMD320_KPASS:
    case JOB_HMAC_BLAKE2S:
    case JOB_BCRYPT:
    case JOB_SHA512CRYPT: case JOB_SHA256CRYPT: case JOB_SHA512CRYPTMD5:
        return GPU_CAT_SALTPASS;
    case JOB_MD5_MD5SALTMD5PASS:
        return GPU_CAT_SALTED;
    case JOB_SHA1DRU:
        return GPU_CAT_ITER;
    case JOB_MD5: case JOB_MD5UC:
    case JOB_MD4:
    case JOB_NTLMH:
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
    case JOB_SHA384RAW: case JOB_SHA512RAW:
    case JOB_MYSQL3:
    case JOB_STREEBOG_32: case JOB_STREEBOG_64:
    case JOB_RMD160:
    case JOB_BLAKE2S256:
        return GPU_CAT_MASK;
    case JOB_HMAC_STREEBOG256_KPASS: case JOB_HMAC_STREEBOG256_KSALT:
    case JOB_HMAC_STREEBOG512_KPASS: case JOB_HMAC_STREEBOG512_KSALT:
        return GPU_CAT_SALTPASS;
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
    case JOB_MD4: case JOB_NTLMH:
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
        return 16;
    /* 384-bit output = 12 words */
    case JOB_SHA384: case JOB_SHA384RAW:
    case JOB_HMAC_SHA384: case JOB_HMAC_SHA384_KPASS:
    case JOB_KECCAK384: case JOB_SHA3_384:
        return 12;
    /* 256-bit output = 8 words */
    case JOB_SHA256: case JOB_SHA256RAW:
    case JOB_SHA256PASSSALT: case JOB_SHA256SALTPASS:
    case JOB_HMAC_SHA224: case JOB_HMAC_SHA224_KPASS:
    case JOB_HMAC_SHA256: case JOB_HMAC_SHA256_KPASS:
    case JOB_HMAC_STREEBOG256_KSALT: case JOB_HMAC_STREEBOG256_KPASS:
    case JOB_HMAC_BLAKE2S:
    case JOB_BLAKE2S256:
    case JOB_KECCAK256: case JOB_SHA3_256:
    case JOB_STREEBOG_32:
    case JOB_MD6256:
    case JOB_SHA256CRYPT:
        return 8;
    /* 224-bit output = 7 words */
    case JOB_SHA224:
    case JOB_KECCAK224: case JOB_SHA3_224:
        return 7;
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
