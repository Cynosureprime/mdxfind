/*
 * gpujob.m — Dedicated GPU worker thread(s) for mdxfind Metal acceleration
 *
 * Receives struct jobg batches from procjob() threads via a yarn lock queue.
 * Dispatches batched pre-hashed words × all salts to GPU in a single Metal
 * command, amortizing the ~5ms dispatch overhead across hundreds of words.
 *
 * Salt snapshot buffers are allocated once at max size (procjob pattern,
 * see mdxfind.c lines 8136-8149), reused for every job.
 *
 * Define GPU_DOUBLE_BUFFER to enable experimental pipelined double-buffered
 * dispatch via per-slot GPU buffers (gpu_metal_submit_slot/wait_slot).
 * Requires: no @autoreleasepool in submit_slot, per-slot command queues.
 * Currently disabled: salt state lifecycle needs duplicate saltpools.
 */

#if defined(__APPLE__) && defined(METAL_GPU)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#include "mdxfind.h"
#include "job_types.h"
#include "gpujob.h"
#include "gpu_metal.h"

extern "C" void mymd5(char *, int, unsigned char *);
extern "C" void mysha256(char *, int, unsigned char *);
extern "C" int checkhashbb(union HashU *, int, char *, struct job *);
uint64_t gpu_mask_total = 0;
extern "C" int build_hashsalt_snapshot(struct saltentry *, char *, void *, char *, int);
extern void **Typehashsalt;

extern "C" {
#include "yarn.h"
#include <Judy.h>
extern int Printall, Maxiter;
extern char Typedone[];
extern void **Typesalt;
extern void *OverflowHash;
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
extern char phpitoa64[];
extern void prfound(struct job *, char *);
int gpu_mask_n_prepend = 0, gpu_mask_n_append = 0;
uint8_t gpu_mask_desc[64];

/* Mask charset helpers — must match kernel's charset_size/charset_char */
static const char *mask_charsets[] = {
    "0123456789", "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", NULL, NULL
};
static const int mask_csizes[] = { 10, 26, 26, 33, 95, 256 };

static int mask_decode(uint32_t mask_idx, const uint8_t *desc, int npos, char *buf) {
    uint32_t idx = mask_idx;
    for (int i = npos - 1; i >= 0; i--) {
        int csid = desc[i];
        int sz = mask_csizes[csid];
        int ci = idx % sz;
        idx /= sz;
        if (csid == 4) buf[i] = (char)(0x20 + ci);
        else if (csid == 5) buf[i] = (char)ci;
        else buf[i] = mask_charsets[csid][ci];
    }
    return npos;
}

/* Reconstruct 13-char DES crypt string from GPU pre-FP output (l, r) and salt */
static void des_reconstruct(uint32_t gl, uint32_t gr, const char *salt, char *out) {
    static const unsigned char DES_FP[64] = {
        40, 8,48,16,56,24,64,32,39, 7,47,15,55,23,63,31,
        38, 6,46,14,54,22,62,30,37, 5,45,13,53,21,61,29,
        36, 4,44,12,52,20,60,28,35, 3,43,11,51,19,59,27,
        34, 2,42,10,50,18,58,26,33, 1,41, 9,49,17,57,25
    };
    uint32_t il = gl, ir = gr;
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
} /* extern "C" */

#define PV_DEC(pv) { unsigned long _old = *(pv); \
  while (_old > 0) { \
    if (__sync_bool_compare_and_swap((pv), _old, _old - 1)) break; \
    _old = *(pv); } }

/* Must match mdxfind.c definition */
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
static int _num_jobg_buffers = 0;
static int _max_salt_count = 0;
static int _max_salt_bytes = 0;
static int overflow_loaded = 0;

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
static struct gpu_waiter *gpu_waiter_pool = NULL;
static int gpu_sched_active_count = 0;

static void gpu_sched_wake_best(void) {
    /* Called with GPUSchedLock possessed */
    if (!gpu_waiter_head) return;

    struct gpu_waiter *best = NULL;
    struct gpu_waiter **best_pp = NULL;

    /* Priority 1: same filename, lower-or-equal line number */
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

    /* Priority 2: no active threads — wake ALL waiters to prevent deadlock */
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

static int _gpujob_count = 1;  /* single thread, pipelined slots */

#define GPU_MAX_RETURN 32768
#define OUTBUFSIZE (1024 * 1024)

/* Pack salts from snapshot into flat arrays for GPU upload.
 * Writes into caller-provided buffers (pre-allocated at max size).
 * Returns number of packed salts. */
static int gpu_pack_salts(struct saltentry *saltsnap, int nsalts,
                          char *salts_packed, uint32_t *soff, uint16_t *slen,
                          int *pack_map, int use_hashsalt) {
    int packed = 0;
    uint32_t gsp = 0;
    for (int i = 0; i < nsalts; i++) {
        if (!Printall && *saltsnap[i].PV == 0) continue;
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

/* Decode phpass iteration count from salt[3] */
static int phpbb3_iter_count(const char *salt, int saltlen) {
    static const char itoa64[] = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    if (saltlen < 4) return 2048;  /* default $H$9 */
    char c = salt[3];
    for (int k = 0; k < 64; k++)
        if (itoa64[k] == c) return 1 << k;
    return 2048;
}

/* Sort packed salts by PHPBB3 iteration count (ascending).
 * Reorders soff[], slen[], pack_map[], and salt data in-place.
 * Returns number of distinct groups. Fills group_start[]/group_count[]/group_iter[]
 * (caller provides arrays sized to max 64 groups). */
static int phpbb3_group_salts(char *salts_packed, uint32_t *soff, uint16_t *slen,
                              int *pack_map, int nsalts,
                              int *group_start, int *group_count, int *group_iter) {
    if (nsalts <= 0) return 0;
    /* Extract iteration counts */
    int *icounts = (int *)malloc_lock(nsalts * sizeof(int), "icounts");
    for (int i = 0; i < nsalts; i++)
        icounts[i] = phpbb3_iter_count(salts_packed + soff[i], slen[i]);
    /* Simple insertion sort by iteration count (stable, small N per group) */
    for (int i = 1; i < nsalts; i++) {
        int ic = icounts[i];
        uint32_t so = soff[i]; uint16_t sl = slen[i]; int pm = pack_map[i];
        int j = i - 1;
        while (j >= 0 && icounts[j] > ic) {
            icounts[j+1] = icounts[j]; soff[j+1] = soff[j];
            slen[j+1] = slen[j]; pack_map[j+1] = pack_map[j];
            j--;
        }
        icounts[j+1] = ic; soff[j+1] = so; slen[j+1] = sl; pack_map[j+1] = pm;
    }
    /* Build groups */
    int ngroups = 0;
    int i = 0;
    while (i < nsalts) {
        group_start[ngroups] = i;
        group_iter[ngroups] = icounts[i];
        int j = i;
        while (j < nsalts && icounts[j] == icounts[i]) j++;
        group_count[ngroups] = j - i;
        ngroups++;
        i = j;
    }
    free(icounts);
    return ngroups;
}

/* Load overflow hash entries into GPU — called once */
static void load_overflow(void) {
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
        uint64_t *okeys = (uint64_t *)malloc_lock(ocnt * sizeof(uint64_t), "load_overflow");
        unsigned char *ohashes = (unsigned char *)malloc_lock(obytes + ocnt * 8, "load_overflow");
        uint32_t *ooffsets = (uint32_t *)malloc_lock(ocnt * sizeof(uint32_t), "load_overflow");
        uint16_t *olengths = (uint16_t *)malloc_lock(ocnt * sizeof(uint16_t), "load_overflow");
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
        gpu_metal_set_overflow(okeys, ohashes, ooffsets, olengths, ocnt);
        free(okeys); free(ohashes); free(ooffsets); free(olengths);
    }
}

/* ---- The GPU worker thread ---- */
void gpujob(void *arg) {
    int my_slot = (int)(intptr_t)arg;
    union HashU curin;
    struct job synthetic_job;
    char *outbuf = (char *)malloc_lock(OUTBUFSIZE, "gpujob");
    uint64_t hashcnt = 0, found = 0;
    char tsalt[4096];

    memset(&synthetic_job, 0, sizeof(synthetic_job));
    synthetic_job.outbuf = outbuf;

    /* Allocate salt snapshot buffers once at max size (procjob pattern) */
    struct saltentry *saltsnap = (struct saltentry *)malloc_lock(
        _max_salt_count * sizeof(struct saltentry), "saltsnap");
    char *saltpool = (char *)malloc_lock(_max_salt_bytes + 16, "saltpool");
    size_t sp_size = _max_salt_bytes + 4096;
    if ((size_t)_max_salt_count * 32 + 4096 > sp_size)
        sp_size = (size_t)_max_salt_count * 32 + 4096;
    char *salts_packed = (char *)malloc_lock(sp_size, "salts_packed");
    uint32_t *soff = (uint32_t *)malloc_lock(_max_salt_count * sizeof(uint32_t), "gpujob");
    uint16_t *slen = (uint16_t *)malloc_lock(_max_salt_count * sizeof(uint16_t), "gpujob");
    int *pack_map = (int *)malloc_lock(_max_salt_count * sizeof(int), "gpujob");
    int nsalts = 0;
    int nsalts_packed = 0;
    int current_op = -1;
    int batch_count = 0;
    int salt_refresh = 0;
#ifdef GPU_DOUBLE_BUFFER
    int pipeline_slot = 0;
    struct jobg *pipeline_prev_g = NULL;
    struct jobg *pipeline_cur_g = NULL;
    int pipeline_prev_nsalts = 0;
    int pipeline_prev_slot = -1;
    /* Saved salt state for previous dispatch — hits reference old packing */
    struct saltentry *prev_saltsnap = (struct saltentry *)malloc_lock(
        _max_salt_count * sizeof(struct saltentry), "prev_saltsnap");
    int *prev_pack_map = (int *)malloc_lock(_max_salt_count * sizeof(int), "prev_pack_map");
    int prev_nsalts_packed = 0;
#endif

    while (1) {
        /* Dequeue a JOBG */
        possess(GPUWorkWaiting);
        wait_for(GPUWorkWaiting, NOT_TO_BE, 0);
        struct jobg *g = GPUWorkHead;
        GPUWorkHead = g->next;
        if (GPUWorkHead == NULL)
            GPUWorkTail = &GPUWorkHead;
        twist(GPUWorkWaiting, BY, -1);

        if (g->op == 2000) { /* JOB_DONE */
#ifdef GPU_DOUBLE_BUFFER
            /* Drain pipeline: wait for last pending dispatch */
            if (pipeline_prev_g) {
                int nhits_drain = 0;
                uint32_t *hits_drain = gpu_metal_wait_slot(pipeline_prev_slot, &nhits_drain);
                if (hits_drain && nhits_drain > 0) {
                    synthetic_job.op = pipeline_prev_g->op;
                    synthetic_job.flags = pipeline_prev_g->flags;
                    synthetic_job.filename = pipeline_prev_g->filename;
                    synthetic_job.doneprint = pipeline_prev_g->doneprint;
                    synthetic_job.found = (unsigned int *)&found;
                    synthetic_job.outlen = 0;
                    int drain_cat = gpu_op_category(pipeline_prev_g->op);
                    int drain_stored = nhits_drain > GPU_MAX_RETURN ? GPU_MAX_RETURN : nhits_drain;
                    int drain_stride = (Maxiter > 1 || drain_cat == GPU_CAT_SALTPASS) ? 7 : 6;
                    int drain_hexlen = 32;
                    for (int h = 0; h < drain_stored; h++) {
                        uint32_t *entry = hits_drain + h * drain_stride;
                        int widx = entry[0], sidx = entry[1];
                        if ((unsigned)widx >= (unsigned)pipeline_prev_g->count || sidx < 0 || sidx >= prev_nsalts_packed) continue;
                        int iter_num = (drain_stride >= 7) ? entry[2] : 1;
                        int eoff = (drain_stride >= 7) ? 3 : 2;
                        for (int w = 0; w < 4; w++) curin.i[w] = entry[eoff + w];
                        char *pass = &pipeline_prev_g->passbuf[pipeline_prev_g->passoff[widx]];
                        int plen = pipeline_prev_g->passlen[widx];
                        memcpy(synthetic_job.line, pass, plen);
                        synthetic_job.line[plen] = 0;
                        synthetic_job.clen = pipeline_prev_g->clen[widx];
                        synthetic_job.pass = synthetic_job.line;
                        synthetic_job.Ruleindex = pipeline_prev_g->ruleindex[widx];
                        int snap_idx = pack_map[sidx];
                        char *s1 = saltsnap[snap_idx].salt;
                        if (iter_num <= 1 && drain_cat == GPU_CAT_SALTED) {
                            if (checkhashkey(&curin, drain_hexlen, s1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                        } else {
                            if (checkhashsalt(&curin, drain_hexlen, s1, saltsnap[snap_idx].saltlen, iter_num, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                        }
                    }
                    if (synthetic_job.outlen > 0) {
                        fwrite(outbuf, synthetic_job.outlen, 1, stdout);
                        fflush(stdout);
                        synthetic_job.outlen = 0;
                    }
                }
                hashcnt += (uint64_t)pipeline_prev_g->count * nsalts_packed * Maxiter;
                /* Return prev JOBG */
                pipeline_prev_g->next = NULL;
                pipeline_prev_g->count = 0;
                pipeline_prev_g->passbuf_pos = 0;
                possess(GPUFreeWaiting);
                if (GPUFreeTail) { *GPUFreeTail = pipeline_prev_g; GPUFreeTail = &(pipeline_prev_g->next); }
                else { GPUFreeHead = pipeline_prev_g; GPUFreeTail = &(pipeline_prev_g->next); }
                twist(GPUFreeWaiting, BY, +1);
            }
#endif
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

        /* Load overflow table once (first thread wins) */
        if (!overflow_loaded && OverflowHash) {
            if (__sync_bool_compare_and_swap(&overflow_loaded, 0, 1))
                load_overflow();
        }

        /* Rebuild salt snapshot on op change or periodically to pick up PV changes */
        batch_count++;
        { int op_cat_rebuild = gpu_op_category(g->op);
        if (g->op != current_op ||
            (salt_refresh && op_cat_rebuild != GPU_CAT_SALTED) ||
            (batch_count >= 10 && nsalts_packed > 0)) {
                salt_refresh = 0;
            if (g->op != current_op) {
                current_op = g->op;
                gpu_metal_set_op(g->op);
            }
            batch_count = 0;
            tsalt[0] = 0;
            { int use_hs = (g->op == JOB_MD5_MD5SALTMD5PASS);
              if (use_hs && Typehashsalt[g->op])
                nsalts = build_hashsalt_snapshot(saltsnap, saltpool,
                                Typehashsalt[g->op], tsalt, Printall);
              else
                nsalts = build_salt_snapshot(saltsnap, saltpool,
                                Typesalt[g->op], tsalt, Printall);
            }
            if (nsalts > 0) {
                int use_hs = (g->op == JOB_MD5_MD5SALTMD5PASS);
                nsalts_packed = gpu_pack_salts(saltsnap, nsalts,
                                               salts_packed, soff, slen, pack_map, use_hs);
            }
            else
                nsalts_packed = 0;
#ifndef GPU_DOUBLE_BUFFER
            if (nsalts_packed > 0)
                gpu_metal_set_salts(salts_packed, soff, slen, nsalts_packed);
            else
                Typedone[g->op] = 1;
#endif
        }
        } /* op_cat_rebuild scope */

        int nhits = 0;
        uint32_t *hits = NULL;
        int op_cat = gpu_op_category(g->op);

        if (g->count == 0 || nsalts_packed == 0) goto return_jobg;

#ifdef GPU_DOUBLE_BUFFER
        /* Pipelined dispatch: submit current batch to GPU, then
         * wait+process the PREVIOUS batch's results while GPU works.
         * On first iteration, just submit and loop (no prev to process).
         *
         * Flow: submit(cur) → wait(prev) → process(prev) → return(prev)
         *       save cur as prev → dequeue next → repeat */
        { int cur_slot = pipeline_slot;
          pipeline_slot = 1 - cur_slot;

          /* Submit current batch — GPU starts immediately */
          gpu_metal_submit_slot(cur_slot,
              g->word_stride ? g->raw : g->hexhash[0], (const uint16_t *)g->hexlen, g->count,
              salts_packed, soff, slen, nsalts_packed);

          if (!pipeline_prev_g) {
              /* First dispatch — nothing to wait for yet.
               * Save salt state and g as prev, loop for next job. */
              pipeline_prev_g = g;
              pipeline_prev_nsalts = nsalts_packed;
              pipeline_prev_slot = cur_slot;
              memcpy(prev_saltsnap, saltsnap, nsalts_packed * sizeof(struct saltentry));
              memcpy(prev_pack_map, pack_map, nsalts_packed * sizeof(int));
              prev_nsalts_packed = nsalts_packed;
              continue;  /* skip hit processing and return_jobg */
          }

          /* Wait for previous dispatch to complete */
          hits = gpu_metal_wait_slot(pipeline_prev_slot, &nhits);

          /* Set up context from PREVIOUS batch for hit processing */
          synthetic_job.op = pipeline_prev_g->op;
          synthetic_job.flags = pipeline_prev_g->flags;
          synthetic_job.filename = pipeline_prev_g->filename;
          synthetic_job.doneprint = pipeline_prev_g->doneprint;
          synthetic_job.found = (unsigned int *)&found;
          synthetic_job.outlen = 0;
          op_cat = gpu_op_category(pipeline_prev_g->op);

          /* Swap g to prev for hit processing.
           * Use prev's saved salt state for hit verification.
           * Save current salt state for next iteration. */
          pipeline_cur_g = g;
          g = pipeline_prev_g;

          /* Temporarily swap in prev salt state for hit processing */
          struct saltentry *save_snap = saltsnap;
          int *save_map = pack_map;
          int save_nsp = nsalts_packed;
          saltsnap = prev_saltsnap;
          pack_map = prev_pack_map;
          nsalts_packed = prev_nsalts_packed;
          /* Save current state into prev buffers for next iteration */
          prev_saltsnap = save_snap;
          prev_pack_map = save_map;
          prev_nsalts_packed = save_nsp;
          pipeline_prev_slot = cur_slot;
        }
#else
        /* Set up synthetic job context */
        synthetic_job.op = g->op;
        synthetic_job.flags = g->flags;
        synthetic_job.filename = g->filename;
        synthetic_job.doneprint = g->doneprint;
        synthetic_job.found = (unsigned int *)&found;
        synthetic_job.outlen = 0;

        /* PHPBB3: group salts by iteration count, dispatch each group.
         * dispatch_batch handles salt chunking internally (single command buffer).
         * Hits processed in the normal path below. */
        if (g->op == JOB_PHPBB3) {
            int grp_start[64], grp_count[64], grp_iter[64];
            int ngroups = phpbb3_group_salts(salts_packed, soff, slen,
                                             pack_map, nsalts_packed,
                                             grp_start, grp_count, grp_iter);
            synthetic_job.op = g->op;
            synthetic_job.flags = g->flags;
            synthetic_job.filename = g->filename;
            synthetic_job.doneprint = g->doneprint;
            synthetic_job.found = (unsigned int *)&found;
            synthetic_job.outlen = 0;

            nhits = 0;
            for (int gi = 0; gi < ngroups; gi++) {
                /* Upload this group's salts; dispatch_batch chunks internally */
                gpu_metal_set_salts(salts_packed, soff + grp_start[gi],
                                    slen + grp_start[gi], grp_count[gi]);
                gpu_metal_set_iter_count(grp_iter[gi]);
                int ghits = 0;
                hits = gpu_metal_dispatch_batch(
                    g->hexhash[0], (const uint16_t *)g->hexlen,
                    g->count, &ghits);

                /* Process this group's hits immediately (buffer reused next group) */
                if (hits && ghits > 0) {
                    int stored = ghits > GPU_MAX_RETURN ? GPU_MAX_RETURN : ghits;
                    for (int h = 0; h < stored; h++) {
                        uint32_t *entry = hits + h * 6;
                        int widx = entry[0], sidx = entry[1];
                        if ((unsigned)widx >= (unsigned)g->count || sidx < 0 || sidx >= grp_count[gi]) continue;
                        for (int w = 0; w < 4; w++) curin.i[w] = entry[2 + w];
                        int pack_idx = grp_start[gi] + sidx;
                        int snap_idx = pack_map[pack_idx];
                        char *s1 = saltsnap[snap_idx].salt;
                        char *pass = &g->passbuf[g->passoff[widx]];
                        int plen = g->passlen[widx];
                        memcpy(synthetic_job.line, pass, plen);
                        synthetic_job.line[plen] = 0;
                        synthetic_job.clen = g->clen[widx];
                        synthetic_job.pass = synthetic_job.line;
                        synthetic_job.Ruleindex = g->ruleindex[widx];
                        synthetic_job.outlen = 0;
                        if (checkhashbb(&curin, 32, s1, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    }
                    if (synthetic_job.outlen > 0) {
                        fwrite(outbuf, synthetic_job.outlen, 1, stdout);
                        fflush(stdout);
                        synthetic_job.outlen = 0;
                    }
                }
                hashcnt += (uint64_t)g->count * grp_count[gi] * grp_iter[gi];
            }
            /* Restore full salt set and reset iter_count */
            if (nsalts_packed > 0)
                gpu_metal_set_salts(salts_packed, soff, slen, nsalts_packed);
            gpu_metal_set_iter_count(0);
            /* Flush hashcnt (goto return_jobg skips the normal flush) */
            if (hashcnt > 0 || found > 0) {
                possess(FreeWaiting);
                Tothash += hashcnt;
                Totfound += found;
                release(FreeWaiting);
                hashcnt = 0;
                found = 0;
            }
            goto return_jobg;
        }

        /* Standard dispatch for non-PHPBB3 types */
        hits = gpu_metal_dispatch_batch(
            g->word_stride ? g->raw : g->hexhash[0], (const uint16_t *)g->hexlen,
            g->count, &nhits);
#endif

        {
        if (hits && nhits > 0) {
            salt_refresh = 1;
            int stored = nhits > GPU_MAX_RETURN ? GPU_MAX_RETURN : nhits;
            int is_sha256 = (g->op == JOB_SHA256PASSSALT || g->op == JOB_SHA256SALTPASS);
            int is_phpbb3 = (g->op == JOB_PHPBB3);
            int is_descrypt = (g->op == JOB_DESCRYPT);
            int hit_stride = is_sha256 ? 11
                : (is_phpbb3 || is_descrypt) ? 6
                : (Maxiter > 1 || op_cat == GPU_CAT_SALTPASS) ? 7 : 6;
            int hash_words = is_sha256 ? 8 : 4;
            int hexlen = is_sha256 ? 64 : 32;
            int overflow = (nhits > GPU_MAX_RETURN);

            /* On overflow, retry ALL words on CPU — partial hits may be stored */

            for (int h = 0; h < stored; h++) {
                uint32_t *entry = hits + h * hit_stride;
                int widx = entry[0];
                int sidx = entry[1];
                if ((unsigned)widx >= (unsigned)g->count || sidx < 0) continue;
                if (sidx >= nsalts_packed) continue;



                int iter_num;
                if (hit_stride >= 7) {
                    iter_num = entry[2];
                    for (int w = 0; w < hash_words; w++)
                        curin.i[w] = entry[3 + w];
                } else {
                    iter_num = 1;
                    for (int w = 0; w < hash_words; w++)
                        curin.i[w] = entry[2 + w];
                }

                if (op_cat == GPU_CAT_MASK || op_cat == GPU_CAT_UNSALTED) {
                    /* Mask/unsalted: reconstruct candidate from base word + mask index */
                    synthetic_job.Ruleindex = (widx < GPUBATCH_MAX) ? g->ruleindex[widx] : 0;
                    uint32_t midx = entry[1];
                    char *base_word;
                    int blen;
                    if (g->word_stride && widx < 4096) {
                        char *slot = g->raw + widx * g->word_stride;
                        int total_len = ((uint32_t *)slot)[14] >> 3;
                        blen = total_len - gpu_mask_n_prepend - gpu_mask_n_append;
                        base_word = slot + gpu_mask_n_prepend;
                    } else {
                        base_word = &g->passbuf[g->passoff[widx]];
                        blen = g->passlen[widx];
                    }
                    uint32_t append_combos = 1;
                    for (int mi = 0; mi < gpu_mask_n_append; mi++)
                        append_combos *= mask_csizes[gpu_mask_desc[gpu_mask_n_prepend + mi]];
                    uint32_t prepend_idx = midx / append_combos;
                    uint32_t append_idx = midx % append_combos;
                    char cand[256];
                    int clen = 0;
                    if (gpu_mask_n_prepend > 0)
                        clen += mask_decode(prepend_idx, gpu_mask_desc, gpu_mask_n_prepend, cand);
                    memcpy(cand + clen, base_word, blen);
                    clen += blen;
                    if (gpu_mask_n_append > 0)
                        clen += mask_decode(append_idx, gpu_mask_desc + gpu_mask_n_prepend,
                                            gpu_mask_n_append, cand + clen);
                    cand[clen] = 0;
                    memcpy(synthetic_job.line, cand, clen + 1);
                    synthetic_job.clen = clen;
                    synthetic_job.pass = synthetic_job.line;
                    checkhash(&curin, hexlen, 1, &synthetic_job);
                } else {
                char *pass = &g->passbuf[g->passoff[widx]];
                int plen = g->passlen[widx];
                memcpy(synthetic_job.line, pass, plen);
                synthetic_job.line[plen] = 0;
                synthetic_job.clen = g->clen[widx];
                synthetic_job.pass = synthetic_job.line;
                synthetic_job.Ruleindex = g->ruleindex[widx];

                int snap_idx = pack_map[sidx];
                char *s1 = saltsnap[snap_idx].salt;
                int saltlen = saltsnap[snap_idx].saltlen;

                if (is_phpbb3) {
                    if (checkhashbb(&curin, 32, s1, &synthetic_job))
                        PV_DEC(saltsnap[snap_idx].PV);
                } else if (is_descrypt) {
                    char desbuf[16];
                    des_reconstruct(curin.i[0], curin.i[1], s1, desbuf);
                    PV_DEC(saltsnap[snap_idx].PV);
                    int cplen = (plen > 8) ? 8 : plen;
                    memcpy(synthetic_job.line, pass, cplen);
                    synthetic_job.line[cplen] = 0;
                    synthetic_job.pass = synthetic_job.line;
                    synthetic_job.clen = cplen;
                    prfound(&synthetic_job, desbuf);
                } else if (iter_num <= 1 && op_cat == GPU_CAT_SALTED) {
                    if (checkhashkey(&curin, hexlen, s1, &synthetic_job))
                        PV_DEC(saltsnap[snap_idx].PV);
                } else {
                    if (checkhashsalt(&curin, hexlen, s1, saltlen, iter_num, &synthetic_job))
                        PV_DEC(saltsnap[snap_idx].PV);
                }
                } /* end non-mask else */
            }

            /* Hit buffer overflow: reprocess words that may have lost hits on CPU */
            if (overflow) {
                for (int wi = 0; wi < g->count; wi++) {
                    synthetic_job.clen = g->clen[wi];
                    synthetic_job.Ruleindex = g->ruleindex[wi];
                    for (int si = 0; si < nsalts_packed; si++) {
                        int snap_idx = pack_map[si];
                        char *s1 = saltsnap[snap_idx].salt;
                        int saltlen = saltsnap[snap_idx].saltlen;
                        char tmpbuf[1024];
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
                        case JOB_PHPBB3: {
                            pass = &g->passbuf[g->passoff[wi]]; plen = g->passlen[wi];
                            /* Decode iteration count from salt[3] */
                            int l2c = 0;
                            for (int kk = 0; kk < 64; kk++)
                                if ("./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[kk] == s1[3])
                                    { l2c = kk; break; }
                            uint32_t cnt = 1u << l2c;
                            /* MD5(salt[4..11] + pass) */
                            memcpy(tmpbuf, s1 + 4, 8);
                            memcpy(tmpbuf + 8, pass, plen);
                            mymd5(tmpbuf, 8 + plen, curin.h);
                            /* Iterate */
                            for (uint32_t ic2 = 0; ic2 < cnt; ic2++) {
                                memcpy(tmpbuf, curin.h, 16);
                                memcpy(tmpbuf + 16, pass, plen);
                                mymd5(tmpbuf, 16 + plen, curin.h);
                            }
                            memcpy(synthetic_job.line, pass, plen);
                            synthetic_job.line[plen] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            if (checkhashbb(&curin, 32, s1, &synthetic_job))
                                PV_DEC(saltsnap[snap_idx].PV);
                            break;
                        }
                        }
                    }
                }
            }
        }
        } /* op_cat scope */

#ifdef GPU_DOUBLE_BUFFER
        hashcnt += (uint64_t)g->count * nsalts_packed * Maxiter;  /* nsalts_packed is prev's after swap */
#else
        hashcnt += (uint64_t)g->count * nsalts_packed * Maxiter;
#endif

        /* Flush output */
        if (synthetic_job.outlen > 0) {
            fwrite(outbuf, synthetic_job.outlen, 1, stdout);
            fflush(stdout);
            synthetic_job.outlen = 0;
        }

        /* Update global stats periodically */
        if (hashcnt > 10000000 || found > 0) {
            possess(FreeWaiting);
            Tothash += hashcnt;
            Totfound += found;
            release(FreeWaiting);
            hashcnt = 0;
            found = 0;
        }

return_jobg:
#ifdef GPU_DOUBLE_BUFFER
        /* In pipeline mode: g is the prev JOBG whose hits we processed.
         * Return it, then save the current (still on GPU) as new prev. */
        if (pipeline_cur_g) {
            pipeline_prev_g = pipeline_cur_g;
            pipeline_prev_nsalts = nsalts_packed;
            pipeline_cur_g = NULL;
        }
        /* else: non-pipeline path (count==0 or nsalts==0), just return g */
#endif
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

    /* Final stats flush */
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
#ifdef GPU_DOUBLE_BUFFER
    free(prev_saltsnap);
    free(prev_pack_map);
#endif
}

/* ---- Public API ---- */

int gpujob_init(int num_jobg) {
    if (!gpu_metal_available()) return -1;

    /* Scan Typesaltcnt/Typesaltbytes for max sizes (procjob pattern) */
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

    gpu_metal_set_max_iter(Maxiter);

#ifdef GPU_DOUBLE_BUFFER
    if (gpu_metal_init_slots(_max_salt_count, _max_salt_bytes) != 0) return -1;
#endif

    _num_jobg_buffers = num_jobg;
    GPUWorkWaiting = new_lock(0);
    GPUFreeWaiting = new_lock(num_jobg);
    GPUWorkTail = &GPUWorkHead;

    /* Scheduler: pre-allocate waiter slots */
    GPUSchedLock = new_lock(0);
    release(GPUSchedLock);
    for (int i = 0; i < num_jobg; i++) {
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
    fprintf(stderr, "Metal GPU: %d gpujob thread%s started (%d batch buffers)\n",
            _gpujob_count, _gpujob_count > 1 ? "s" : "", num_jobg);
    return 0;
}

void gpujob_shutdown(void) {
    if (!_gpujob_ready) return;

    /* Wake all threads blocked in the priority scheduler */
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

    /* Wait for GPU work queue to drain */
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
        gpu_sched_filename = filename;
        gpu_sched_curline = startline;
        gpu_sched_active = 1;
        gpu_sched_active_count++;
        release(GPUSchedLock);
    } else if (filename == gpu_sched_filename && startline <= gpu_sched_curline) {
        if (startline < gpu_sched_curline)
            gpu_sched_curline = startline;
        gpu_sched_active_count++;
        release(GPUSchedLock);
    } else if (filename == gpu_sched_filename) {
        /* Same file, higher line — grant without blocking */
        gpu_sched_active_count++;
        release(GPUSchedLock);
    } else {
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

        possess(my_wake);
        wait_for(my_wake, TO_BE, 1);
        twist(my_wake, TO, 0);

        possess(GPUSchedLock);
        w->next = gpu_waiter_pool;
        gpu_waiter_pool = w;
        release(GPUSchedLock);
    }

get_buffer:
    possess(GPUFreeWaiting);
    wait_for(GPUFreeWaiting, NOT_TO_BE, 0);
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

    possess(GPUSchedLock);
    gpu_sched_active_count--;
    gpu_sched_wake_best();
    release(GPUSchedLock);
}

/* Non-blocking version: returns NULL immediately if no free buffer.
 * Used by hybrid types (PHPBB3) where CPU fallback is preferred over waiting. */
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
    return GPUBATCH_MAX;
}

extern "C" int gpujob_queue_depth(void) {
    if (!_gpujob_ready) return 0;
    return (int)peek_lock(GPUWorkWaiting);
}

extern "C" int gpujob_free_count(void) {
    if (!_gpujob_ready) return 0;
    return (int)peek_lock(GPUFreeWaiting);
}

int gpu_op_category(int op) {
    switch (op) {
    case JOB_MD5SALT:
    case JOB_MD5UCSALT:
    case JOB_MD5revMD5SALT:
    case JOB_MD5sub8_24SALT:
    case JOB_MD5_MD5SALTMD5PASS:
        return GPU_CAT_SALTED;
    case JOB_MD5SALTPASS:
    case JOB_MD5PASSSALT:
    case JOB_SHA256SALTPASS:
    case JOB_SHA256PASSSALT:
    case JOB_PHPBB3:
    case JOB_DESCRYPT:
        return GPU_CAT_SALTPASS;
    /* Bare MD5 iterated -- disabled pending larger batch sizes
    case JOB_MD5:
    case JOB_MD5UC:
        return GPU_CAT_ITER;
    */
    default:
        return GPU_CAT_NONE;
    }
}

int is_gpu_op(int op) {
    return gpu_op_category(op) != GPU_CAT_NONE;
}

#endif /* __APPLE__ && METAL_GPU */
