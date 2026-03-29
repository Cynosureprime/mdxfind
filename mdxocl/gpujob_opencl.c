/*
 * gpujob_cuda.c — GPU worker thread for mdxfind CUDA acceleration
 *
 * Same architecture as gpujob.m (Metal), but uses opencl_md5salt.h API.
 * Compiled only on Linux with OPENCL_GPU defined.
 */

#if defined(OPENCL_GPU)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include "mdxfind.h"
#include "job_types.h"
#include "gpujob.h"
#include "opencl_md5salt.h"
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

#define PV_DEC(pv) { unsigned long _old = *(pv); \
  while (_old > 0) { \
    if (__sync_bool_compare_and_swap((pv), _old, _old - 1)) break; \
    _old = *(pv); } }

struct saltentry {
    char *salt;
    unsigned long *PV;
    int saltlen;
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
     * If no active threads remain, we MUST wake someone to prevent deadlock.
     * Pick lowest same-file line, or lowest (filename, line) overall. */
    if (!best && gpu_waiter_head && gpu_sched_active_count == 0) {
        struct gpu_waiter **pp = &gpu_waiter_head;
        while (*pp) {
            struct gpu_waiter *w = *pp;
            if (!best ||
                (uintptr_t)w->filename < (uintptr_t)best->filename ||
                (w->filename == best->filename &&
                 w->startline < best->startline)) {
                best = w;
                best_pp = pp;
            }
            pp = &w->next;
        }
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

#define GPU_MAX_RETURN 32768
#define OUTBUFSIZE (1024 * 1024)

static int gpu_pack_salts(struct saltentry *saltsnap, int nsalts,
                          char *salts_packed, uint32_t *soff, uint16_t *slen,
                          int *pack_map) {
    int packed = 0;
    uint32_t gsp = 0;
    for (int i = 0; i < nsalts; i++) {
        if (!Printall && *saltsnap[i].PV == 0) continue;
        soff[packed] = gsp;
        slen[packed] = saltsnap[i].saltlen;
        pack_map[packed] = i;
        memcpy(salts_packed + gsp, saltsnap[i].salt, saltsnap[i].saltlen);
        gsp += saltsnap[i].saltlen;
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
        uint64_t *okeys = (uint64_t *)malloc(ocnt * sizeof(uint64_t));
        unsigned char *ohashes = (unsigned char *)malloc(obytes + ocnt * 8);
        uint32_t *ooffsets = (uint32_t *)malloc(ocnt * sizeof(uint32_t));
        uint16_t *olengths = (uint16_t *)malloc(ocnt * sizeof(uint16_t));
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
        opencl_md5salt_set_overflow(dev_idx, okeys, ohashes, ooffsets, olengths, ocnt);
        free(okeys); free(ohashes); free(ooffsets); free(olengths);
    }
}

void gpujob(void *arg) {
    int my_slot = (int)(intptr_t)arg;
    union HashU curin;
    struct job synthetic_job;
    char *outbuf = (char *)malloc(OUTBUFSIZE);
    uint64_t hashcnt = 0, found = 0;
    char tsalt[4096];

    memset(&synthetic_job, 0, sizeof(synthetic_job));
    synthetic_job.outbuf = outbuf;

    struct saltentry *saltsnap = (struct saltentry *)malloc(
        _max_salt_count * sizeof(struct saltentry));
    char *saltpool = (char *)malloc(_max_salt_bytes + 16);
    char *salts_packed = (char *)malloc(_max_salt_bytes + 4096);
    uint32_t *soff = (uint32_t *)malloc(_max_salt_count * sizeof(uint32_t));
    uint16_t *slen = (uint16_t *)malloc(_max_salt_count * sizeof(uint16_t));
    int *pack_map = (int *)malloc(_max_salt_count * sizeof(int));
    int nsalts = 0;
    int nsalts_packed = 0;
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

        /* Rebuild salt snapshot on op change or periodically to pick up PV changes */
        if (op_cat == GPU_CAT_SALTED) {
            batch_count++;
            if (g->op != current_op || (batch_count >= 100 && nsalts_packed > 0)) {
                if (g->op != current_op) {
                    current_op = g->op;
                    opencl_md5salt_set_op(g->op);
                }
                batch_count = 0;
                tsalt[0] = 0;
                nsalts = build_salt_snapshot(saltsnap, saltpool,
                                Typesalt[g->op], tsalt, Printall);
                if (nsalts > 0)
                    nsalts_packed = gpu_pack_salts(saltsnap, nsalts,
                                                   salts_packed, soff, slen, pack_map);
                else
                    nsalts_packed = 0;
                if (nsalts_packed > 0)
                    opencl_md5salt_set_salts(my_slot, salts_packed, soff, slen, nsalts_packed);
                else
                    Typedone[g->op] = 1;
            }
        } else if (g->op != current_op) {
            current_op = g->op;
            opencl_md5salt_set_op(g->op);
        }

        int nhits = 0;
        uint32_t *hits = NULL;

        if (g->count == 0) goto return_jobg;
        if (op_cat == GPU_CAT_SALTED && nsalts_packed == 0) goto return_jobg;

        synthetic_job.op = g->op;
        synthetic_job.flags = g->flags;
        synthetic_job.filename = g->filename;
        synthetic_job.doneprint = g->doneprint;
        synthetic_job.found = (unsigned int *)&found;
        synthetic_job.outlen = 0;

        hits = opencl_md5salt_dispatch_batch(my_slot,
            g->hexhash[0], (const uint16_t *)g->hexlen,
            g->count, &nhits);


        if (hits && nhits > 0) {
            int stored = nhits > GPU_MAX_RETURN ? GPU_MAX_RETURN : nhits;
            int hit_stride = (Maxiter > 1 || op_cat == GPU_CAT_ITER) ? 7 : 6;

            for (int h = 0; h < stored; h++) {
                uint32_t *entry = hits + h * hit_stride;
                int widx = entry[0];
                int sidx = entry[1];
                if (widx >= g->count) continue;

                int iter_num;
                if (hit_stride == 7) {
                    iter_num = entry[2];
                    curin.i[0] = entry[3];
                    curin.i[1] = entry[4];
                    curin.i[2] = entry[5];
                    curin.i[3] = entry[6];
                } else {
                    iter_num = 1;
                    curin.i[0] = entry[2];
                    curin.i[1] = entry[3];
                    curin.i[2] = entry[4];
                    curin.i[3] = entry[5];
                }

                char *pass = &g->passbuf[g->passoff[widx]];
                int plen = g->passlen[widx];
                memcpy(synthetic_job.line, pass, plen);
                synthetic_job.line[plen] = 0;
                synthetic_job.clen = g->clen[widx];
                synthetic_job.pass = synthetic_job.line;
                synthetic_job.Ruleindex = g->ruleindex[widx];

                if (op_cat == GPU_CAT_ITER) {
                    /* Bare iteration: no salt, use checkhash */
                    checkhash(&curin, 32, iter_num, &synthetic_job);
                } else {
                    /* Salted: use checkhashkey/checkhashsalt with salt lookup */
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    char *s1 = saltsnap[snap_idx].salt;
                    int saltlen = saltsnap[snap_idx].saltlen;

                    if (iter_num <= 1) {
                        if (checkhashkey(&curin, 32, s1, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    } else {
                        if (checkhashsalt(&curin, 32, s1, saltlen, iter_num, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    }
                }
            }
        }

        if (op_cat == GPU_CAT_ITER)
            hashcnt += (uint64_t)g->count * (Maxiter - 1);
        else
            hashcnt += (uint64_t)g->count * nsalts_packed * Maxiter;

        if (synthetic_job.outlen > 0) {
            fwrite(outbuf, synthetic_job.outlen, 1, stdout);
            fflush(stdout);
            synthetic_job.outlen = 0;
        }

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
    if (!opencl_md5salt_available()) return -1;

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

    opencl_md5salt_set_max_iter(Maxiter);

    /* One gpujob thread per GPU device */
    _gpujob_count = opencl_md5salt_num_devices();
    if (_gpujob_count < 1) _gpujob_count = 1;

    _num_jobg_buffers = num_jobg;
    GPUWorkWaiting = new_lock(0);
    GPUFreeWaiting = new_lock(num_jobg);
    GPUWorkTail = &GPUWorkHead;

    /* Scheduler: pre-allocate maxt+1 waiter slots */
    GPUSchedLock = new_lock(0);
    release(GPUSchedLock);
    _gpu_waiter_count = num_jobg;  /* at least maxt+1 */
    for (int i = 0; i < _gpu_waiter_count; i++) {
        struct gpu_waiter *w = (struct gpu_waiter *)calloc(1, sizeof(*w));
        w->wake = new_lock(0);
        w->next = gpu_waiter_pool;
        gpu_waiter_pool = w;
    }

    for (int i = 0; i < num_jobg; i++) {
        struct jobg *g = (struct jobg *)calloc(1, sizeof(struct jobg));
        if (!g) return -1;
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
    } else {
        /* Must wait — get a waiter from pool */
        struct gpu_waiter *w = gpu_waiter_pool;
        if (w) gpu_waiter_pool = w->next;
        else w = (struct gpu_waiter *)calloc(1, sizeof(*w));
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

    /* Wake the best waiting procjob thread */
    possess(GPUSchedLock);
    gpu_sched_active_count--;
    gpu_sched_wake_best();
    release(GPUSchedLock);
}

int gpujob_available(void) {
    return _gpujob_ready;
}

int gpu_op_category(int op) {
    switch (op) {
    /* Salted MD5 variants -- GPU iterates over salts */
    case JOB_MD5SALT:
    case JOB_MD5UCSALT:
    case JOB_MD5revMD5SALT:
    case JOB_MD5sub8_24SALT:
        return GPU_CAT_SALTED;
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

#endif /* OPENCL_GPU */
