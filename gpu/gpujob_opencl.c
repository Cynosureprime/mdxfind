/*
 * gpujob_cuda.c — GPU worker thread for mdxfind CUDA acceleration
 *
 * Same architecture as gpujob.m (Metal), but uses gpu_opencl.h API.
 * Compiled only on Linux with OPENCL_GPU defined.
 */

#if defined(OPENCL_GPU)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <stdatomic.h>
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
extern void mymd5(char *, int, unsigned char *);
extern void mysha256(char *, int, unsigned char *);
extern void snapshot_attach_hashsalt(struct saltentry *, int, Pvoid_t);
extern Pvoid_t *Typehashsalt;
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
    char *hashsalt;
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
        gpu_opencl_set_overflow(dev_idx, okeys, ohashes, ooffsets, olengths, ocnt);
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
    /* Buffer must hold either raw salts OR 32-byte hex hashes per salt */
    size_t sp_size = _max_salt_bytes + 4096;
    if ((size_t)_max_salt_count * 32 + 4096 > sp_size)
        sp_size = (size_t)_max_salt_count * 32 + 4096;
    char *salts_packed = (char *)malloc(sp_size);
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
        if (op_cat == GPU_CAT_SALTED || op_cat == GPU_CAT_SALTPASS) {
            batch_count++;
            if (g->op != current_op || (batch_count >= 100 && nsalts_packed > 0)) {
                if (g->op != current_op) {
                    current_op = g->op;
                    gpu_opencl_set_op(g->op);
                }
                batch_count = 0;
                tsalt[0] = 0;
                nsalts = build_salt_snapshot(saltsnap, saltpool,
                                Typesalt[g->op], tsalt, Printall);
                if (nsalts > 0) {
                    int use_hs = (g->op == JOB_MD5_MD5SALTMD5PASS ||
                                  g->op == JOB_SHA1_MD5_MD5SALTMD5PASS ||
                                  g->op == JOB_SHA1_MD5_MD5SALTMD5PASS_SALT ||
                                  g->op == JOB_SHA1_MD5PEPPER_MD5SALTMD5PASS);
                    if (use_hs)
                        snapshot_attach_hashsalt(saltsnap, nsalts,
                                        Typehashsalt[g->op]);
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
        if ((op_cat == GPU_CAT_SALTED || op_cat == GPU_CAT_SALTPASS) && nsalts_packed == 0) goto return_jobg;

        synthetic_job.op = g->op;
        synthetic_job.flags = g->flags;
        synthetic_job.filename = g->filename;
        synthetic_job.doneprint = g->doneprint;
        synthetic_job.found = (unsigned int *)&found;
        synthetic_job.outlen = 0;

        hits = gpu_opencl_dispatch_batch(my_slot,
            g->hexhash[0], (const uint16_t *)g->hexlen,
            g->count, &nhits);

        if (hits && nhits > 0) {
            int stored = nhits > GPU_MAX_RETURN ? GPU_MAX_RETURN : nhits;
            int is_sha256 = (g->op == JOB_SHA256PASSSALT || g->op == JOB_SHA256SALTPASS);
            int is_md5crypt = (g->op == JOB_MD5CRYPT);
            int hit_stride = is_sha256 ? 11
                : is_md5crypt ? 6
                : (Maxiter > 1 || op_cat == GPU_CAT_ITER || op_cat == GPU_CAT_SALTPASS) ? 7 : 6;
            int hash_words = is_sha256 ? 8 : 4;
            int hexlen = is_sha256 ? 64 : 32;
            int overflow = (nhits > GPU_MAX_RETURN);

            /* On overflow, we must retry ALL words on CPU since any word's
             * hits may have been partially stored. word_hit is not used. */

            for (int h = 0; h < stored; h++) {
                uint32_t *entry = hits + h * hit_stride;
                int widx = entry[0];
                int sidx = entry[1];
                if ((unsigned)widx >= (unsigned)g->count || sidx < 0) continue;


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

                char *pass = &g->passbuf[g->passoff[widx]];
                int plen = g->passlen[widx];
                memcpy(synthetic_job.line, pass, plen);
                synthetic_job.line[plen] = 0;
                synthetic_job.clen = g->clen[widx];
                synthetic_job.pass = synthetic_job.line;
                synthetic_job.Ruleindex = g->ruleindex[widx];

                if (op_cat == GPU_CAT_ITER) {
                    checkhash(&curin, hexlen, iter_num, &synthetic_job);
                } else if (is_md5crypt) {
                    /* MD5CRYPT: verify via hybrid_check, output $1$salt$base64 */
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    int match_len; unsigned short *match_flags;
                    int hf = hybrid_check(curin.h, 16, &match_len, &match_flags);
                    if (hf && *match_flags != (unsigned short)g->op) {
                        *match_flags = g->op;
                        PV_DEC(saltsnap[snap_idx].PV);
                        /* Reconstruct $1$salt$base64 output */
                        char *sp = saltsnap[snap_idx].salt; /* "$1$salt$" */
                        int splen = saltsnap[snap_idx].saltlen;
                        char mdbuf[128];
                        memcpy(mdbuf, sp, splen);
                        md5crypt_b64encode(curin.h, mdbuf + splen);
                        prfound(&synthetic_job, mdbuf);
                    }
                } else {
                    if (sidx >= nsalts_packed) continue;
                    int snap_idx = pack_map[sidx];
                    char *s1 = saltsnap[snap_idx].salt;
                    int saltlen = saltsnap[snap_idx].saltlen;

                    if (iter_num <= 1 && op_cat == GPU_CAT_SALTED) {
                        if (checkhashkey(&curin, hexlen, s1, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    } else {
                        if (checkhashsalt(&curin, hexlen, s1, saltlen, iter_num, &synthetic_job))
                            PV_DEC(saltsnap[snap_idx].PV);
                    }
                }
            }

            /* Hit buffer overflow: reprocess words that may have lost hits on CPU.
             * word_hit[wi] is set for words that appeared in the stored hits.
             * Words WITHOUT hits may have been dropped — recompute on CPU. */
            if (overflow) {
                for (int wi = 0; wi < g->count; wi++) {
                    synthetic_job.clen = g->clen[wi];
                    synthetic_job.Ruleindex = g->ruleindex[wi];

                    for (int si = 0; si < nsalts_packed; si++) {
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
                              if (!hs) break; /* no precomputed hash — shouldn't happen */
                              memcpy(tmpbuf, hs, 32);
                              memcpy(tmpbuf + 32, g->hexhash[wi], g->hexlen[wi]);
                              mymd5(tmpbuf, 32 + g->hexlen[wi], curin.h);
                            }
                            memcpy(synthetic_job.line, g->hexhash[wi], g->hexlen[wi]);
                            synthetic_job.line[g->hexlen[wi]] = 0;
                            synthetic_job.pass = synthetic_job.line;
                            /* Use original salt for output (not the hex hash) */
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
                        case JOB_MD5CRYPT: {
                            /* Full MD5CRYPT compute for overflow retry */
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
                        }
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

int gpujob_available(void) {
    return _gpujob_ready;
}

int gpujob_batch_max(void) {
    return _gpu_batch_max;
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
    case JOB_MD5CRYPT:
        return GPU_CAT_SALTPASS;
    case JOB_MD5_MD5SALTMD5PASS:
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
