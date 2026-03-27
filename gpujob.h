/*
 * gpujob.h — Dedicated GPU worker thread for mdxfind Metal acceleration
 *
 * Receives batches of pre-hashed words from procjob() threads,
 * dispatches them to the GPU as a single Metal command.
 * Only available on Apple Silicon with METAL_GPU defined.
 */

#ifndef GPUJOB_H
#define GPUJOB_H

#if defined(__APPLE__) && defined(METAL_GPU)

#include <stdint.h>

/* Forward declarations for mdxfind types */
struct job;
union HashU;

#ifdef __cplusplus
extern "C" {
#endif

#define GPUBATCH_MAX   512
#define GPUBATCH_PASS  (GPUBATCH_MAX * 256)  /* ~128KB password buffer */

struct jobg {
    struct jobg *next;
    int          op;                        /* JOB_MD5SALT etc. */
    int          count;                     /* entries filled (0..GPUBATCH_MAX) */
    uint32_t     passbuf_pos;               /* fill cursor into passbuf */

    /* Pre-computed hex hashes: sized for any algorithm */
    char         hexhash[GPUBATCH_MAX][256];
    uint16_t     hexlen[GPUBATCH_MAX];

    /* Password data for checkhashkey output */
    char         passbuf[GPUBATCH_PASS];
    uint32_t     passoff[GPUBATCH_MAX];
    uint16_t     passlen[GPUBATCH_MAX];

    /* Job context for checkhashkey */
    int          clen[GPUBATCH_MAX];
    int          ruleindex[GPUBATCH_MAX];
    char        *filename;
    int          flags;
    int         *doneprint;
};

/* Initialize GPU work queue, allocate JOBG structs, launch gpujob thread.
 * Call from main() after metal_md5salt_init() and set_compact_table().
 * Returns 0 on success, -1 on failure. */
int gpujob_init(int num_jobg);

/* Send JOB_DONE to GPU queue and join the gpujob thread.
 * Call from main() after all procjob threads have exited. */
void gpujob_shutdown(void);

/* Get a free JOBG from the pool. Blocks if none available. */
struct jobg *gpujob_get_free(void);

/* Submit a filled JOBG to the GPU work queue. */
void gpujob_submit(struct jobg *g);

/* Returns 1 if GPU job system is initialized and ready. */
int gpujob_available(void);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ && METAL_GPU */
#endif /* GPUJOB_H */
