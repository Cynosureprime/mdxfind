/*
 * gpujob.h — Dedicated GPU worker thread for mdxfind Metal acceleration
 *
 * Receives batches of pre-hashed words from procjob() threads,
 * dispatches them to the GPU as a single Metal command.
 * Only available on Apple Silicon with METAL_GPU defined.
 */

#ifndef GPUJOB_H
#define GPUJOB_H

#if (defined(__APPLE__) && defined(METAL_GPU)) || defined(CUDA_GPU) || defined(OPENCL_GPU)

#include <stdint.h>

/* Forward declarations for mdxfind types */
struct job;
union HashU;

#ifdef __cplusplus
extern "C" {
#endif

#define GPUBATCH_MAX   512
#define GPUBATCH_PASS  (GPUBATCH_MAX * 256)  /* ~128KB password buffer */

/* GPU algorithm categories */
#define GPU_CAT_NONE     0   /* not GPU-capable */
#define GPU_CAT_SALTED   1   /* salted: GPU does MD5(hex_hash + salt) per salt */
#define GPU_CAT_ITER     2   /* unsalted iterated: GPU does MD5(hex) iterations */

/* Returns GPU category for an op code, or GPU_CAT_NONE */
int gpu_op_category(int op);

/* Returns 1 if op has any GPU support */
int is_gpu_op(int op);

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
    unsigned int line_num;              /* starting line number for priority ordering */
};

/* Initialize GPU work queue, allocate JOBG structs, launch gpujob thread.
 * Call from main() after metal_md5salt_init() and set_compact_table().
 * Returns 0 on success, -1 on failure. */
int gpujob_init(int num_jobg);

/* Send JOB_DONE to GPU queue and join the gpujob thread.
 * Call from main() after all procjob threads have exited. */
void gpujob_shutdown(void);

/* Get a free JOBG from the pool. Priority scheduling ensures earlier
 * lines in the file get GPU buffers first. Pass NULL filename for
 * shutdown sentinels (bypasses scheduling). */
struct jobg *gpujob_get_free(char *filename, unsigned int startline);

/* Submit a filled JOBG to the GPU work queue. */
void gpujob_submit(struct jobg *g);

/* Returns 1 if GPU job system is initialized and ready. */
int gpujob_available(void);

#ifdef __cplusplus
}
#endif

#endif /* (__APPLE__ && METAL_GPU) || CUDA_GPU || OPENCL_GPU */
#endif /* GPUJOB_H */
