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

/* Kernel family IDs for gpu_opencl_compile_families() bitmask */
enum {
    FAM_MD5SALT, FAM_MD5SALTPASS, FAM_MD5ITER, FAM_PHPBB3,
    FAM_MD5CRYPT, FAM_MD5_MD5SALTMD5PASS, FAM_SHA1, FAM_SHA256,
    FAM_MD5MASK, FAM_DESCRYPT, FAM_MD5UNSALTED, FAM_MD4UNSALTED,
    FAM_SHA1UNSALTED, FAM_SHA256UNSALTED, FAM_SHA512UNSALTED,
    FAM_WRLUNSALTED, FAM_MD6256UNSALTED, FAM_KECCAKUNSALTED,
    FAM_HMAC_SHA512, FAM_MYSQL3UNSALTED, FAM_HMAC_RMD160, FAM_HMAC_RMD320,
    FAM_HMAC_BLAKE2S, FAM_STREEBOG, FAM_SHA512CRYPT, FAM_SHA256CRYPT,
    FAM_RMD160UNSALTED, FAM_BLAKE2S256UNSALTED, FAM_BCRYPT, FAM_COUNT
};

/* Forward declarations for mdxfind types */
struct job;
union HashU;

#ifdef __cplusplus
extern "C" {
#endif

/* Checked malloc: zeroes memory, exits on failure */
void *malloc_lock(size_t size, const char *reason);

#define GPUBATCH_MAX   512
#define GPUBATCH_PASS  (GPUBATCH_MAX * 256)  /* ~128KB password buffer */

/* GPU algorithm categories */
#define GPU_CAT_NONE     0   /* not GPU-capable */
#define GPU_CAT_SALTED   1   /* salted: GPU does MD5(hex_hash + salt) per salt */
#define GPU_CAT_ITER     2   /* unsalted iterated: GPU does MD5(hex) iterations */
#define GPU_CAT_SALTPASS 3   /* salted: GPU does MD5(salt + raw_password) per salt */
#define GPU_CAT_MASK     4   /* unsalted + mask: GPU generates mask candidates */
#define GPU_CAT_UNSALTED 5   /* unsalted pre-padded: GPU fills masks into M[] */

#define GPU_MAX_PASSLEN  55  /* max password length for GPU (single MD5 block with salt) */

/* Universal GPU hit entry stride: 19 uint32 words per hit.
 * Layout: [0]=word_idx [1]=salt_idx [2]=iter_num [3..18]=hash[0..15]
 * All kernels emit this format. No exceptions. */
#define GPU_HIT_STRIDE   19

/* Returns GPU category for an op code, or GPU_CAT_NONE */
int gpu_op_category(int op);

/* Returns 1 if op has any GPU support */
int is_gpu_op(int op);

/* Returns the number of uint32 hash words this algorithm produces.
 * Used to determine hit_stride, hash_words, and hexlen for GPU hits.
 * hexlen = hash_words * 8; hit_stride = 2 + hash_words (+ 1 if iter). */
int gpu_hash_words(int op);

struct jobg {
    struct jobg *next;
    char        *filename;
    /* Word data: 256KB contiguous for unsalted 64-byte stride (4096 words).
     * Legacy accesses first half as passbuf, second half as hexhash[]. */
    union {
        char     raw[GPUBATCH_PASS + GPUBATCH_MAX * 256]; /* 256KB contiguous */
        struct { char passbuf[GPUBATCH_PASS]; char hexhash[GPUBATCH_MAX][256]; };
    };
    uint16_t     hexlen[GPUBATCH_MAX];

    /* Password data for checkhashkey output */
    uint32_t     passoff[GPUBATCH_MAX];
    uint16_t     passlen[GPUBATCH_MAX];

    /* Job context for checkhashkey */
    int          clen[GPUBATCH_MAX];
    int          ruleindex[GPUBATCH_MAX];
    int         *doneprint;
    int          flags;
    int          op;                        /* JOB_MD5SALT etc. */
    int          count;                     /* entries filled (0..GPUBATCH_MAX) */
    int          max_count;                 /* max entries for this batch (stride-dependent) */
    uint32_t     passbuf_pos;               /* fill cursor into passbuf */
    uint32_t     word_stride;               /* bytes per word slot (64, 128, 256) */
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

/* Non-blocking: returns NULL immediately if no free buffer.
 * Used by hybrid types where CPU fallback is preferred over waiting. */
struct jobg *gpujob_try_get_free(void);

/* Submit a filled JOBG to the GPU work queue. */
void gpujob_submit(struct jobg *g);

/* Return a JOBG to the free list without submitting for work. */
void gpujob_return_free(struct jobg *g);

/* Returns 1 if GPU job system is initialized and ready. */
int gpujob_available(void);

/* Returns per-device batch limit (min across all GPUs). */
int gpujob_batch_max(void);

#ifdef __cplusplus
}
#endif

#endif /* (__APPLE__ && METAL_GPU) || CUDA_GPU || OPENCL_GPU */
#endif /* GPUJOB_H */
