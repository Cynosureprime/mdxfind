/*
 * gpu_opencl.h — OpenCL GPU acceleration for mdxfind
 *
 * Cross-vendor GPU support via OpenCL runtime.
 * Supports multiple GPU devices.
 */

#ifndef GPU_OPENCL_H
#define GPU_OPENCL_H

#if defined(OPENCL_GPU)

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


int gpu_opencl_init(void);
void gpu_opencl_compile_families(unsigned int fam_mask);
void gpu_opencl_shutdown(void);
int gpu_opencl_available(void);
int gpu_opencl_num_devices(void);
void gpu_opencl_list_devices(void);

/* Per-device APIs — dev_idx from 0 to num_devices-1 */
int gpu_opencl_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len);

int gpu_opencl_set_salts(int dev_idx,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts);

int gpu_opencl_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count);

void gpu_opencl_set_max_iter(int max_iter);
void gpu_opencl_set_mask_resume(uint32_t start);
void gpu_opencl_set_op(int op);
int gpu_opencl_max_batch(int dev_idx);
int gpu_opencl_set_mask(const uint8_t *prepend, int npre,
                        const uint8_t *append, int napp);

uint32_t *gpu_opencl_dispatch_batch(int dev_idx,
    const char *hexhashes, const uint16_t *hexlens,
    int num_words, int *nhits_out);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_GPU */
#endif /* GPU_OPENCL_H */
