/*
 * metal_md5salt.h — Apple Metal GPU acceleration for MD5SALT (E31)
 *
 * Pure C interface. Implementation in metal_md5salt.m (Objective-C).
 * Only available on Apple Silicon (ARM64 macOS with Metal support).
 */

#ifndef METAL_MD5SALT_H
#define METAL_MD5SALT_H

#if defined(__APPLE__) && defined(METAL_GPU)

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Metal device, command queue, and compute pipeline.
 * Returns 0 on success, -1 on failure (no GPU, unsupported, etc).
 * Safe to call on any macOS system — returns -1 if Metal is unavailable. */
int metal_md5salt_init(void);

/* Teardown Metal resources. */
void metal_md5salt_shutdown(void);

/* Returns 1 if Metal GPU is initialized and ready for dispatch. */
int metal_md5salt_available(void);

/* Register the compact hash table for zero-copy GPU access.
 * Must be called once after the compact table is built (after hash loading).
 * Returns 0 on success, -1 on failure. */
int metal_md5salt_set_compact_table(
    uint32_t *compact_fp,
    uint32_t *compact_idx,
    uint64_t compact_size,
    uint64_t compact_mask,
    unsigned char *hash_data_buf,
    size_t hash_data_buf_size,
    size_t *hash_data_off,
    size_t hash_data_count,
    unsigned short *hash_data_len);

/* Dispatch a batch of words x salts to the GPU for MD5SALT computation.
 *
 * For each (word, salt) pair, the GPU computes:
 *   MD5(hex(MD5(word)) + salt)
 * and probes the compact hash table for a match.
 *
 * words:        packed word buffer (all words concatenated)
 * word_offsets: offset into words[] for each word (num_words entries)
 * word_lens:    byte length of each word (num_words entries)
 * num_words:    number of words in this batch
 * salts:        packed salt buffer (all salts concatenated)
 * salt_offsets: offset into salts[] for each salt (num_salts entries)
 * salt_lens:    byte length of each salt (num_salts entries)
 * num_salts:    number of salts in this batch
 * hits_out:     caller-allocated array for hit pairs [word_idx, salt_idx, ...]
 * max_hits:     capacity of hits_out (in pairs, so max_hits*2 uint32_t entries)
 *
 * Returns: number of hits found (compact table matches).
 *          Each hit occupies 2 entries in hits_out: [word_idx, salt_idx]. */
int metal_md5salt_dispatch(
    const char *words,
    const uint32_t *word_offsets,
    const uint16_t *word_lens,
    int num_words,
    const char *salts,
    const uint32_t *salt_offsets,
    const uint16_t *salt_lens,
    int num_salts,
    uint32_t *hits_out,
    int max_hits);

/* Pre-load salt data into persistent GPU buffers.
 * Call once after packing salts, before dispatching words.
 * Avoids per-dispatch Metal buffer allocation. */
int metal_md5salt_set_salts(
    const char *salts,
    const uint32_t *salt_offsets,
    const uint16_t *salt_lens,
    int num_salts);

/* Dispatch one pre-hashed word against all pre-loaded salts.
 * hexhash:     hex MD5 of the candidate word (CPU-computed)
 * hexlen:      length of hexhash (normally 32)
 * nhits_out:   receives total number of hits (may exceed buffer capacity)
 *
 * Salts must be pre-loaded via metal_md5salt_set_salts().
 * GPU computes MD5(hexhash + salt) for each salt, probes compact table.
 * Returns pointer to hit buffer in shared GPU/CPU memory (zero-copy).
 * Each hit is 5 uint32s: {salt_index, hash[0], hash[1], hash[2], hash[3]}.
 * At most 32768 hits are stored; if *nhits_out > 32768, caller must
 * remove matched salts and re-dispatch.
 * Pointer is valid until the next call. */
uint32_t *metal_md5salt_probe_salts(
    const char *hexhash,
    int hexlen,
    int *nhits_out);

/* Load overflow hash entries into GPU buffer for binary search fallback.
 * keys:     sorted array of uint64_t keys (first 8 bytes of each hash)
 * hashes:   packed hash data (variable length per entry)
 * offsets:  byte offset into hashes for each entry
 * lengths:  byte length of each hash entry
 * count:    number of overflow entries
 * Returns 0 on success. */
int metal_md5salt_set_overflow(
    const uint64_t *keys,
    const unsigned char *hashes,
    const uint32_t *offsets,
    const uint16_t *lengths,
    int count);

/* Set maximum iteration count for GPU dispatch. Default 1.
 * Each iteration computes MD5(hex(previous_result)) and probes the compact table. */
void metal_md5salt_set_max_iter(int max_iter);

/* Dispatch a batch of pre-hashed words against all pre-loaded salts.
 * hexhashes:   packed hex hashes, 256 bytes per word (only hexlens[i] used)
 * hexlens:     length of each hex hash (cast from int* to uint16_t* by caller)
 * num_words:   number of words in this batch
 * nhits_out:   receives total number of hits (may exceed 32768)
 *
 * Salts must be pre-loaded via metal_md5salt_set_salts().
 * GPU computes MD5(hexhash + salt) for each (word, salt) pair, probes compact table.
 * Returns pointer to hit buffer in shared GPU/CPU memory (zero-copy).
 * Each hit is 6 uint32s: {word_idx, salt_idx, hash[0], hash[1], hash[2], hash[3]}.
 * At most 32768 hits stored; if *nhits_out > 32768, caller must process stored
 * hits, remove matched salts, and re-dispatch.
 * Pointer valid until next call. */
uint32_t *metal_md5salt_dispatch_batch(
    const char *hexhashes,
    const uint16_t *hexlens,
    int num_words,
    int *nhits_out);

/* Set maximum iteration count for GPU dispatch. Default 1.
 * Each iteration computes MD5(hex(previous_result)) and probes the compact table.
 * When max_iter > 1, uses a separate kernel that doesn't affect the fast path. */
void metal_md5salt_set_max_iter(int max_iter);

/* Set the current op type for GPU kernel selection.
 * Used to select specialized kernels (e.g., sub8-24 = op 542). */
void metal_md5salt_set_op(int op);

/* ---- Double-buffer slot API ---- */

/* Initialize double-buffer dispatch slots. Call once from gpujob_init.
 * max_salt_count/max_salt_bytes: maximum across all active hash types. */
int metal_md5salt_init_slots(int max_salt_count, int max_salt_bytes);

/* Submit a batch to a slot. Copies all data into pre-allocated GPU buffers.
 * Returns immediately — GPU processes asynchronously. */
int metal_md5salt_submit_slot(int slot,
    const char *hexhashes, const uint16_t *hexlens, int num_words,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts);

/* Wait for slot dispatch to complete. Returns pointer to hit buffer.
 * Each hit: 6 uint32s {word_idx, salt_idx, hash[0..3]}.
 * Pointer valid until next submit_slot on same slot. */
uint32_t *metal_md5salt_wait_slot(int slot, int *nhits_out);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ && METAL_GPU */
#endif /* METAL_MD5SALT_H */
