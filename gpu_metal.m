/*
 * metal_md5salt.m — Apple Metal GPU acceleration for MD5SALT (E31)
 *
 * Thin Objective-C wrapper around Metal compute pipeline.
 * Exports a pure C interface defined in gpu_metal.h.
 */

#if defined(__APPLE__) && defined(METAL_GPU)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "gpu_metal.h"
#include "job_types.h"
#include "gpujob.h"
#include <string.h>
#include <stdio.h>
#include <mach-o/dyld.h>

/* ---- Metal per-family kernel sources (auto-generated from metal_*.metal) ---- */
#include "gpu/metal_common_str.h"
#include "gpu/metal_md5salt_str.h"
#include "gpu/metal_md5saltpass_str.h"
#include "gpu/metal_md5_md5saltmd5pass_str.h"
#include "gpu/metal_sha256_str.h"
#include "gpu/metal_phpbb3_str.h"
#include "gpu/metal_descrypt_str.h"
#include "gpu/metal_md5unsalted_str.h"

/* Family IDs from gpujob.h FAM_* enum — Metal uses a subset */
static const char *mtl_family_source[FAM_COUNT] = {
    [FAM_MD5SALT]            = metal_md5salt_str,
    [FAM_MD5SALTPASS]        = metal_md5saltpass_str,
    [FAM_MD5_MD5SALTMD5PASS] = metal_md5_md5saltmd5pass_str,
    [FAM_SHA256]             = metal_sha256_str,
    [FAM_PHPBB3]             = metal_phpbb3_str,
    [FAM_DESCRYPT]           = metal_descrypt_str,
    [FAM_MD5UNSALTED]        = metal_md5unsalted_str,
};

/* ---- Metal state ---- */
static id<MTLDevice>              mtl_device;
static id<MTLCommandQueue>        mtl_queue;
static id<MTLComputePipelineState> mtl_pipeline;       /* probe kernel */
static id<MTLComputePipelineState> mtl_pipeline_salts;  /* salts-only kernel */
static id<MTLComputePipelineState> mtl_pipeline_iter;   /* salted iteration override */
static id<MTLComputePipelineState> mtl_pipelines[JOB_DONE]; /* per-op dispatch kernels */
static id<MTLLibrary>             mtl_fam_lib[FAM_COUNT]; /* per-family compiled libraries */

/* Metal kernel-to-op mapping table (mirrors OpenCL kernel_map) */
static const struct {
    const char *name;
    int ops[8];
    int family;
} metal_kernel_map[] = {
    {"md5salt_batch_prehashed", {JOB_MD5SALT, JOB_MD5UCSALT, JOB_MD5revMD5SALT, -1}, FAM_MD5SALT},
    {"md5salt_batch_sub8_24",   {JOB_MD5sub8_24SALT, -1}, FAM_MD5SALT},
    {"md5salt_batch_iter",      {-1}, FAM_MD5SALT},  /* special: salted iteration override */
    {"md5saltpass_batch",       {JOB_MD5SALTPASS, -1}, FAM_MD5SALTPASS},
    {"md5passsalt_batch",       {JOB_MD5PASSSALT, -1}, FAM_MD5SALTPASS},
    {"sha256passsalt_batch",    {JOB_SHA256PASSSALT, -1}, FAM_SHA256},
    {"sha256saltpass_batch",    {JOB_SHA256SALTPASS, -1}, FAM_SHA256},
    {"md5_md5saltmd5pass_batch", {JOB_MD5_MD5SALTMD5PASS, -1}, FAM_MD5_MD5SALTMD5PASS},
    {"phpbb3_batch",             {JOB_PHPBB3, -1}, FAM_PHPBB3},
    {"descrypt_batch",           {JOB_DESCRYPT, -1}, FAM_DESCRYPT},
    {"md5_unsalted_batch",       {JOB_MD5, -1}, FAM_MD5UNSALTED},
    {NULL, {-1}, 0}
};
#define MTL_KERN_ITER_IDX 2
static int                        mtl_ready = 0;

/* ---- Compact table GPU buffers (zero-copy where possible) ---- */
static id<MTLBuffer> buf_compact_fp;
static id<MTLBuffer> buf_compact_idx;
static id<MTLBuffer> buf_hash_data;
static id<MTLBuffer> buf_hash_data_off;
static id<MTLBuffer> buf_hash_data_len;

/* ---- Cached compact table params ---- */
static uint64_t _compact_mask = 0;
static uint32_t _hash_data_count = 0;

/* ---- Persistent hit output buffer ---- */
#define MAX_GPU_HITS 65536
static id<MTLBuffer> buf_hits;
static id<MTLBuffer> buf_hit_count;

/* ---- Reusable dispatch buffers (avoids per-call Metal allocation) ---- */
static id<MTLBuffer> buf_word_data;   /* 256 bytes — one word at a time */
static id<MTLBuffer> buf_word_off;    /* single uint32_t */
static id<MTLBuffer> buf_word_len;    /* single uint16_t */
static id<MTLBuffer> buf_salt_data;   /* sized at set_salts */
static id<MTLBuffer> buf_salt_off;
static id<MTLBuffer> buf_salt_len;
static id<MTLBuffer> buf_params;      /* MetalParams struct */
static id<MTLBuffer> buf_dispatch_hits;
static id<MTLBuffer> buf_dispatch_hit_count;
static id<MTLBuffer> buf_dispatch_hexhashes; /* persistent word buffer */
static id<MTLBuffer> buf_dispatch_hexlens;   /* persistent hexlen buffer */
static id<MTLBuffer> buf_dummy;              /* 4-byte dummy for missing overflow */
static int           _dispatch_bufs_ready = 0;
static size_t        _salt_data_capacity = 0;
static int           _salts_count = 0;
static int           _max_hits = 0;

/* ---- Overflow hash table for GPU binary search fallback ---- */
static id<MTLBuffer> buf_overflow_keys;    /* sorted uint64_t keys */
static id<MTLBuffer> buf_overflow_hashes;  /* packed hash data (variable length) */
static id<MTLBuffer> buf_overflow_offsets; /* byte offset per entry */
static id<MTLBuffer> buf_overflow_lengths; /* byte length per entry */
static int           _overflow_count = 0;

/* ---- Double-buffer dispatch slots ---- */
#define GPU_NUM_SLOTS 2
#define GPU_SLOT_MAX_HITS 32768
static struct gpu_slot {
    id<MTLBuffer> buf_hexhashes;
    id<MTLBuffer> buf_hexlens;
    id<MTLBuffer> buf_params;
    id<MTLBuffer> buf_salt_data;
    id<MTLBuffer> buf_salt_off;
    id<MTLBuffer> buf_salt_len;
    id<MTLBuffer> buf_hits;
    id<MTLBuffer> buf_hit_count;
    id<MTLCommandBuffer> cmdbuf;
    id<MTLCommandQueue> queue;   /* per-slot queue for independent dispatch */
} gpu_slots[GPU_NUM_SLOTS];
static int gpu_slots_ready = 0;

/* ---- Params buffer ---- */
typedef struct {
    uint64_t compact_mask;
    uint32_t num_words;
    uint32_t num_salts;
    uint32_t salt_start;
    uint32_t max_probe;
    uint32_t hash_data_count;
    uint32_t max_hits;
    uint32_t overflow_count;
    uint32_t max_iter;
    uint32_t num_masks;
    uint32_t mask_start;
    uint32_t n_prepend;
    uint32_t n_append;
    uint32_t iter_count;   /* PHPBB3: uniform iteration count for this dispatch group */
} MetalParams;

static int _max_iter = 1;
static int _gpu_op = 0;  /* current op type for kernel selection */
static int mtl_max_dispatch = 256 * 1024 * 1024;  /* max work items per dispatch (256M) */
static MTLCompileOptions *mtl_compile_opts;  /* saved for deferred family compilation */

/* ---- Helper: wrap buffer with zero-copy if page-aligned, else copy ---- */
static id<MTLBuffer> wrap_buffer(void *ptr, size_t size) {
    if (size == 0) size = 4;
    /* Always copy for safety in PoC — zero-copy optimization later */
    return [mtl_device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
}

/* ---- Public API ---- */

int gpu_metal_init(void) {
    @autoreleasepool {
        mtl_device = MTLCreateSystemDefaultDevice();
        if (!mtl_device) {
            /* Fallback: MTLCreateSystemDefaultDevice() can return nil over SSH.
             * MTLCopyAllDevices() works in all contexts. */
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            if (all.count > 0)
                mtl_device = all[0];
        }
        if (!mtl_device) {
            fprintf(stderr, "Metal: no GPU device found\n");
            return -1;
        }

        mtl_queue = [mtl_device newCommandQueue];
        if (!mtl_queue) {
            fprintf(stderr, "Metal: failed to create command queue\n");
            return -1;
        }

        /* Compile only md5salt family at init (needed for probe/selftest).
         * Other families compiled on demand by gpu_metal_compile_families(). */
        NSError *error = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        mtl_compile_opts = opts;

        memset(mtl_fam_lib, 0, sizeof(mtl_fam_lib));
        memset(mtl_pipelines, 0, sizeof(mtl_pipelines));
        mtl_pipeline_iter = nil;

        /* Compile md5salt family */
        { NSString *src = [NSString stringWithFormat:@"%s%s",
                           metal_common_str, mtl_family_source[FAM_MD5SALT]];
          mtl_fam_lib[FAM_MD5SALT] = [mtl_device newLibraryWithSource:src
                                                                   options:opts
                                                                     error:&error];
        }
        if (!mtl_fam_lib[FAM_MD5SALT]) {
            fprintf(stderr, "Metal: md5salt family failed to compile\n");
            return -1;
        }
        id<MTLFunction> func = [mtl_fam_lib[FAM_MD5SALT] newFunctionWithName:@"md5salt_probe"];
        if (!func) {
            fprintf(stderr, "Metal: md5salt_probe not found\n");
            return -1;
        }
        mtl_pipeline = [mtl_device newComputePipelineStateWithFunction:func error:&error];
        if (!mtl_pipeline) {
            fprintf(stderr, "Metal: pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        id<MTLFunction> func2 = [mtl_fam_lib[FAM_MD5SALT] newFunctionWithName:@"md5salt_salts_only"];
        if (func2) {
            mtl_pipeline_salts = [mtl_device newComputePipelineStateWithFunction:func2 error:&error];
        }

        /* Register md5salt kernels */
        for (int k = 0; metal_kernel_map[k].name; k++) {
            if (metal_kernel_map[k].family != FAM_MD5SALT) continue;
            NSString *fname = [NSString stringWithUTF8String:metal_kernel_map[k].name];
            id<MTLFunction> fn = [mtl_fam_lib[FAM_MD5SALT] newFunctionWithName:fname];
            if (!fn) continue;
            id<MTLComputePipelineState> ps = [mtl_device newComputePipelineStateWithFunction:fn error:&error];
            if (!ps) continue;
            if (k == MTL_KERN_ITER_IDX)
                mtl_pipeline_iter = ps;
            for (int j = 0; metal_kernel_map[k].ops[j] >= 0; j++)
                mtl_pipelines[metal_kernel_map[k].ops[j]] = ps;
        }

        /* Allocate persistent hit buffers */
        buf_hits = [mtl_device newBufferWithLength:MAX_GPU_HITS * 2 * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];
        buf_hit_count = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];

        /* Allocate reusable per-dispatch buffers */
        buf_word_data = [mtl_device newBufferWithLength:256
                                                options:MTLResourceStorageModeShared];
        buf_word_off  = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        buf_word_len  = [mtl_device newBufferWithLength:sizeof(uint16_t)
                                                options:MTLResourceStorageModeShared];
        buf_params    = [mtl_device newBufferWithLength:sizeof(MetalParams)
                                                options:MTLResourceStorageModeShared];
        buf_dispatch_hit_count = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];

        fprintf(stderr, "Metal GPU: %s, max threads/group=%lu\n",
                [[mtl_device name] UTF8String],
                (unsigned long)[mtl_pipeline maxTotalThreadsPerThreadgroup]);

        mtl_ready = 1;
        return 0;
    }
}

void gpu_metal_shutdown(void) {
    @autoreleasepool {
        buf_compact_fp = nil;
        buf_compact_idx = nil;
        buf_hash_data = nil;
        buf_hash_data_off = nil;
        buf_hash_data_len = nil;
        buf_hits = nil;
        buf_hit_count = nil;
        mtl_pipeline = nil;
        mtl_queue = nil;
        mtl_device = nil;
        mtl_ready = 0;
    }
}

int gpu_metal_available(void) {
    return mtl_ready;
}

/* Compile additional kernel families on demand. Called from main() after
 * hash types are known. fam_mask is a bitmask of MTL_FAM_* values. */
void gpu_metal_compile_families(unsigned int fam_mask) {
    if (!mtl_ready || !mtl_compile_opts) return;
    @autoreleasepool {
        NSError *error = nil;
        for (int f = 0; f < FAM_COUNT; f++) {
            if (mtl_fam_lib[f]) continue;  /* already compiled */
            if (!(fam_mask & (1u << f))) continue;  /* not requested */
            NSString *src = [NSString stringWithFormat:@"%s%s",
                             metal_common_str, mtl_family_source[f]];
            mtl_fam_lib[f] = [mtl_device newLibraryWithSource:src
                                                       options:mtl_compile_opts
                                                         error:&error];
            if (!mtl_fam_lib[f]) {
                fprintf(stderr, "Metal: family %d compile error: %s\n", f,
                        [[error localizedDescription] UTF8String]);
                continue;
            }
            /* Register kernels from this family */
            for (int k = 0; metal_kernel_map[k].name; k++) {
                if (metal_kernel_map[k].family != f) continue;
                NSString *fname = [NSString stringWithUTF8String:metal_kernel_map[k].name];
                id<MTLFunction> fn = [mtl_fam_lib[f] newFunctionWithName:fname];
                if (!fn) continue;
                id<MTLComputePipelineState> ps = [mtl_device newComputePipelineStateWithFunction:fn error:&error];
                if (!ps) continue;
                if (k == MTL_KERN_ITER_IDX)
                    mtl_pipeline_iter = ps;
                for (int j = 0; metal_kernel_map[k].ops[j] >= 0; j++)
                    mtl_pipelines[metal_kernel_map[k].ops[j]] = ps;
            }
        }
    }
}

int gpu_metal_set_compact_table(
    uint32_t *compact_fp,
    uint32_t *compact_idx,
    uint64_t compact_size,
    uint64_t compact_mask,
    unsigned char *hash_data_buf,
    size_t hash_data_buf_size,
    size_t *hash_data_off,
    size_t hash_data_count,
    unsigned short *hash_data_len)
{
    @autoreleasepool {
        if (!mtl_ready) return -1;

        _compact_mask = compact_mask;
        _hash_data_count = (uint32_t)hash_data_count;

        buf_compact_fp  = wrap_buffer(compact_fp,  compact_size * sizeof(uint32_t));
        buf_compact_idx = wrap_buffer(compact_idx, compact_size * sizeof(uint32_t));

        if (hash_data_buf_size > 0)
            buf_hash_data = wrap_buffer(hash_data_buf, hash_data_buf_size);
        else
            buf_hash_data = [mtl_device newBufferWithLength:4 options:MTLResourceStorageModeShared];

        buf_hash_data_off = wrap_buffer(hash_data_off, hash_data_count * sizeof(size_t));
        buf_hash_data_len = wrap_buffer(hash_data_len, hash_data_count * sizeof(unsigned short));

        if (!buf_compact_fp || !buf_compact_idx || !buf_hash_data ||
            !buf_hash_data_off || !buf_hash_data_len) {
            fprintf(stderr, "Metal: failed to create compact table buffers\n");
            mtl_ready = 0;
            return -1;
        }

        fprintf(stderr, "Metal GPU: compact table registered (%llu slots, %zu hashes)\n",
                compact_size, hash_data_count);
        return 0;
    }
}

int gpu_metal_set_salts(
    const char *salts,
    const uint32_t *salt_offsets,
    const uint16_t *salt_lens,
    int num_salts)
{
    @autoreleasepool {
        if (!mtl_ready || num_salts <= 0) return -1;
        size_t salts_size = 0;
        for (int i = 0; i < num_salts; i++)
            salts_size += salt_lens[i];
        if (!salts_size) salts_size = 1;

        /* Reallocate only if capacity is insufficient */
        if (!buf_salt_data || salts_size > _salt_data_capacity) {
            _salt_data_capacity = salts_size + salts_size / 4; /* 25% headroom */
            buf_salt_data = [mtl_device newBufferWithLength:_salt_data_capacity
                                                    options:MTLResourceStorageModeShared];
        }
        if (!buf_salt_off || num_salts * sizeof(uint32_t) > [buf_salt_off length]) {
            buf_salt_off = [mtl_device newBufferWithLength:num_salts * sizeof(uint32_t) * 2
                                                   options:MTLResourceStorageModeShared];
        }
        if (!buf_salt_len || num_salts * sizeof(uint16_t) > [buf_salt_len length]) {
            buf_salt_len = [mtl_device newBufferWithLength:num_salts * sizeof(uint16_t) * 2
                                                   options:MTLResourceStorageModeShared];
        }

        /* Copy data into persistent buffers */
        memcpy([buf_salt_data contents], salts, salts_size);
        memcpy([buf_salt_off contents], salt_offsets, num_salts * sizeof(uint32_t));
        memcpy([buf_salt_len contents], salt_lens, num_salts * sizeof(uint16_t));
        _salts_count = num_salts;

        /* Size hit buffer for chunk dispatch — 5 uint32s per hit */
        #define GPU_MAX_HITS 32768
        _max_hits = GPU_MAX_HITS;
        buf_dispatch_hits = [mtl_device newBufferWithLength:_max_hits * 7 * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
        *(uint32_t *)[buf_dispatch_hit_count contents] = 0;

        /* Allocate persistent word buffers (reused across dispatches) */
        if (!buf_dispatch_hexhashes)
            buf_dispatch_hexhashes = [mtl_device newBufferWithLength:GPUBATCH_MAX * 256
                                                             options:MTLResourceStorageModeShared];
        if (!buf_dispatch_hexlens)
            buf_dispatch_hexlens = [mtl_device newBufferWithLength:GPUBATCH_MAX * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
        if (!buf_dummy)
            buf_dummy = [mtl_device newBufferWithLength:4
                                                options:MTLResourceStorageModeShared];

        _dispatch_bufs_ready = 1;
        return 0;
    }
}

uint32_t *gpu_metal_probe_salts(
    const char *hexhash,
    int hexlen,
    int *nhits_out)
{
    @autoreleasepool {
        *nhits_out = 0;
        if (!mtl_ready || !_dispatch_bufs_ready || !mtl_pipeline_salts) return NULL;
        if (hexlen <= 0 || hexlen > 255 || _salts_count <= 0) return NULL;

        /* Copy word into persistent buffer */
        memcpy([buf_word_data contents], hexhash, hexlen);
        *(uint32_t *)[buf_word_off contents] = 0;
        *(uint16_t *)[buf_word_len contents] = (uint16_t)hexlen;

        /* Params */
        MetalParams *p = (MetalParams *)[buf_params contents];
        p->compact_mask = _compact_mask;
        p->num_words = 1;
        p->num_salts = _salts_count;
        p->max_probe = 256;
        p->hash_data_count = _hash_data_count;
        p->max_hits = _max_hits;

        /* Zero hit counter */
        *(uint32_t *)[buf_dispatch_hit_count contents] = 0;

        /* Encode and dispatch */
        id<MTLCommandBuffer> cmdbuf = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        [enc setComputePipelineState:mtl_pipeline_salts];
        [enc setBuffer:buf_word_data         offset:0 atIndex:0];
        [enc setBuffer:buf_word_off          offset:0 atIndex:1];
        [enc setBuffer:buf_word_len          offset:0 atIndex:2];
        [enc setBuffer:buf_salt_data         offset:0 atIndex:3];
        [enc setBuffer:buf_salt_off          offset:0 atIndex:4];
        [enc setBuffer:buf_salt_len          offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp        offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx       offset:0 atIndex:7];
        [enc setBuffer:buf_params            offset:0 atIndex:8];
        [enc setBuffer:buf_hash_data         offset:0 atIndex:9];
        [enc setBuffer:buf_hash_data_off     offset:0 atIndex:10];
        [enc setBuffer:buf_hash_data_len     offset:0 atIndex:11];
        [enc setBuffer:buf_dispatch_hits     offset:0 atIndex:12];
        [enc setBuffer:buf_dispatch_hit_count offset:0 atIndex:13];

        NSUInteger tpg = [mtl_pipeline_salts maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        MTLSize gridSize = MTLSizeMake(_salts_count, 1, 1);
        MTLSize groupSize = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        /* Return raw hit count (may exceed _max_hits) and pointer to buffer */
        uint32_t raw_nhits = *(uint32_t *)[buf_dispatch_hit_count contents];
        int stored = (int)raw_nhits;
        if (stored > _max_hits) stored = _max_hits;
        *nhits_out = (int)raw_nhits; /* caller sees total, including overflow */
        return (uint32_t *)[buf_dispatch_hits contents];
    }
}

int gpu_metal_set_overflow(
    const uint64_t *keys,
    const unsigned char *hashes,
    const uint32_t *offsets,
    const uint16_t *lengths,
    int count)
{
    @autoreleasepool {
        if (!mtl_ready) return -1;
        _overflow_count = count;
        if (count <= 0) {
            buf_overflow_keys = nil;
            buf_overflow_hashes = nil;
            buf_overflow_offsets = nil;
            buf_overflow_lengths = nil;
            return 0;
        }
        buf_overflow_keys = [mtl_device newBufferWithBytes:keys
                                                    length:count * sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
        /* Compute total hash data size from offsets + lengths */
        size_t total = offsets[count - 1] + lengths[count - 1];
        buf_overflow_hashes = [mtl_device newBufferWithBytes:hashes
                                                      length:total
                                                     options:MTLResourceStorageModeShared];
        buf_overflow_offsets = [mtl_device newBufferWithBytes:offsets
                                                       length:count * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
        buf_overflow_lengths = [mtl_device newBufferWithBytes:lengths
                                                       length:count * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        fprintf(stderr, "Metal GPU: %d overflow entries loaded for GPU fallback\n", count);
        return 0;
    }
}

void gpu_metal_set_max_iter(int max_iter) {
    _max_iter = (max_iter < 1) ? 1 : max_iter;
}

static int _iter_count = 0;  /* PHPBB3: uniform iteration count for current dispatch group */

void gpu_metal_set_iter_count(int count) {
    _iter_count = count;
}

void gpu_metal_set_op(int op) {
    _gpu_op = op;
}

uint32_t *gpu_metal_dispatch_batch(
    const char *hexhashes,
    const uint16_t *hexlens,
    int num_words,
    int *nhits_out)
{
    @autoreleasepool {
        *nhits_out = 0;
        if (!mtl_ready || !_dispatch_bufs_ready || !mtl_pipelines[JOB_MD5SALT]) return NULL;
        if (num_words <= 0 || _salts_count <= 0) return NULL;

        /* Copy word data into persistent buffers */
        size_t words_size = (size_t)num_words * 256;
        memcpy([buf_dispatch_hexhashes contents], hexhashes, words_size);
        memcpy([buf_dispatch_hexlens contents], hexlens, num_words * sizeof(uint16_t));

        /* Params */
        MetalParams params;
        params.compact_mask = _compact_mask;
        params.num_words = num_words;
        params.num_salts = _salts_count;
        params.salt_start = 0;
        params.max_probe = 256;
        params.hash_data_count = _hash_data_count;
        params.max_hits = _max_hits;
        params.overflow_count = _overflow_count;
        params.max_iter = _max_iter;
        params.iter_count = _iter_count;

        /* Salt chunking — avoid overwhelming the GPU with huge dispatches */
        int total_salts = _salts_count;
        int salt_chunk = total_salts;
        if (mtl_max_dispatch > 0 && num_words > 0 && total_salts > 0) {
            salt_chunk = mtl_max_dispatch / num_words;
            if (salt_chunk < 1024) salt_chunk = 1024;
            if (salt_chunk > total_salts) salt_chunk = total_salts;
        }
        int num_chunks = (total_salts + salt_chunk - 1) / salt_chunk;

        /* Zero hit counter */
        *(uint32_t *)[buf_dispatch_hit_count contents] = 0;

        /* Encode and dispatch — use iter kernel when max_iter > 1 */
        id<MTLCommandBuffer> cmdbuf = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        id<MTLComputePipelineState> pipeline =
            (_max_iter > 1 && mtl_pipeline_iter) ? mtl_pipeline_iter :
            (_gpu_op >= 0 && _gpu_op < JOB_DONE && mtl_pipelines[_gpu_op]) ? mtl_pipelines[_gpu_op] : nil;
        if (!pipeline) { [cmdbuf release]; return NULL; }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_dispatch_hexhashes offset:0 atIndex:0];
        [enc setBuffer:buf_dispatch_hexlens  offset:0 atIndex:1];
        [enc setBuffer:buf_dispatch_hexlens  offset:0 atIndex:2]; /* unused slot */
        [enc setBuffer:buf_salt_data     offset:0 atIndex:3];
        [enc setBuffer:buf_salt_off      offset:0 atIndex:4];
        [enc setBuffer:buf_salt_len      offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp    offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx   offset:0 atIndex:7];
        [enc setBytes:&params length:sizeof(params) atIndex:8];
        [enc setBuffer:buf_hash_data     offset:0 atIndex:9];
        [enc setBuffer:buf_hash_data_off offset:0 atIndex:10];
        [enc setBuffer:buf_hash_data_len offset:0 atIndex:11];
        [enc setBuffer:buf_dispatch_hits offset:0 atIndex:12];
        [enc setBuffer:buf_dispatch_hit_count offset:0 atIndex:13];
        /* Overflow buffers — use persistent dummy if no overflow */
        [enc setBuffer:(buf_overflow_keys    ? buf_overflow_keys    : buf_dummy) offset:0 atIndex:14];
        [enc setBuffer:(buf_overflow_hashes  ? buf_overflow_hashes  : buf_dummy) offset:0 atIndex:15];
        [enc setBuffer:(buf_overflow_offsets ? buf_overflow_offsets : buf_dummy) offset:0 atIndex:16];
        [enc setBuffer:(buf_overflow_lengths ? buf_overflow_lengths : buf_dummy) offset:0 atIndex:17];

        NSUInteger tpg = [pipeline maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        MTLSize groupSize = MTLSizeMake(tpg, 1, 1);

        /* Salt chunking: encode chunks into command buffer using setBytes for params */
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            if (num_chunks > 1) {
                params.salt_start = chunk * salt_chunk;
                params.num_salts = total_salts - params.salt_start;
                if ((int)params.num_salts > salt_chunk) params.num_salts = salt_chunk;
                /* Use setBytes to inline params — avoids modifying shared buffer mid-encode */
                [enc setBytes:&params length:sizeof(params) atIndex:8];
            }

            uint64_t chunk_threads = (uint64_t)num_words * params.num_salts;
            MTLSize gridSize = MTLSizeMake(chunk_threads, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        }

        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        /* Read back — return raw count and pointer */
        uint32_t raw_nhits = *(uint32_t *)[buf_dispatch_hit_count contents];
        *nhits_out = (int)raw_nhits;

        cmdbuf = nil; enc = nil;

        return (uint32_t *)[buf_dispatch_hits contents];
    }
}

/* ---- Double-buffer slot API ---- */

int gpu_metal_init_slots(int max_salt_count, int max_salt_bytes) {
    @autoreleasepool {
        if (!mtl_ready) return -1;
        if (max_salt_count < 1024) max_salt_count = 1024;
        if (max_salt_bytes < 8192) max_salt_bytes = 8192;

        for (int i = 0; i < GPU_NUM_SLOTS; i++) {
            gpu_slots[i].buf_hexhashes = [mtl_device newBufferWithLength:GPUBATCH_MAX * 256
                                                                 options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_hexlens = [mtl_device newBufferWithLength:GPUBATCH_MAX * sizeof(uint16_t)
                                                                options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_params = [mtl_device newBufferWithLength:sizeof(MetalParams)
                                                              options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_salt_data = [mtl_device newBufferWithLength:max_salt_bytes + 4096
                                                                 options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_salt_off = [mtl_device newBufferWithLength:max_salt_count * sizeof(uint32_t)
                                                                options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_salt_len = [mtl_device newBufferWithLength:max_salt_count * sizeof(uint16_t)
                                                                options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_hits = [mtl_device newBufferWithLength:GPU_SLOT_MAX_HITS * 7 * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
            gpu_slots[i].buf_hit_count = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
            gpu_slots[i].cmdbuf = nil;
            gpu_slots[i].queue = [mtl_device newCommandQueue];
            if (!gpu_slots[i].buf_hexhashes || !gpu_slots[i].buf_salt_data ||
                !gpu_slots[i].buf_hits || !gpu_slots[i].queue) return -1;
        }
        gpu_slots_ready = 1;
        fprintf(stderr, "Metal GPU: %d double-buffer slots initialized (%d max salts, %d max bytes)\n",
                GPU_NUM_SLOTS, max_salt_count, max_salt_bytes);
        return 0;
    }
}

int gpu_metal_submit_slot(int slot,
    const char *hexhashes, const uint16_t *hexlens, int num_words,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts)
{
    {   /* no @autoreleasepool — cmdbuf must survive until wait_slot */
        if (!gpu_slots_ready || !mtl_pipelines[JOB_MD5SALT]) return -1;
        if (slot < 0 || slot >= GPU_NUM_SLOTS) return -1;
        if (num_words <= 0 || num_words > GPUBATCH_MAX) return -1;
        if (num_salts <= 0) return -1;

        struct gpu_slot *s = &gpu_slots[slot];

        memcpy([s->buf_hexhashes contents], hexhashes, (size_t)num_words * 256);
        memcpy([s->buf_hexlens contents], hexlens, num_words * sizeof(uint16_t));

        size_t salt_bytes = 0;
        if (num_salts > 0)
            salt_bytes = salt_offsets[num_salts - 1] + salt_lens[num_salts - 1];
        memcpy([s->buf_salt_data contents], salts, salt_bytes);
        memcpy([s->buf_salt_off contents], salt_offsets, num_salts * sizeof(uint32_t));
        memcpy([s->buf_salt_len contents], salt_lens, num_salts * sizeof(uint16_t));

        MetalParams *p = (MetalParams *)[s->buf_params contents];
        p->compact_mask = _compact_mask;
        p->num_words = num_words;
        p->num_salts = num_salts;
        p->salt_start = 0;
        p->max_probe = 256;
        p->hash_data_count = _hash_data_count;
        p->max_hits = GPU_SLOT_MAX_HITS;
        p->overflow_count = _overflow_count;
        p->max_iter = _max_iter;

        *(uint32_t *)[s->buf_hit_count contents] = 0;

        id<MTLCommandBuffer> cmdbuf = [s->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        { id<MTLComputePipelineState> ps =
            (_max_iter > 1 && mtl_pipeline_iter) ? mtl_pipeline_iter :
            (_gpu_op >= 0 && _gpu_op < JOB_DONE && mtl_pipelines[_gpu_op]) ? mtl_pipelines[_gpu_op] : nil;
          if (!ps) { [cmdbuf release]; return -1; }
          [enc setComputePipelineState:ps];
        }
        [enc setBuffer:s->buf_hexhashes  offset:0 atIndex:0];
        [enc setBuffer:s->buf_hexlens    offset:0 atIndex:1];
        [enc setBuffer:s->buf_hexlens    offset:0 atIndex:2];
        [enc setBuffer:s->buf_salt_data  offset:0 atIndex:3];
        [enc setBuffer:s->buf_salt_off   offset:0 atIndex:4];
        [enc setBuffer:s->buf_salt_len   offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp    offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx   offset:0 atIndex:7];
        [enc setBuffer:s->buf_params     offset:0 atIndex:8];
        [enc setBuffer:buf_hash_data     offset:0 atIndex:9];
        [enc setBuffer:buf_hash_data_off offset:0 atIndex:10];
        [enc setBuffer:buf_hash_data_len offset:0 atIndex:11];
        [enc setBuffer:s->buf_hits       offset:0 atIndex:12];
        [enc setBuffer:s->buf_hit_count  offset:0 atIndex:13];
        if (buf_overflow_keys) {
            [enc setBuffer:buf_overflow_keys    offset:0 atIndex:14];
            [enc setBuffer:buf_overflow_hashes  offset:0 atIndex:15];
            [enc setBuffer:buf_overflow_offsets offset:0 atIndex:16];
            [enc setBuffer:buf_overflow_lengths offset:0 atIndex:17];
        } else {
            [enc setBuffer:buf_dummy offset:0 atIndex:14];
            [enc setBuffer:buf_dummy offset:0 atIndex:15];
            [enc setBuffer:buf_dummy offset:0 atIndex:16];
            [enc setBuffer:buf_dummy offset:0 atIndex:17];
        }

        uint64_t total_threads = (uint64_t)num_words * num_salts;
        NSUInteger tpg = [mtl_pipelines[JOB_MD5SALT] maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
        MTLSize groupSize = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];

        [cmdbuf commit];
        s->cmdbuf = cmdbuf;
        return 0;
    }
}

uint32_t *gpu_metal_wait_slot(int slot, int *nhits_out) {
    @autoreleasepool {
        *nhits_out = 0;
        if (slot < 0 || slot >= GPU_NUM_SLOTS) return NULL;
        struct gpu_slot *s = &gpu_slots[slot];
        if (!s->cmdbuf) return NULL;

        [s->cmdbuf waitUntilCompleted];
        s->cmdbuf = nil;

        uint32_t raw_nhits = *(uint32_t *)[s->buf_hit_count contents];
        *nhits_out = (int)raw_nhits;
        return (uint32_t *)[s->buf_hits contents];
    }
}

int gpu_metal_dispatch(
    const char *words,
    const uint32_t *word_offsets,
    const uint16_t *word_lens,
    int num_words,
    const char *salts,
    const uint32_t *salt_offsets,
    const uint16_t *salt_lens,
    int num_salts,
    uint32_t *hits_out,
    int max_hits)
{
    @autoreleasepool {
        if (!mtl_ready || num_words <= 0 || num_salts <= 0) return 0;

        uint64_t total_threads = (uint64_t)num_words * num_salts;

        /* Calculate packed buffer sizes */
        size_t words_size = 0;
        for (int i = 0; i < num_words; i++)
            words_size += word_lens[i];
        size_t salts_size = 0;
        for (int i = 0; i < num_salts; i++)
            salts_size += salt_lens[i];

        /* Create transient buffers */
        id<MTLBuffer> b_words = [mtl_device newBufferWithBytes:words
                                                        length:words_size ?: 1
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_woff  = [mtl_device newBufferWithBytes:word_offsets
                                                        length:num_words * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_wlen  = [mtl_device newBufferWithBytes:word_lens
                                                        length:num_words * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_salts = [mtl_device newBufferWithBytes:salts
                                                        length:salts_size ?: 1
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_soff  = [mtl_device newBufferWithBytes:salt_offsets
                                                        length:num_salts * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_slen  = [mtl_device newBufferWithBytes:salt_lens
                                                        length:num_salts * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];

        /* Params */
        /* Read compact_mask from the fp buffer metadata — stored during set_compact_table */
        MetalParams params;
        params.compact_mask = 0; /* will be set below */
        params.num_words = num_words;
        params.num_salts = num_salts;
        params.max_probe = 256;
        params.hash_data_count = 0;

        params.compact_mask = _compact_mask;
        params.hash_data_count = _hash_data_count;

        id<MTLBuffer> b_params_buf = [mtl_device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];
        int hit_capacity = (max_hits < 1024) ? max_hits : 1024;
        id<MTLBuffer> local_hit_count = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                                options:MTLResourceStorageModeShared];
        *(uint32_t *)[local_hit_count contents] = 0;
        id<MTLBuffer> local_hits = [mtl_device newBufferWithLength:hit_capacity * 2 * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];

        /* Encode and dispatch */
        id<MTLCommandBuffer> cmdbuf = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        /* Use salts-only kernel when available and num_words==1 */
        id<MTLComputePipelineState> pipeline =
            (num_words == 1 && mtl_pipeline_salts) ? mtl_pipeline_salts : mtl_pipeline;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:b_words         offset:0 atIndex:0];
        [enc setBuffer:b_woff          offset:0 atIndex:1];
        [enc setBuffer:b_wlen          offset:0 atIndex:2];
        [enc setBuffer:b_salts         offset:0 atIndex:3];
        [enc setBuffer:b_soff          offset:0 atIndex:4];
        [enc setBuffer:b_slen          offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp  offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx offset:0 atIndex:7];
        [enc setBuffer:b_params_buf    offset:0 atIndex:8];
        [enc setBuffer:buf_hash_data   offset:0 atIndex:9];
        [enc setBuffer:buf_hash_data_off offset:0 atIndex:10];
        [enc setBuffer:buf_hash_data_len offset:0 atIndex:11];
        [enc setBuffer:local_hits      offset:0 atIndex:12];
        [enc setBuffer:local_hit_count offset:0 atIndex:13];

        /* For salts-only kernel, grid = num_salts; for full kernel, grid = words*salts */
        if (num_words == 1 && mtl_pipeline_salts)
            total_threads = num_salts;

        NSUInteger tpg = [pipeline maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
        MTLSize groupSize = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        /* Read back hits */
        uint32_t nhits = *(uint32_t *)[local_hit_count contents];
        if (nhits > MAX_GPU_HITS) nhits = MAX_GPU_HITS;
        if ((int)nhits > max_hits) nhits = max_hits;

        uint32_t *hit_data = (uint32_t *)[local_hits contents];
        memcpy(hits_out, hit_data, nhits * 2 * sizeof(uint32_t));

        /* Release transient buffers */
        cmdbuf = nil; enc = nil;
        b_words = nil; b_woff = nil; b_wlen = nil;
        b_salts = nil; b_soff = nil; b_slen = nil;
        b_params_buf = nil; local_hits = nil; local_hit_count = nil;

        return (int)nhits;
    }
}

#endif /* __APPLE__ && METAL_GPU */
