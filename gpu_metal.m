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
/* mach-o/dyld.h no longer needed — metallib is embedded, not loaded from file */

/* ---- Metal per-family kernel sources (auto-generated from metal_*.metal) ---- */
#include "gpu/metal_common_str.h"
#include "gpu/metal_md5salt_str.h"
#include "gpu/metal_md5saltpass_str.h"
#include "gpu/metal_md5_md5saltmd5pass_str.h"
#include "gpu/metal_sha256_str.h"
#include "gpu/metal_phpbb3_str.h"
#include "gpu/metal_descrypt_str.h"
#include "gpu/metal_md5unsalted_str.h"
#include "gpu/metal_md4unsalted_str.h"
#include "gpu/metal_sha1unsalted_str.h"
#include "gpu/metal_sha256unsalted_str.h"
#include "gpu/metal_sha512unsalted_str.h"
#include "gpu/metal_wrlunsalted_str.h"
#include "gpu/metal_md6256unsalted_str.h"
#include "gpu/metal_keccakunsalted_str.h"
/* HMAC-SHA256 kernels are in metal_sha256.metal (FAM_SHA256) */
#include "gpu/metal_hmac_sha512_str.h"
#include "gpu/metal_mysql3unsalted_str.h"
#include "gpu/metal_hmac_rmd160_str.h"
#include "gpu/metal_hmac_rmd320_str.h"
#include "gpu/metal_hmac_blake2s_str.h"
#include "gpu/metal_streebog_str.h"
/* metal_sha512crypt_str.h removed in SHA512CRYPTMD5 carrier ship
 * (2026-05-08): the OpenCL slab kernel sha512crypt_batch is retired and
 * its Metal sibling follows. Both JOB_SHA512CRYPT (Phase 3) and JOB_-
 * SHA512CRYPTMD5 (Phase 4) route through the unified template path via
 * the SHACRYPT shared core. RCS history retains the source at
 * gpu/RCS/metal_sha512crypt.metal,v. */
#include "gpu/metal_sha256crypt_str.h"
#include "gpu/metal_rmd160unsalted_str.h"
#include "gpu/metal_blake2s256unsalted_str.h"
/* metal_bcrypt_str.h retired 2026-05-09 (task #218): Phase 6 OpenCL ship
 * (2026-05-08) severed the Metal BCRYPT dispatch path by removing the
 * procjob gpu_try_pack call under #ifdef GPU_ENABLED (the chokepoint
 * admit + salt-snapshot precondition + pack-site clamp that replaced it
 * are #ifdef OPENCL_GPU only — Metal builds never reach them). Metal
 * BCRYPT has been CPU-only since that ship; this commit cleans the dark
 * kernel from working tree. RCS history retained at gpu/RCS/. Future
 * OpenCL-to-Metal template-path conversion is on the roadmap and would
 * re-add Metal BCRYPT support via a Metal template scaffold + algo_mode
 * dispatch (multi-month parallel-implementation; not Phase 6 followup). */
#include "gpu/metal_sha1_str.h"
#include "gpu/metal_md5crypt_str.h"
/* Packed-password dispatch kernels (GPU rule path) */
#include "gpu/metal_md5_packed_str.h"
#include "gpu/metal_md4_packed_str.h"
#include "gpu/metal_sha1_packed_str.h"
#include "gpu/metal_sha256_packed_str.h"
#include "gpu/metal_sha512_packed_str.h"

/* Family IDs from gpujob.h FAM_* enum — Metal uses a subset */
static const char *mtl_family_source[FAM_COUNT] = {
    [FAM_MD5SALT]            = metal_md5salt_str,
    [FAM_MD5SALTPASS]        = metal_md5saltpass_str,
    [FAM_MD5_MD5SALTMD5PASS] = metal_md5_md5saltmd5pass_str,
    [FAM_SHA256]             = metal_sha256_str,
    [FAM_PHPBB3]             = metal_phpbb3_str,
    [FAM_DESCRYPT]           = metal_descrypt_str,
    [FAM_MD5UNSALTED]        = metal_md5unsalted_str,
    [FAM_MD4UNSALTED]        = metal_md4unsalted_str,
    [FAM_SHA1UNSALTED]       = metal_sha1unsalted_str,
    [FAM_SHA256UNSALTED]     = metal_sha256unsalted_str,
    [FAM_SHA512UNSALTED]     = metal_sha512unsalted_str,
    [FAM_WRLUNSALTED]        = metal_wrlunsalted_str,
    [FAM_MD6256UNSALTED]     = metal_md6256unsalted_str,
    [FAM_KECCAKUNSALTED]     = metal_keccakunsalted_str,
    [FAM_HMAC_SHA512]        = metal_hmac_sha512_str,
    [FAM_MYSQL3UNSALTED]     = metal_mysql3unsalted_str,
    [FAM_HMAC_RMD160]       = metal_hmac_rmd160_str,
    [FAM_HMAC_RMD320]       = metal_hmac_rmd320_str,
    [FAM_HMAC_BLAKE2S]      = metal_hmac_blake2s_str,
    [FAM_STREEBOG]          = metal_streebog_str,
    /* [FAM_SHA512CRYPT] retired in SHA512CRYPTMD5 carrier ship
     * (2026-05-08): both Phase 3 SHA512CRYPT and Phase 4 SHA512CRYPTMD5
     * now route through the unified template path. Implicit NULL
     * designated initializer. */
    [FAM_SHA256CRYPT]       = metal_sha256crypt_str,
    [FAM_RMD160UNSALTED]    = metal_rmd160unsalted_str,
    [FAM_BLAKE2S256UNSALTED] = metal_blake2s256unsalted_str,
    /* [FAM_BCRYPT] retired 2026-05-09 (task #218); implicit NULL designated
     * initializer. See retirement comment at #include block above. */
    [FAM_MD5CRYPT]           = metal_md5crypt_str,
    [FAM_SHA1]               = metal_sha1_str,
};

/* ---- Metal state ---- */
static id<MTLDevice>              mtl_device;
static id<MTLCommandQueue>        mtl_queue;
static id<MTLComputePipelineState> mtl_pipeline;       /* probe kernel */
static id<MTLComputePipelineState> mtl_pipeline_salts;  /* salts-only kernel */
static id<MTLComputePipelineState> mtl_pipeline_iter;   /* salted iteration override */
static id<MTLComputePipelineState> mtl_pipelines[JOB_DONE]; /* per-op dispatch kernels */
static id<MTLLibrary>             mtl_fam_lib[FAM_COUNT]; /* per-family compiled libraries */

/* Per-family packed kernel pipelines — compiled lazily on first packed dispatch.
 * Keyed by FAM_MD5UNSALTED/FAM_SHA1UNSALTED/FAM_SHA256UNSALTED/FAM_MD4UNSALTED/
 * FAM_SHA512UNSALTED. All packed kernels baked into the embedded metallib. */
static id<MTLComputePipelineState> mtl_pipeline_packed[FAM_COUNT];
static int                         mtl_pipeline_packed_tried[FAM_COUNT];

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
    {"md4_unsalted_batch",       {JOB_MD4, -1}, FAM_MD4UNSALTED},
    {"md4utf16_unsalted_batch",  {JOB_NTLMH, -1}, FAM_MD4UNSALTED},
    {"sha1_unsalted_batch",      {JOB_SHA1, -1}, FAM_SHA1UNSALTED},
    {"sha256_unsalted_batch",    {JOB_SHA256, -1}, FAM_SHA256UNSALTED},
    {"sha224_unsalted_batch",    {JOB_SHA224, -1}, FAM_SHA256UNSALTED},
    {"sha512_unsalted_batch",    {JOB_SHA512, -1}, FAM_SHA512UNSALTED},
    {"sha384_unsalted_batch",    {JOB_SHA384, -1}, FAM_SHA512UNSALTED},
    {"wrl_unsalted_batch",       {JOB_WRL, -1}, FAM_WRLUNSALTED},
    {"md6_256_unsalted_batch",   {JOB_MD6256, -1}, FAM_MD6256UNSALTED},
    {"keccak224_unsalted_batch", {JOB_KECCAK224, -1}, FAM_KECCAKUNSALTED},
    {"keccak256_unsalted_batch", {JOB_KECCAK256, -1}, FAM_KECCAKUNSALTED},
    {"keccak384_unsalted_batch", {JOB_KECCAK384, -1}, FAM_KECCAKUNSALTED},
    {"keccak512_unsalted_batch", {JOB_KECCAK512, -1}, FAM_KECCAKUNSALTED},
    {"sha3_224_unsalted_batch",  {JOB_SHA3_224, -1}, FAM_KECCAKUNSALTED},
    {"sha3_256_unsalted_batch",  {JOB_SHA3_256, -1}, FAM_KECCAKUNSALTED},
    {"sha3_384_unsalted_batch",  {JOB_SHA3_384, -1}, FAM_KECCAKUNSALTED},
    {"sha3_512_unsalted_batch",  {JOB_SHA3_512, -1}, FAM_KECCAKUNSALTED},
    {"hmac_sha256_ksalt_batch",  {JOB_HMAC_SHA256, -1}, FAM_SHA256},
    {"hmac_sha256_kpass_batch",  {JOB_HMAC_SHA256_KPASS, -1}, FAM_SHA256},
    {"hmac_sha224_ksalt_batch",  {JOB_HMAC_SHA224, -1}, FAM_SHA256},
    {"hmac_sha224_kpass_batch",  {JOB_HMAC_SHA224_KPASS, -1}, FAM_SHA256},
    {"hmac_md5_ksalt_batch",    {JOB_HMAC_MD5, -1}, FAM_MD5SALT},
    {"hmac_md5_kpass_batch",    {JOB_HMAC_MD5_KPASS, -1}, FAM_MD5SALT},
    {"sha1passsalt_batch",      {JOB_SHA1PASSSALT, -1}, FAM_SHA1},
    {"sha1saltpass_batch",      {JOB_SHA1SALTPASS, -1}, FAM_SHA1},
    {"sha1dru_batch",           {JOB_SHA1DRU, -1}, FAM_SHA1},
    {"hmac_sha1_ksalt_batch",   {JOB_HMAC_SHA1, -1}, FAM_SHA1},
    {"hmac_sha1_kpass_batch",   {JOB_HMAC_SHA1_KPASS, -1}, FAM_SHA1},
    {"sha512passsalt_batch",    {JOB_SHA512PASSSALT, -1}, FAM_HMAC_SHA512},
    {"sha512saltpass_batch",   {JOB_SHA512SALTPASS, -1}, FAM_HMAC_SHA512},
    {"hmac_sha512_ksalt_batch", {JOB_HMAC_SHA512, -1}, FAM_HMAC_SHA512},
    {"hmac_sha512_kpass_batch", {JOB_HMAC_SHA512_KPASS, -1}, FAM_HMAC_SHA512},
    {"hmac_sha384_ksalt_batch", {JOB_HMAC_SHA384, -1}, FAM_HMAC_SHA512},
    {"hmac_sha384_kpass_batch", {JOB_HMAC_SHA384_KPASS, -1}, FAM_HMAC_SHA512},
    {"md5_unsalted_batch",     {JOB_MD5RAW, JOB_MD5UC, -1}, FAM_MD5UNSALTED},
    {"sha1_unsalted_batch",    {JOB_SHA1RAW, -1}, FAM_SHA1UNSALTED},
    {"sql5_unsalted_batch",    {JOB_SQL5, -1}, FAM_SHA1UNSALTED},
    {"sha256_unsalted_batch",  {JOB_SHA256RAW, -1}, FAM_SHA256UNSALTED},
    {"sha384_unsalted_batch",  {JOB_SHA384RAW, -1}, FAM_SHA512UNSALTED},
    {"sha512_unsalted_batch",  {JOB_SHA512RAW, -1}, FAM_SHA512UNSALTED},
    {"mysql3_unsalted_batch",  {JOB_MYSQL3, -1}, FAM_MYSQL3UNSALTED},
    {"hmac_rmd160_ksalt_batch", {JOB_HMAC_RMD160, -1}, FAM_HMAC_RMD160},
    {"hmac_rmd160_kpass_batch", {JOB_HMAC_RMD160_KPASS, -1}, FAM_HMAC_RMD160},
    {"hmac_rmd320_ksalt_batch", {JOB_HMAC_RMD320, -1}, FAM_HMAC_RMD320},
    {"hmac_rmd320_kpass_batch", {JOB_HMAC_RMD320_KPASS, -1}, FAM_HMAC_RMD320},
    {"hmac_blake2s_kpass_batch", {JOB_HMAC_BLAKE2S, -1}, FAM_HMAC_BLAKE2S},
    {"streebog256_unsalted_batch",     {JOB_STREEBOG_32, -1}, FAM_STREEBOG},
    {"streebog512_unsalted_batch",     {JOB_STREEBOG_64, -1}, FAM_STREEBOG},
    {"hmac_streebog256_kpass_batch",   {JOB_HMAC_STREEBOG256_KPASS, -1}, FAM_STREEBOG},
    {"hmac_streebog256_ksalt_batch",   {JOB_HMAC_STREEBOG256_KSALT, -1}, FAM_STREEBOG},
    {"hmac_streebog512_kpass_batch",   {JOB_HMAC_STREEBOG512_KPASS, -1}, FAM_STREEBOG},
    {"hmac_streebog512_ksalt_batch",   {JOB_HMAC_STREEBOG512_KSALT, -1}, FAM_STREEBOG},
    /* sha512crypt_batch retired in Phase 4 SHA512CRYPTMD5 ship (2026-05-08);
     * orphan kernel_map row was inconsistent with [FAM_SHA512CRYPT]=NULL.
     * Cleaned here alongside bcrypt_batch retirement (task #218). */
    {"sha256crypt_batch",              {JOB_SHA256CRYPT, -1}, FAM_SHA256CRYPT},
    {"rmd160_unsalted_batch",          {JOB_RMD160, -1}, FAM_RMD160UNSALTED},
    {"blake2s256_unsalted_batch",      {JOB_BLAKE2S256, -1}, FAM_BLAKE2S256UNSALTED},
    /* bcrypt_batch retired 2026-05-09 (task #218); see #include block above. */
    {"md5crypt_batch",                 {JOB_MD5CRYPT, -1}, FAM_MD5CRYPT},
    {NULL, {-1}, 0}
};
#define MTL_KERN_ITER_IDX 2

/* Packed kernel registry: maps FAM_* to packed kernel entry point + JIT source.
 * Precompiled metallib is primary; JIT source is fallback only (not expected). */
static const struct {
    const char *source;      /* JIT fallback source (includes metal_common if not self-contained) */
    int         self_contained; /* 1 = no metal_common prefix in JIT path */
    const char *entry;       /* kernel function name */
} mtl_packed_kernel_map[FAM_COUNT] = {
    [FAM_MD5UNSALTED]    = { metal_md5_packed_str,    0, "md5_packed_batch" },
    [FAM_SHA1UNSALTED]   = { metal_sha1_packed_str,   0, "sha1_packed_batch" },
    [FAM_SHA256UNSALTED] = { metal_sha256_packed_str, 0, "sha256_packed_batch" },
    [FAM_MD4UNSALTED]    = { metal_md4_packed_str,    0, "md4_packed_batch" },
    [FAM_SHA512UNSALTED] = { metal_sha512_packed_str, 1, "sha512_packed_batch" },
};

/* Map JOB_* op to packed kernel family. -1 if no packed kernel.
 * Includes JOB_NTLMH + JOB_NTLM (MD4 family, utf-16 packed on host side). */
static int metal_packed_family(int op) {
    switch (op) {
    case JOB_MD5: case JOB_MD5UC: case JOB_MD5RAW:
        return FAM_MD5UNSALTED;
    case JOB_SHA1: case JOB_SHA1RAW: case JOB_SQL5:
        return FAM_SHA1UNSALTED;
    case JOB_SHA256: case JOB_SHA256RAW: case JOB_SHA224:
        return FAM_SHA256UNSALTED;
    case JOB_MD4:
    case JOB_NTLMH:
    case JOB_NTLM:
        return FAM_MD4UNSALTED;
    case JOB_SHA512: case JOB_SHA512RAW: case JOB_SHA384: case JOB_SHA384RAW:
        return FAM_SHA512UNSALTED;
    default:
        return -1;
    }
}

int gpu_metal_packed_capable(int op) {
    return metal_packed_family(op) >= 0;
}

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

/* ---- Packed password buffers (GPU rule dispatch) ---- */
static id<MTLBuffer> buf_packed_data;   /* 128MB password data [len][bytes]... */
static id<MTLBuffer> buf_packed_wordoff;/* uint32 byte offset per word */
static id<MTLBuffer> buf_packed_hits;   /* GPU_PACKED_MAX_HITS * 19 uint32 */
static id<MTLBuffer> buf_packed_hit_cnt;/* atomic uint hit counter */
static id<MTLBuffer> buf_packed_params; /* MetalParams uniform */
static size_t        packed_data_cap = 0;
static size_t        packed_wordoff_cap = 0;
static uint32_t     *packed_hits_host = NULL;  /* caller-visible hit buffer */

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

/* ---- GPU Params struct: 128-byte uniform API (must match kernel) ----
 * All uint64 fields first (8-aligned), then uint32 fields, then reserved.
 * Identical layout for OpenCL and Metal. No padding holes. */
typedef struct {
    /* 8-byte fields (offset 0-31) */
    uint64_t compact_mask;    /*  0: hash table mask */
    uint64_t mask_start;      /*  8: mask keyspace offset for chunking */
    uint64_t mask_base0;      /* 16: pre-decomposed mask_start positions 0-7 */
    uint64_t mask_base1;      /* 24: pre-decomposed mask_start positions 8-15 */
    /* 4-byte fields (offset 32-79) */
    uint32_t num_words;       /* 32: words in this batch */
    uint32_t num_salts;       /* 36: salts for this dispatch */
    uint32_t salt_start;      /* 40: starting salt index */
    uint32_t max_probe;       /* 44: compact table max probe depth */
    uint32_t hash_data_count; /* 48: entries in hash_data */
    uint32_t max_hits;        /* 52: hit buffer capacity */
    uint32_t overflow_count;  /* 56: overflow table entries */
    uint32_t max_iter;        /* 60: iteration count from -i */
    uint32_t num_masks;       /* 64: mask combinations per chunk */
    uint32_t n_prepend;       /* 68: prepend mask positions (-N) */
    uint32_t n_append;        /* 72: append mask positions (-n) */
    uint32_t iter_count;      /* 76: per-dispatch iteration count (PHPBB3) */
    /* Reserved (offset 80-127) */
    uint32_t reserved32[4];   /* 80-95: reserved uint32 */
    uint64_t reserved64[4];   /* 96-127: reserved uint64 */
} MetalParams;

static int _max_iter = 1;
static int _gpu_op = 0;  /* current op type for kernel selection */
static int mtl_max_dispatch = 256 * 1024 * 1024;  /* max work items per dispatch (256M) */

/* ---- Timing-based dispatch sizing ---- */
#define TIMING_BUDGET_MIN_MS  200.0
#define TIMING_BUDGET_MAX_MS  400.0
#define TIMING_INITIAL_SIZE   65536   /* 64K work items */
#define TIMING_WATCHDOG_MS    5000.0

extern int gpu_op_family(int op);

static uint32_t mtl_fam_max_items[FAM_COUNT];
static int      mtl_fam_timed[FAM_COUNT];
static MTLCompileOptions *mtl_compile_opts;  /* saved for deferred family compilation */
static id<MTLLibrary> mtl_precompiled_lib;  /* precompiled metallib (nil = JIT fallback) */

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
        if (@available(macOS 13.0, *))
            opts.optimizationLevel = MTLLibraryOptimizationLevelSize;
        mtl_compile_opts = opts;

        memset(mtl_fam_lib, 0, sizeof(mtl_fam_lib));
        memset(mtl_pipelines, 0, sizeof(mtl_pipelines));
        mtl_pipeline_iter = nil;

        /* Load embedded precompiled metallib — avoids JIT crashes on complex kernels.
         * The metallib is compiled offline by gpu/build_metallib.sh and embedded
         * as a byte array via xxd -i. Architecture-neutral AIR format. */
        mtl_precompiled_lib = nil;
        {
#include "gpu/mdxfind_metallib.h"
            dispatch_data_t mdata = dispatch_data_create(
                gpu_mdxfind_metallib, gpu_mdxfind_metallib_len,
                NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *lerr = nil;
            mtl_precompiled_lib = [mtl_device newLibraryWithData:mdata error:&lerr];
            if (mtl_precompiled_lib)
                fprintf(stderr, "Metal GPU: loaded embedded metallib (%u bytes)\n",
                        gpu_mdxfind_metallib_len);
            else
                fprintf(stderr, "Metal GPU: embedded metallib failed: %s, using JIT\n",
                        lerr ? [[lerr localizedDescription] UTF8String] : "unknown");
        }

        /* Compile/load md5salt family (needed for probe/selftest) */
        if (mtl_precompiled_lib) {
            /* Use precompiled library for md5salt family */
            mtl_fam_lib[FAM_MD5SALT] = mtl_precompiled_lib;
        } else {
            /* JIT fallback */
            NSString *src = [NSString stringWithFormat:@"%s%s",
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
        buf_hits = [mtl_device newBufferWithLength:MAX_GPU_HITS * GPU_HIT_STRIDE * sizeof(uint32_t)
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

            /* Try precompiled metallib first — kernel lookup by name */
            if (mtl_precompiled_lib) {
                int found = 0;
                for (int k = 0; metal_kernel_map[k].name; k++) {
                    if (metal_kernel_map[k].family != f) continue;
                    NSString *fname = [NSString stringWithUTF8String:metal_kernel_map[k].name];
                    id<MTLFunction> fn = [mtl_precompiled_lib newFunctionWithName:fname];
                    if (!fn) continue;
                    id<MTLComputePipelineState> ps = [mtl_device newComputePipelineStateWithFunction:fn error:&error];
                    if (!ps) { fprintf(stderr, "Metal: pipeline failed for '%s': %s\n", metal_kernel_map[k].name, error ? [[error localizedDescription] UTF8String] : "unknown"); continue; }
                    if (k == MTL_KERN_ITER_IDX)
                        mtl_pipeline_iter = ps;
                    for (int j = 0; metal_kernel_map[k].ops[j] >= 0; j++)
                        mtl_pipelines[metal_kernel_map[k].ops[j]] = ps;
                    found++;
                }
                if (found > 0) {
                    mtl_fam_lib[f] = mtl_precompiled_lib;
                    continue;  /* family fully loaded from metallib */
                }
                /* No kernels found in metallib — fall through to JIT */
                fprintf(stderr, "Metal: family %d not in metallib, trying JIT\n", f);
            }

            /* JIT fallback: compile from embedded source strings */
            /* SHA512 family is self-contained (includes own K512, compress, etc.)
             * to reduce combined source size below Metal JIT memory limits. */
            NSString *src;
            if (f == FAM_SHA512UNSALTED || f == FAM_HMAC_SHA512) {
                src = [NSString stringWithUTF8String:mtl_family_source[f]];
                fprintf(stderr, "Metal: family %d self-contained (%lu bytes)\n", f, (unsigned long)[src length]);
            } else {
                src = [NSString stringWithFormat:@"%s%s",
                         metal_common_str, mtl_family_source[f]];
            }
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
                if (!fn) { fprintf(stderr, "Metal: kernel '%s' not found in family %d\n", metal_kernel_map[k].name, f); continue; }
                id<MTLComputePipelineState> ps = [mtl_device newComputePipelineStateWithFunction:fn error:&error];
                if (!ps) { fprintf(stderr, "Metal: pipeline failed for '%s': %s\n", metal_kernel_map[k].name, error ? [[error localizedDescription] UTF8String] : "unknown"); continue; }
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

        /* Hit buffer: 19 uint32 words per hit (universal stride) */
        #define GPU_MAX_HITS 32768
        _max_hits = GPU_MAX_HITS;
        buf_dispatch_hits = [mtl_device newBufferWithLength:_max_hits * GPU_HIT_STRIDE * sizeof(uint32_t)
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

static id<MTLBuffer> buf_mask_desc = nil;

int gpu_metal_set_mask(const uint8_t *sizes, const uint8_t tables[][256],
                       int npre, int napp) {
    if (!mtl_ready) return -1;
    extern uint8_t gpu_mask_desc[];
    extern uint8_t gpu_mask_sizes[];
    extern int gpu_mask_n_prepend, gpu_mask_n_append;
    extern uint64_t gpu_mask_total;

    int ntotal = npre + napp;
    gpu_mask_n_prepend = npre;
    gpu_mask_n_append = napp;
    memcpy(gpu_mask_desc, sizes, ntotal);
    memcpy(gpu_mask_sizes, sizes, ntotal);
    for (int i = 0; i < ntotal; i++)
        memcpy(gpu_mask_desc + ntotal + i * 256, tables[i], 256);
    gpu_mask_total = 1;
    for (int i = 0; i < ntotal; i++)
        gpu_mask_total *= sizes[i];

    int bufsize = ntotal + ntotal * 256;
    buf_mask_desc = [mtl_device newBufferWithBytes:gpu_mask_desc
                                           length:bufsize
                                          options:MTLResourceStorageModeShared];
    fprintf(stderr, "Metal GPU: mask mode: %d prepend + %d append = %llu combinations\n",
            npre, napp, (unsigned long long)gpu_mask_total);
    return 0;
}

void gpu_metal_set_max_iter(int max_iter) {
    _max_iter = (max_iter < 1) ? 1 : max_iter;
    (void)0;
}

static int _iter_count = 0;  /* PHPBB3: uniform iteration count for current dispatch group */

void gpu_metal_set_iter_count(int count) {
    _iter_count = count;
}

void gpu_metal_set_op(int op) {
    _gpu_op = op;
}

static uint64_t _mask_resume = 0;
static uint32_t _salt_resume = 0;
static uint64_t _last_mask_start = 0;  /* mask_start of the chunk that returned hits */

void gpu_metal_set_mask_resume(uint32_t start) { _mask_resume = start; }
void gpu_metal_set_salt_resume(uint32_t start) { _salt_resume = start; }
int gpu_metal_has_resume(void) { return _mask_resume > 0 || _salt_resume > 0; }
uint64_t gpu_metal_last_mask_start(void) { return _last_mask_start; }

uint32_t *gpu_metal_dispatch_batch(
    const char *hexhashes,
    const uint16_t *hexlens,
    int num_words,
    int *nhits_out)
{
    @autoreleasepool {
        *nhits_out = 0;
        if (!mtl_ready) return NULL;
        if (num_words <= 0) return NULL;

        /* Lazy init dispatch buffers if not yet allocated (unsalted-only runs) */
        if (!_dispatch_bufs_ready) {
            #define GPU_MAX_HITS 32768
            _max_hits = GPU_MAX_HITS;
            if (!buf_dispatch_hits)
                buf_dispatch_hits = [mtl_device newBufferWithLength:_max_hits * GPU_HIT_STRIDE * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
            if (!buf_dispatch_hit_count)
                buf_dispatch_hit_count = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
            if (!buf_dispatch_hexhashes)
                buf_dispatch_hexhashes = [mtl_device newBufferWithLength:4096 * 712
                                                                 options:MTLResourceStorageModeShared];
            if (!buf_dispatch_hexlens)
                buf_dispatch_hexlens = [mtl_device newBufferWithLength:4096 * sizeof(uint16_t)
                                                                options:MTLResourceStorageModeShared];
            if (!buf_dummy)
                buf_dummy = [mtl_device newBufferWithLength:4
                                                    options:MTLResourceStorageModeShared];
            if (!buf_salt_data)
                buf_salt_data = [mtl_device newBufferWithLength:16
                                                        options:MTLResourceStorageModeShared];
            if (!buf_salt_off)
                buf_salt_off = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
            if (!buf_salt_len)
                buf_salt_len = [mtl_device newBufferWithLength:sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
            _dispatch_bufs_ready = 1;
        }
        int cat = gpu_op_category(_gpu_op);
        int is_unsalted = (cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED);
        if (_salts_count <= 0 && !is_unsalted) return NULL;

        /* Copy word data into persistent buffers.
         * Unsalted pre-padded: num_words can exceed GPUBATCH_MAX (up to 4096),
         * use actual word_stride from caller (64/128/etc).
         * Salted types: always 256-byte stride, max GPUBATCH_MAX words. */
        int word_stride;
        if (!is_unsalted) {
            word_stride = 256;
        } else if (_gpu_op == JOB_SHA512 || _gpu_op == JOB_SHA384 ||
                   _gpu_op == JOB_SHA512RAW || _gpu_op == JOB_SHA384RAW) {
            word_stride = 128;
        } else if (_gpu_op == JOB_MD6256) {
            word_stride = 712;
        } else if (_gpu_op == JOB_KECCAK224 || _gpu_op == JOB_SHA3_224) {
            word_stride = 152;
        } else if (_gpu_op == JOB_KECCAK256 || _gpu_op == JOB_SHA3_256) {
            word_stride = 144;
        } else if (_gpu_op == JOB_KECCAK384 || _gpu_op == JOB_SHA3_384) {
            word_stride = 112;
        } else if (_gpu_op == JOB_KECCAK512 || _gpu_op == JOB_SHA3_512) {
            word_stride = 80;
        } else {
            word_stride = 64;
        }
        size_t words_size = (size_t)num_words * word_stride;
        memcpy([buf_dispatch_hexhashes contents], hexhashes, words_size);
        memcpy([buf_dispatch_hexlens contents], hexlens, num_words * sizeof(uint16_t));

        /* Params */
        MetalParams params;
        memset(&params, 0, sizeof(params));
        params.compact_mask = _compact_mask;
        params.num_words = num_words;
        params.num_salts = is_unsalted ? 0 : _salts_count;
        params.salt_start = 0;
        params.max_probe = 256;
        params.hash_data_count = _hash_data_count;
        params.max_hits = _max_hits;
        params.overflow_count = _overflow_count;
        params.max_iter = _max_iter;
        params.iter_count = _iter_count;
        { extern int gpu_mask_n_prepend, gpu_mask_n_append;
          extern uint64_t gpu_mask_total;
          params.num_masks = is_unsalted ? (gpu_mask_total > 0 ? (gpu_mask_total < 0xFFFFFFFF ? (uint32_t)gpu_mask_total : 1) : 1) : 0;
          params.mask_start = 0;
          params.n_prepend = gpu_mask_n_prepend;
          params.n_append = gpu_mask_n_append;
        }

        /* Determine mask/salt chunking parameters */
        extern uint64_t gpu_mask_total;
        int is_mask = (is_unsalted && gpu_mask_total > 1);
        int total_salts = _salts_count;

        /* --- Timing-based dispatch sizing probe --- */
        int fam = gpu_op_family(_gpu_op);
        int is_bcrypt = (fam == FAM_BCRYPT);

        /* Look up pipeline early — needed for timing probe */
        id<MTLComputePipelineState> pipeline =
            (!is_unsalted && _max_iter > 1 && mtl_pipeline_iter) ? mtl_pipeline_iter :
            (_gpu_op >= 0 && _gpu_op < JOB_DONE && mtl_pipelines[_gpu_op]) ? mtl_pipelines[_gpu_op] : nil;
        if (!pipeline) {
            fprintf(stderr, "Metal: no pipeline for op %d\n", _gpu_op);
            return NULL;
        }

        NSUInteger tpg = [pipeline maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        if (_gpu_op == JOB_BCRYPT && tpg > 8) tpg = 8;
        if (_gpu_op == JOB_SHA512CRYPT || _gpu_op == JOB_SHA512CRYPTMD5) {
            if (tpg > 32) tpg = 32;
        }
        MTLSize groupSize = MTLSizeMake(tpg, 1, 1);

        if (fam >= 0 && fam < FAM_COUNT && !mtl_fam_timed[fam] && (is_mask || !is_unsalted)) {
            uint32_t probe_size = is_bcrypt ? 1 : TIMING_INITIAL_SIZE;
            uint32_t best_size = probe_size;
            double best_ms = 0;

            for (int probe = 0; probe < 20; probe++) {
                *(uint32_t *)[buf_dispatch_hit_count contents] = 0;

                MetalParams probe_params = params;
                if (is_mask) {
                    uint32_t pm = probe_size / (num_words > 0 ? num_words : 1);
                    if (pm < 1) pm = 1;
                    if ((uint64_t)pm > gpu_mask_total) pm = (gpu_mask_total < 0xFFFFFFFF) ? (uint32_t)gpu_mask_total : 0xFFFFFFFF;
                    probe_params.num_masks = pm;
                    probe_params.mask_start = 0;
                } else {
                    int ps = (int)(probe_size / (num_words > 0 ? num_words : 1));
                    if (ps < 1) ps = 1;
                    if (ps > total_salts) ps = total_salts;
                    probe_params.num_salts = ps;
                    probe_params.salt_start = 0;
                }

                uint64_t probe_threads = is_mask
                    ? (uint64_t)num_words * probe_params.num_masks
                    : (uint64_t)num_words * probe_params.num_salts;

                id<MTLCommandBuffer> pcmd = [mtl_queue commandBuffer];
                id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
                [penc setComputePipelineState:pipeline];
                [penc setBuffer:buf_dispatch_hexhashes offset:0 atIndex:0];
                [penc setBuffer:buf_dispatch_hexlens  offset:0 atIndex:1];
                [penc setBuffer:buf_dispatch_hexlens  offset:0 atIndex:2];
                [penc setBuffer:(is_unsalted ? (buf_mask_desc ? buf_mask_desc : buf_dummy) : buf_salt_data)
                                                 offset:0 atIndex:3];
                [penc setBuffer:buf_salt_off      offset:0 atIndex:4];
                [penc setBuffer:buf_salt_len      offset:0 atIndex:5];
                [penc setBuffer:buf_compact_fp    offset:0 atIndex:6];
                [penc setBuffer:buf_compact_idx   offset:0 atIndex:7];
                [penc setBytes:&probe_params length:sizeof(probe_params) atIndex:8];
                [penc setBuffer:buf_hash_data     offset:0 atIndex:9];
                [penc setBuffer:buf_hash_data_off offset:0 atIndex:10];
                [penc setBuffer:buf_hash_data_len offset:0 atIndex:11];
                [penc setBuffer:buf_dispatch_hits offset:0 atIndex:12];
                [penc setBuffer:buf_dispatch_hit_count offset:0 atIndex:13];
                [penc setBuffer:(buf_overflow_keys    ? buf_overflow_keys    : buf_dummy) offset:0 atIndex:14];
                [penc setBuffer:(buf_overflow_hashes  ? buf_overflow_hashes  : buf_dummy) offset:0 atIndex:15];
                [penc setBuffer:(buf_overflow_offsets ? buf_overflow_offsets : buf_dummy) offset:0 atIndex:16];
                [penc setBuffer:(buf_overflow_lengths ? buf_overflow_lengths : buf_dummy) offset:0 atIndex:17];

                MTLSize probeGrid = MTLSizeMake(probe_threads, 1, 1);
                [penc dispatchThreads:probeGrid threadsPerThreadgroup:groupSize];
                [penc endEncoding];

                NSDate *pstart = [NSDate date];
                [pcmd commit];
                [pcmd waitUntilCompleted];
                double pms = -[pstart timeIntervalSinceNow] * 1000.0;

                best_size = probe_size;
                best_ms = pms;

                if (is_bcrypt) {
                    if (pms > 0) {
                        double per_item = pms;
                        uint32_t target = (uint32_t)(TIMING_BUDGET_MIN_MS / per_item);
                        if (target < 1) target = 1;
                        best_size = target;
                    }
                    break;
                }

                if (pms >= TIMING_BUDGET_MIN_MS && pms <= TIMING_BUDGET_MAX_MS)
                    break;
                if (pms > TIMING_BUDGET_MAX_MS) {
                    best_size = probe_size / 2;
                    if (best_size < TIMING_INITIAL_SIZE) best_size = TIMING_INITIAL_SIZE;
                    break;
                }
                if (pms >= TIMING_WATCHDOG_MS)
                    break;

                uint64_t next = (uint64_t)probe_size * 2;
                if (next > 0x7FFFFFFF) next = 0x7FFFFFFF;
                probe_size = (uint32_t)next;
            }

            mtl_fam_max_items[fam] = best_size;
            mtl_fam_timed[fam] = 1;
            fprintf(stderr, "Metal GPU: timed family %d: max_items=%u (%.1fms)\n",
                    fam, best_size, best_ms);
        }

        /* Use timed chunk size if available, otherwise fall back to defaults */
        uint32_t timed_max = (fam >= 0 && fam < FAM_COUNT && mtl_fam_timed[fam])
            ? mtl_fam_max_items[fam] : 0;

        int salt_chunk = total_salts;
        if (timed_max > 0 && !is_unsalted && num_words > 0) {
            salt_chunk = (int)(timed_max / num_words);
            if (salt_chunk < 1024) salt_chunk = 1024;
            if (salt_chunk > total_salts) salt_chunk = total_salts;
        } else if (mtl_max_dispatch > 0 && num_words > 0 && total_salts > 0) {
            salt_chunk = mtl_max_dispatch / num_words;
            if (salt_chunk < 1024) salt_chunk = 1024;
            if (salt_chunk > total_salts) salt_chunk = total_salts;
        }

#define MASK_CHUNK_DEFAULT 4194304  /* 4M mask combinations per dispatch (fallback) */
        /* Apply resume point from previous hit-interrupted dispatch */
        uint64_t mask_start_base = is_mask ? _mask_resume : 0;
        _mask_resume = 0;  /* consumed */
        int salt_start_base = !is_unsalted ? (int)_salt_resume : 0;
        _salt_resume = 0;

        uint64_t mask_total = is_mask ? (gpu_mask_total - mask_start_base) : 0;
        uint64_t mask_chunk;
        if (is_mask) {
            if (timed_max > 0 && num_words > 0)
                mask_chunk = timed_max / num_words;
            else
                mask_chunk = MASK_CHUNK_DEFAULT;
            if (mask_chunk < 1) mask_chunk = 1;
            if (mask_chunk > mask_total) mask_chunk = mask_total;
        } else {
            mask_chunk = 0;
        }

        int num_chunks;
        if (is_mask)
            num_chunks = (int)((mask_total + mask_chunk - 1) / mask_chunk);
        else if (!is_unsalted)
            num_chunks = (total_salts + salt_chunk - 1) / salt_chunk;
        else
            num_chunks = 1;

        /* Chunk loop: each chunk is committed separately.
         * Zero hit counter before each chunk. If any hits are found,
         * return immediately so the gpujob can process them.
         * The caller uses mask_resume/salt_resume to continue. */
        extern unsigned long long Tothash;
        extern uint8_t gpu_mask_sizes[];
        extern int gpu_mask_n_prepend, gpu_mask_n_append;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            if (is_mask) {
                params.mask_start = mask_start_base + (uint64_t)chunk * mask_chunk;
                { uint64_t remaining = mask_total - (uint64_t)chunk * mask_chunk;
                  if (remaining > mask_chunk) remaining = mask_chunk;
                  params.num_masks = (uint32_t)remaining;
                }
                /* Pre-decompose mask_start into per-position base offsets.
                 * The kernel does fast uint32 local decomposition and adds
                 * to these bases with carry, avoiding 64-bit GPU division. */
                { uint64_t ms = params.mask_start;
                  uint64_t b0 = 0, b1 = 0;
                  int ntot = gpu_mask_n_prepend + gpu_mask_n_append;
                  for (int i = ntot - 1; i >= 0; i--) {
                    uint8_t sz = gpu_mask_sizes[i];
                    uint8_t digit = (uint8_t)(ms % sz);
                    ms /= sz;
                    if (i < 8) b0 |= (uint64_t)digit << (i * 8);
                    else       b1 |= (uint64_t)digit << ((i - 8) * 8);
                  }
                  params.mask_base0 = b0;
                  params.mask_base1 = b1;
                }
            } else if (num_chunks > 1) {
                params.salt_start = chunk * salt_chunk;
                params.num_salts = total_salts - params.salt_start;
                if ((int)params.num_salts > salt_chunk) params.num_salts = salt_chunk;
            }

            uint64_t chunk_threads = is_mask
                ? (uint64_t)num_words * params.num_masks
                : is_unsalted
                ? (uint64_t)num_words * (params.num_masks > 0 ? params.num_masks : 1)
                : (uint64_t)num_words * params.num_salts;

            /* Zero hit counter before each chunk */
            *(uint32_t *)[buf_dispatch_hit_count contents] = 0;

            /* Encode, commit, and wait */
            id<MTLCommandBuffer> ccmdbuf = [mtl_queue commandBuffer];
            id<MTLComputeCommandEncoder> cenc = [ccmdbuf computeCommandEncoder];
            [cenc setComputePipelineState:pipeline];
            [cenc setBuffer:buf_dispatch_hexhashes offset:0 atIndex:0];
            [cenc setBuffer:buf_dispatch_hexlens  offset:0 atIndex:1];
            [cenc setBuffer:buf_dispatch_hexlens  offset:0 atIndex:2];
            [cenc setBuffer:(is_unsalted ? (buf_mask_desc ? buf_mask_desc : buf_dummy) : buf_salt_data)
                                                     offset:0 atIndex:3];
            [cenc setBuffer:buf_salt_off      offset:0 atIndex:4];
            [cenc setBuffer:buf_salt_len      offset:0 atIndex:5];
            [cenc setBuffer:buf_compact_fp    offset:0 atIndex:6];
            [cenc setBuffer:buf_compact_idx   offset:0 atIndex:7];
            [cenc setBytes:&params length:sizeof(params) atIndex:8];
            [cenc setBuffer:buf_hash_data     offset:0 atIndex:9];
            [cenc setBuffer:buf_hash_data_off offset:0 atIndex:10];
            [cenc setBuffer:buf_hash_data_len offset:0 atIndex:11];
            [cenc setBuffer:buf_dispatch_hits offset:0 atIndex:12];
            [cenc setBuffer:buf_dispatch_hit_count offset:0 atIndex:13];
            [cenc setBuffer:(buf_overflow_keys    ? buf_overflow_keys    : buf_dummy) offset:0 atIndex:14];
            [cenc setBuffer:(buf_overflow_hashes  ? buf_overflow_hashes  : buf_dummy) offset:0 atIndex:15];
            [cenc setBuffer:(buf_overflow_offsets ? buf_overflow_offsets : buf_dummy) offset:0 atIndex:16];
            [cenc setBuffer:(buf_overflow_lengths ? buf_overflow_lengths : buf_dummy) offset:0 atIndex:17];
            MTLSize gridSize = MTLSizeMake(chunk_threads, 1, 1);
            [cenc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
            [cenc endEncoding];
            [ccmdbuf commit];
            [ccmdbuf waitUntilCompleted];

            /* Update Tothash for progress tracking */
            __sync_fetch_and_add(&Tothash, chunk_threads);

            /* Check for hits — return immediately if any found */
            uint32_t chunk_hits = *(uint32_t *)[buf_dispatch_hit_count contents];
            if (chunk_hits > 0) {
                *nhits_out = (int)chunk_hits;
                _last_mask_start = params.mask_start;  /* for gpujob hit reconstruction */
                /* Set resume point for next call: mask or salt position after this chunk */
                if (is_mask) {
                    uint64_t next_start = mask_start_base + (uint64_t)(chunk + 1) * mask_chunk;
                    if (next_start < gpu_mask_total)
                        _mask_resume = next_start;
                } else if (!is_unsalted && chunk + 1 < num_chunks) {
                    _salt_resume = salt_start_base + (chunk + 1) * salt_chunk;
                }
                return (uint32_t *)[buf_dispatch_hits contents];
            }
        }

        /* No hits found across all chunks */
        *nhits_out = 0;
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
            gpu_slots[i].buf_hits = [mtl_device newBufferWithLength:GPU_SLOT_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t)
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
          if (!ps) { [enc endEncoding]; [cmdbuf commit]; [cmdbuf waitUntilCompleted]; return -1; }
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
        id<MTLComputePipelineState> active_ps =
            (_max_iter > 1 && mtl_pipeline_iter) ? mtl_pipeline_iter :
            mtl_pipelines[_gpu_op] ? mtl_pipelines[_gpu_op] : mtl_pipelines[JOB_MD5SALT];
        NSUInteger tpg = [active_ps maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        /* bcrypt kernel partitions threadgroup sbox_pool by lid — cap to BCRYPT_WG_SIZE */
        if (_gpu_op == JOB_BCRYPT && tpg > 8) tpg = 8;
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
        memset(&params, 0, sizeof(params));
        params.num_words = num_words;
        params.num_salts = num_salts;
        params.max_probe = 256;
        params.hash_data_count = 0;

        params.compact_mask = _compact_mask;
        params.hash_data_count = _hash_data_count;
        params.iter_count = _iter_count;
        params.overflow_count = _overflow_count;
        params.max_iter = _max_iter;

        int hit_capacity = (max_hits < 1024) ? max_hits : 1024;
        params.max_hits = hit_capacity;

        id<MTLBuffer> b_params_buf = [mtl_device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> local_hit_count = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                                options:MTLResourceStorageModeShared];
        *(uint32_t *)[local_hit_count contents] = 0;
        id<MTLBuffer> local_hits = [mtl_device newBufferWithLength:hit_capacity * GPU_HIT_STRIDE * sizeof(uint32_t)
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

/* ---- Packed password dispatch (GPU rule path) ---- */

/* Lazy-compile packed kernel pipeline for a given family.
 * Primary path: precompiled metallib. Falls back to JIT only if metallib
 * is unavailable or doesn't contain the kernel.
 * Returns the pipeline, or nil on failure. */
static id<MTLComputePipelineState> mtl_get_packed_pipeline(int pfam) {
    if (pfam < 0 || pfam >= FAM_COUNT) return nil;
    if (mtl_pipeline_packed[pfam]) return mtl_pipeline_packed[pfam];
    if (mtl_pipeline_packed_tried[pfam]) return nil;
    mtl_pipeline_packed_tried[pfam] = 1;

    const char *entry = mtl_packed_kernel_map[pfam].entry;
    if (!entry) return nil;
    NSString *fname = [NSString stringWithUTF8String:entry];
    NSError *error = nil;

    /* Primary: precompiled metallib */
    if (mtl_precompiled_lib) {
        id<MTLFunction> fn = [mtl_precompiled_lib newFunctionWithName:fname];
        if (fn) {
            id<MTLComputePipelineState> ps = [mtl_device newComputePipelineStateWithFunction:fn error:&error];
            if (ps) {
                mtl_pipeline_packed[pfam] = ps;
                return ps;
            }
            fprintf(stderr, "Metal: packed pipeline '%s' failed: %s\n", entry,
                    error ? [[error localizedDescription] UTF8String] : "unknown");
        } else {
            fprintf(stderr, "Metal: packed kernel '%s' not in metallib\n", entry);
        }
    }

    /* JIT fallback (should not happen in production — metallib is authoritative) */
    const char *src = mtl_packed_kernel_map[pfam].source;
    int selfc = mtl_packed_kernel_map[pfam].self_contained;
    if (!src || !mtl_compile_opts) return nil;
    NSString *nssrc = selfc ?
        [NSString stringWithUTF8String:src] :
        [NSString stringWithFormat:@"%s%s", metal_common_str, src];
    id<MTLLibrary> lib = [mtl_device newLibraryWithSource:nssrc options:mtl_compile_opts error:&error];
    if (!lib) {
        fprintf(stderr, "Metal: packed JIT '%s' compile error: %s\n", entry,
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:fname];
    if (!fn) return nil;
    id<MTLComputePipelineState> ps = [mtl_device newComputePipelineStateWithFunction:fn error:&error];
    if (ps) mtl_pipeline_packed[pfam] = ps;
    return ps;
}

uint32_t *gpu_metal_dispatch_packed(
    const char *packed_buf,
    uint32_t packed_size,
    const uint32_t *word_offset,
    uint32_t num_words,
    uint32_t word_start,
    int op,
    int *nhits_out)
{
    *nhits_out = 0;
    if (!mtl_ready || num_words == 0 || !packed_buf || !word_offset) return NULL;

    int pfam = metal_packed_family(op);
    if (pfam < 0) return NULL;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = mtl_get_packed_pipeline(pfam);
        if (!pipeline) return NULL;

        /* Lazy-allocate persistent packed buffers */
        if (packed_data_cap < packed_size) {
            buf_packed_data = [mtl_device newBufferWithLength:packed_size
                                                     options:MTLResourceStorageModeShared];
            packed_data_cap = packed_size;
        }
        size_t wo_size = (size_t)num_words * sizeof(uint32_t);
        if (packed_wordoff_cap < wo_size) {
            buf_packed_wordoff = [mtl_device newBufferWithLength:wo_size
                                                         options:MTLResourceStorageModeShared];
            packed_wordoff_cap = wo_size;
        }
        if (!buf_packed_hits) {
            buf_packed_hits = [mtl_device newBufferWithLength:GPU_PACKED_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
            buf_packed_hit_cnt = [mtl_device newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
            buf_packed_params = [mtl_device newBufferWithLength:sizeof(MetalParams)
                                                        options:MTLResourceStorageModeShared];
            packed_hits_host = (uint32_t *)[buf_packed_hits contents];
        }

        /* Copy data + offsets on first chunk only (chunked dispatch reuses buffer) */
        if (word_start == 0) {
            memcpy([buf_packed_data contents], packed_buf, packed_size);
            memcpy([buf_packed_wordoff contents], word_offset, wo_size);
        }

        /* Fill params */
        MetalParams *params = (MetalParams *)[buf_packed_params contents];
        memset(params, 0, sizeof(*params));
        params->compact_mask = _compact_mask;
        params->num_words = num_words;
        params->max_probe = 256;
        params->hash_data_count = _hash_data_count;
        params->max_hits = GPU_PACKED_MAX_HITS;
        params->overflow_count = _overflow_count;
        params->max_iter = (_max_iter > 0) ? _max_iter : 1;
        params->reserved32[0] = word_start;
        params->reserved32[1] = packed_size;

        /* Zero hit counter */
        *((uint32_t *)[buf_packed_hit_cnt contents]) = 0;

        /* Compute dispatch size: one thread per word, capped at 16M per chunk */
        uint32_t dispatch_words = num_words - word_start;
        if (dispatch_words > 16777216u) dispatch_words = 16777216u;

        id<MTLCommandBuffer> cmdbuf = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_packed_data       offset:0 atIndex:0];
        [enc setBuffer:buf_packed_wordoff    offset:0 atIndex:1];
        /* Buffers 2-5: unused placeholders (kernel signature compatibility).
         * Reuse small existing buffers to satisfy Metal argument requirements. */
        [enc setBuffer:(buf_word_data ? buf_word_data : buf_packed_data) offset:0 atIndex:2];
        [enc setBuffer:(buf_word_off  ? buf_word_off  : buf_packed_wordoff) offset:0 atIndex:3];
        [enc setBuffer:(buf_word_len  ? buf_word_len  : buf_packed_hit_cnt) offset:0 atIndex:4];
        [enc setBuffer:(buf_salt_data ? buf_salt_data : buf_packed_data) offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp        offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx       offset:0 atIndex:7];
        [enc setBuffer:buf_packed_params     offset:0 atIndex:8];
        [enc setBuffer:buf_hash_data         offset:0 atIndex:9];
        [enc setBuffer:buf_hash_data_off     offset:0 atIndex:10];
        [enc setBuffer:buf_hash_data_len     offset:0 atIndex:11];
        [enc setBuffer:buf_packed_hits       offset:0 atIndex:12];
        [enc setBuffer:buf_packed_hit_cnt    offset:0 atIndex:13];
        [enc setBuffer:(buf_overflow_keys    ? buf_overflow_keys    : buf_packed_data) offset:0 atIndex:14];
        [enc setBuffer:(buf_overflow_hashes  ? buf_overflow_hashes  : buf_packed_data) offset:0 atIndex:15];
        [enc setBuffer:(buf_overflow_offsets ? buf_overflow_offsets : buf_packed_wordoff) offset:0 atIndex:16];
        [enc setBuffer:(buf_overflow_lengths ? buf_overflow_lengths : buf_packed_wordoff) offset:0 atIndex:17];

        NSUInteger tg_size = [pipeline maxTotalThreadsPerThreadgroup];
        if (tg_size > 256) tg_size = 256;
        MTLSize grid = MTLSizeMake(dispatch_words, 1, 1);
        MTLSize tgroup = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgroup];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        uint32_t raw_nhits = *((uint32_t *)[buf_packed_hit_cnt contents]);
        *nhits_out = (int)raw_nhits;
        if (raw_nhits > GPU_PACKED_MAX_HITS) raw_nhits = GPU_PACKED_MAX_HITS;
        return (raw_nhits > 0) ? packed_hits_host : NULL;
    }
}

#endif /* __APPLE__ && METAL_GPU */
