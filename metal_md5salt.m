/*
 * metal_md5salt.m — Apple Metal GPU acceleration for MD5SALT (E31)
 *
 * Thin Objective-C wrapper around Metal compute pipeline.
 * Exports a pure C interface defined in metal_md5salt.h.
 */

#if defined(__APPLE__) && defined(METAL_GPU)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_md5salt.h"
#include "gpujob.h"
#include <string.h>
#include <stdio.h>
#include <mach-o/dyld.h>

/* ---- Metal state ---- */
static id<MTLDevice>              mtl_device;
static id<MTLCommandQueue>        mtl_queue;
static id<MTLComputePipelineState> mtl_pipeline;
static id<MTLComputePipelineState> mtl_pipeline_salts; /* salts-only kernel */
static id<MTLComputePipelineState> mtl_pipeline_batch; /* batch pre-hashed kernel */
static id<MTLComputePipelineState> mtl_pipeline_iter;  /* iterating batch kernel */
static id<MTLComputePipelineState> mtl_pipeline_sub8;  /* sub8-24 batch kernel */
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
} gpu_slots[GPU_NUM_SLOTS];
static int gpu_slots_ready = 0;

/* ---- Params buffer ---- */
typedef struct {
    uint64_t compact_mask;
    uint32_t num_words;
    uint32_t num_salts;
    uint32_t max_probe;
    uint32_t hash_data_count;
    uint32_t max_hits;
    uint32_t overflow_count;
    uint32_t max_iter;
} MetalParams;

static int _max_iter = 1;
static int _gpu_op = 0;  /* current op type for kernel selection */

/* ---- MSL Kernel Source ---- */
static NSString *kernel_source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

/* MD5 constants */
constant uint K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};
constant uint S[64] = {
    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
    5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
};
constant uint G[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    1,6,11,0,5,10,15,4,9,14,3,8,13,2,7,12,
    5,8,11,14,1,4,7,10,13,0,3,6,9,12,15,2,
    0,7,14,5,12,3,10,1,8,15,6,13,4,11,2,9
};

/* MD5 compress: one 64-byte block */
void md5_block(thread uint4 &state, thread const uint *M) {
    uint a = state.x, b = state.y, c = state.z, d = state.w;
    for (int i = 0; i < 64; i++) {
        uint f, g = G[i];
        if (i < 16)      f = (b & c) | (~b & d);
        else if (i < 32) f = (d & b) | (~d & c);
        else if (i < 48) f = b ^ c ^ d;
        else              f = c ^ (~d | b);
        f = f + a + K[i] + M[g];
        a = d; d = c; c = b;
        b = b + ((f << S[i]) | (f >> (32 - S[i])));
    }
    state += uint4(a, b, c, d);
}

/* MD5 compress from round 8: fully unrolled, no branches.
 * Takes pre-computed (a,b,c,d) after rounds 0-7.
 * Adds IV to produce final hash. */
#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + ((a<<s)|(a>>(32-s))); }
#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + ((a<<s)|(a>>(32-s))); }
#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + ((a<<s)|(a>>(32-s))); }
#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + ((a<<s)|(a>>(32-s))); }

void md5_block_from8(thread uint4 &state, thread const uint *M) {
    uint a = state.x, b = state.y, c = state.z, d = state.w;
    /* Rounds 8-15: F function */
    FF(a,b,c,d, M[ 8], 7, 0x698098d8)
    FF(d,a,b,c, M[ 9],12, 0x8b44f7af)
    FF(c,d,a,b, M[10],17, 0xffff5bb1)
    FF(b,c,d,a, M[11],22, 0x895cd7be)
    FF(a,b,c,d, M[12], 7, 0x6b901122)
    FF(d,a,b,c, M[13],12, 0xfd987193)
    FF(c,d,a,b, M[14],17, 0xa679438e)
    FF(b,c,d,a, M[15],22, 0x49b40821)
    /* Rounds 16-31: G function */
    GG(a,b,c,d, M[ 1], 5, 0xf61e2562)
    GG(d,a,b,c, M[ 6], 9, 0xc040b340)
    GG(c,d,a,b, M[11],14, 0x265e5a51)
    GG(b,c,d,a, M[ 0],20, 0xe9b6c7aa)
    GG(a,b,c,d, M[ 5], 5, 0xd62f105d)
    GG(d,a,b,c, M[10], 9, 0x02441453)
    GG(c,d,a,b, M[15],14, 0xd8a1e681)
    GG(b,c,d,a, M[ 4],20, 0xe7d3fbc8)
    GG(a,b,c,d, M[ 9], 5, 0x21e1cde6)
    GG(d,a,b,c, M[14], 9, 0xc33707d6)
    GG(c,d,a,b, M[ 3],14, 0xf4d50d87)
    GG(b,c,d,a, M[ 8],20, 0x455a14ed)
    GG(a,b,c,d, M[13], 5, 0xa9e3e905)
    GG(d,a,b,c, M[ 2], 9, 0xfcefa3f8)
    GG(c,d,a,b, M[ 7],14, 0x676f02d9)
    GG(b,c,d,a, M[12],20, 0x8d2a4c8a)
    /* Rounds 32-47: H function */
    HH(a,b,c,d, M[ 5], 4, 0xfffa3942)
    HH(d,a,b,c, M[ 8],11, 0x8771f681)
    HH(c,d,a,b, M[11],16, 0x6d9d6122)
    HH(b,c,d,a, M[14],23, 0xfde5380c)
    HH(a,b,c,d, M[ 1], 4, 0xa4beea44)
    HH(d,a,b,c, M[ 4],11, 0x4bdecfa9)
    HH(c,d,a,b, M[ 7],16, 0xf6bb4b60)
    HH(b,c,d,a, M[10],23, 0xbebfbc70)
    HH(a,b,c,d, M[13], 4, 0x289b7ec6)
    HH(d,a,b,c, M[ 0],11, 0xeaa127fa)
    HH(c,d,a,b, M[ 3],16, 0xd4ef3085)
    HH(b,c,d,a, M[ 6],23, 0x04881d05)
    HH(a,b,c,d, M[ 9], 4, 0xd9d4d039)
    HH(d,a,b,c, M[12],11, 0xe6db99e5)
    HH(c,d,a,b, M[15],16, 0x1fa27cf8)
    HH(b,c,d,a, M[ 2],23, 0xc4ac5665)
    /* Rounds 48-63: I function */
    II(a,b,c,d, M[ 0], 6, 0xf4292244)
    II(d,a,b,c, M[ 7],10, 0x432aff97)
    II(c,d,a,b, M[14],15, 0xab9423a7)
    II(b,c,d,a, M[ 5],21, 0xfc93a039)
    II(a,b,c,d, M[12], 6, 0x655b59c3)
    II(d,a,b,c, M[ 3],10, 0x8f0ccc92)
    II(c,d,a,b, M[10],15, 0xffeff47d)
    II(b,c,d,a, M[ 1],21, 0x85845dd1)
    II(a,b,c,d, M[ 8], 6, 0x6fa87e4f)
    II(d,a,b,c, M[15],10, 0xfe2ce6e0)
    II(c,d,a,b, M[ 6],15, 0xa3014314)
    II(b,c,d,a, M[13],21, 0x4e0811a1)
    II(a,b,c,d, M[ 4], 6, 0xf7537e82)
    II(d,a,b,c, M[11],10, 0xbd3af235)
    II(c,d,a,b, M[ 2],15, 0x2ad7d2bb)
    II(b,c,d,a, M[ 9],21, 0xeb86d391)
    state = uint4(0x67452301 + a, 0xEFCDAB89 + b, 0x98BADCFE + c, 0x10325476 + d);
}

/* Full MD5 hash for messages up to 55 bytes (single block) */
void md5_short(thread const uint8_t *msg, int len, thread uint4 &hash) {
    uint M[16] = {0};
    for (int i = 0; i < len; i++)
        ((thread uint8_t *)M)[i] = msg[i];
    ((thread uint8_t *)M)[len] = 0x80;
    M[14] = len * 8;
    hash = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block(hash, M);
}

/* MD5 for messages 56-119 bytes (two blocks) */
void md5_two(thread const uint8_t *msg, int len, thread uint4 &hash) {
    uint M[16] = {0};
    /* first block */
    for (int i = 0; i < 64 && i < len; i++)
        ((thread uint8_t *)M)[i] = msg[i];
    if (len < 64) {
        ((thread uint8_t *)M)[len] = 0x80;
        if (len < 56) { M[14] = len * 8; }
    }
    hash = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block(hash, M);
    if (len >= 56) {
        /* second block */
        for (int i = 0; i < 16; i++) M[i] = 0;
        for (int i = 64; i < len; i++)
            ((thread uint8_t *)M)[i - 64] = msg[i];
        if (len >= 64)
            ((thread uint8_t *)M)[len - 64] = 0x80;
        M[14] = len * 8;
        md5_block(hash, M);
    }
}

/* Hex-encode 16 bytes to 32 chars */
void hex_encode(thread const uint8_t *bin, thread uint8_t *hex) {
    for (int i = 0; i < 16; i++) {
        uint8_t hi = bin[i] >> 4;
        uint8_t lo = bin[i] & 0x0f;
        hex[i*2]   = hi < 10 ? hi + '0' : hi - 10 + 'a';
        hex[i*2+1] = lo < 10 ? lo + '0' : lo - 10 + 'a';
    }
}

/* compact_mix: XOR-fold first 8 hash bytes */
uint64_t compact_mix(uint64_t k) {
    return k ^ (k >> 32);
}

struct MetalParams {
    uint64_t compact_mask;
    uint     num_words;
    uint     num_salts;
    uint     max_probe;
    uint     hash_data_count;
    uint     max_hits;
    uint     overflow_count;
    uint     max_iter;
};

/* Hex-encode 4 uint32 hash to 32 bytes in M[0..7] for iteration */
static inline void hash_to_hex_M(uint4 h, thread uint *M) {
    /* Copy to local array to ensure addressable byte layout */
    uint hwords[4] = { h.x, h.y, h.z, h.w };
    thread uint8_t *mb = (thread uint8_t *)M;
    thread const uint8_t *hb = (thread const uint8_t *)hwords;
    for (int i = 0; i < 16; i++) {
        uint8_t hi = hb[i] >> 4;
        uint8_t lo = hb[i] & 0xf;
        mb[i*2]   = hi + (hi < 10 ? '0' : 'a' - 10);
        mb[i*2+1] = lo + (lo < 10 ? '0' : 'a' - 10);
    }
}

kernel void md5salt_probe(
    device const uint8_t    *words        [[buffer(0)]],
    device const uint       *word_offsets [[buffer(1)]],
    device const ushort     *word_lens    [[buffer(2)]],
    device const uint8_t    *salts        [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens    [[buffer(5)]],
    device const uint       *compact_fp   [[buffer(6)]],
    device const uint       *compact_idx  [[buffer(7)]],
    constant MetalParams    &params       [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits          [[buffer(12)]],
    device atomic_uint      *hit_count     [[buffer(13)]],
    uint                     tid           [[thread_position_in_grid]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    /* Step 1: Read word */
    uint woff = word_offsets[word_idx];
    int wlen = word_lens[word_idx];
    uint8_t word[256];
    for (int i = 0; i < wlen && i < 255; i++)
        word[i] = words[woff + i];

    /* Step 2: MD5(word) */
    uint4 h1;
    if (wlen <= 55)
        md5_short(word, wlen, h1);
    else
        md5_two(word, wlen, h1);

    /* Step 3: Hex-encode to 32 chars */
    uint8_t hexbuf[128];
    thread uint8_t *bin = (thread uint8_t *)&h1;
    hex_encode(bin, hexbuf);

    /* Step 4: Append salt */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    for (int i = 0; i < slen && i < 90; i++)
        hexbuf[32 + i] = salts[soff + i];
    int total_len = 32 + slen;

    /* Step 5: MD5(hex + salt) */
    uint4 h2;
    if (total_len <= 55)
        md5_short(hexbuf, total_len, h2);
    else
        md5_two(hexbuf, total_len, h2);

    /* Step 6: Compact table probe */
    thread uint8_t *hashbytes = (thread uint8_t *)&h2;
    uint64_t key = *(thread const uint64_t *)hashbytes;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) return;  /* empty slot = miss */
        if (cfp == fp) {
            /* fingerprint match — compare full hash */
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                int hlen = hash_data_len[idx];
                if (hlen > 16) hlen = 16;
                uint64_t off = hash_data_off[idx];
                bool match = true;
                for (int i = 0; i < hlen; i++) {
                    if (hashbytes[i] != hash_data_buf[off + i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < 65536) {
                        hits[slot * 2]     = word_idx;
                        hits[slot * 2 + 1] = salt_idx;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Max probe exhausted — possible overflow. Report as hit for CPU validation. */
    {
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (slot < params.max_hits) {
            uint base = slot * 5;
            hits[base]     = tid;
            hits[base + 1] = h2.x;
            hits[base + 2] = h2.y;
            hits[base + 3] = h2.z;
            hits[base + 4] = h2.w;
        }
    }
}

/* Second kernel: pre-hashed input. "words" buffer contains the 32-char hex hash
 * already computed by CPU. GPU just appends each salt and does one MD5 + probe.
 * num_words is always 1, num_salts GPU threads. */
kernel void md5salt_salts_only(
    device const uint8_t    *hexhash      [[buffer(0)]],
    device const uint       *word_offsets [[buffer(1)]],
    device const ushort     *word_lens    [[buffer(2)]],
    device const uint8_t    *salts        [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens    [[buffer(5)]],
    device const uint       *compact_fp   [[buffer(6)]],
    device const uint       *compact_idx  [[buffer(7)]],
    constant MetalParams    &params       [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits          [[buffer(12)]],
    device atomic_uint      *hit_count     [[buffer(13)]],
    uint                     tid           [[thread_position_in_grid]])
{
    if (tid >= params.num_salts) return;

    /* Copy pre-computed 32-char hex hash */
    uint8_t buf[128];
    int hlen = word_lens[0];
    for (int i = 0; i < hlen && i < 55; i++)
        buf[i] = hexhash[i];

    /* Append salt */
    uint soff = salt_offsets[tid];
    int slen = salt_lens[tid];
    for (int i = 0; i < slen && i < 90; i++)
        buf[hlen + i] = salts[soff + i];
    int total_len = hlen + slen;

    /* MD5(hex + salt) */
    uint4 h2;
    if (total_len <= 55)
        md5_short(buf, total_len, h2);
    else
        md5_two(buf, total_len, h2);

    /* Compact table probe */
    thread uint8_t *hashbytes = (thread uint8_t *)&h2;
    uint64_t key = *(thread const uint64_t *)hashbytes;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) return;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                int dlen = hash_data_len[idx];
                if (dlen > 16) dlen = 16;
                uint64_t off = hash_data_off[idx];
                bool match = true;
                for (int i = 0; i < dlen; i++) {
                    if (hashbytes[i] != hash_data_buf[off + i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 5;
                        hits[base]     = tid;
                        hits[base + 1] = h2.x;
                        hits[base + 2] = h2.y;
                        hits[base + 3] = h2.z;
                        hits[base + 4] = h2.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Max probe exhausted — possible overflow. Report as hit for CPU validation. */
    {
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (slot < params.max_hits) {
            uint base = slot * 5;
            hits[base]     = tid;
            hits[base + 1] = h2.x;
            hits[base + 2] = h2.y;
            hits[base + 3] = h2.z;
            hits[base + 4] = h2.w;
        }
    }
}

/* Third kernel: batch of pre-hashed words × all salts.
 * Grid: num_words * num_salts threads.
 * hexhashes buffer: 256 bytes per word, packed contiguously. */
kernel void md5salt_batch_prehashed(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *salts       [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens   [[buffer(5)]],
    device const uint       *compact_fp  [[buffer(6)]],
    device const uint       *compact_idx [[buffer(7)]],
    constant MetalParams    &params      [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits         [[buffer(12)]],
    device atomic_uint      *hit_count    [[buffer(13)]],
    device const uint64_t   *overflow_keys   [[buffer(14)]],
    device const uint8_t    *overflow_hashes [[buffer(15)]],
    device const uint       *overflow_offsets [[buffer(16)]],
    device const ushort     *overflow_lengths [[buffer(17)]],
    uint                     tid          [[thread_position_in_grid]],
    uint                     lid          [[thread_position_in_threadgroup]],
    uint                     tgsize       [[threads_per_threadgroup]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    /* Shared prestate: thread 0 computes MD5 rounds 0-7 for its word.
     * Other threads use it ONLY if they share the same word_idx.
     * At threadgroup boundaries (num_salts % tgsize != 0), some threads
     * may have a different word_idx and must compute their own. */
    threadgroup uint shared_M[8];
    threadgroup uint4 shared_prestate;
    threadgroup uint shared_word_idx;

    if (lid == 0) {
        shared_word_idx = word_idx;
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            shared_M[i] = mwords[i];

        uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;
        for (int i = 0; i < 8; i++) {
            uint f = (b & c) | (~b & d);
            f = f + a + K[i] + shared_M[i];
            a = d; d = c; c = b;
            b = b + ((f << S[i]) | (f >> (32 - S[i])));
        }
        shared_prestate = uint4(a, b, c, d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Load M[0..7] — use shared if same word, else load own */
    uint M[16];
    bool use_shared = (word_idx == shared_word_idx);
    if (use_shared) {
        for (int i = 0; i < 8; i++)
            M[i] = shared_M[i];
    } else {
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            M[i] = mwords[i];
    }

    /* Append salt into M[8..] at byte position 32 */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;

    uint4 h2;
    if (slen <= 23) {
        /* Single block: fill M[8..15] with salt + padding + bitlen. */
        for (int i = 8; i < 16; i++) M[i] = 0;
        thread uint8_t *mbytes = (thread uint8_t *)M;
        for (int i = 0; i < slen; i++)
            mbytes[32 + i] = salts[soff + i];
        mbytes[total_len] = 0x80;
        M[14] = total_len * 8;
        if (use_shared) {
            h2 = shared_prestate;
            md5_block_from8(h2, M);
        } else {
            /* Full md5_block for boundary threads */
            h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
            md5_block(h2, M);
        }
    } else {
        /* Two-block: fall back to full computation */
        uint8_t buf[128];
        thread uint *buf32 = (thread uint *)buf;
        for (int i = 0; i < 8; i++)
            buf32[i] = M[i];
        for (int i = 0; i < slen && i < 90; i++)
            buf[32 + i] = salts[soff + i];
        if (total_len <= 55)
            md5_short(buf, total_len, h2);
        else
            md5_two(buf, total_len, h2);
    }

    /* Compact table probe — use wide compares */
    uint64_t key = ((thread const uint64_t *)&h2)[0];
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                uint64_t off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h2.x == ref[0] && h2.y == ref[1] &&
                    h2.z == ref[2] && h2.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Binary search the overflow table */
    if (params.overflow_count == 0) return;
    {
        int lo = 0, hi = (int)params.overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            uint64_t mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                /* Key match — compare full hash as 4 uint32 */
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                bool match = (h2.x == oref[0] && h2.y == oref[1] &&
                              h2.z == oref[2] && h2.w == oref[3]);
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
                /* Key matched but hash didn't — search duplicates both directions */
                for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                for (int d = mid + 1; d < (int)params.overflow_count && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                return;
            }
        }
    }
}

/* Iterating batch kernel: MD5(hex+salt), then MD5(hex(result)) for each iteration.
 * Identical setup to md5salt_batch_prehashed, but probes at each iteration. */
kernel void md5salt_batch_iter(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *salts       [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens   [[buffer(5)]],
    device const uint       *compact_fp  [[buffer(6)]],
    device const uint       *compact_idx [[buffer(7)]],
    constant MetalParams    &params      [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits         [[buffer(12)]],
    device atomic_uint      *hit_count    [[buffer(13)]],
    device const uint64_t   *overflow_keys   [[buffer(14)]],
    device const uint8_t    *overflow_hashes [[buffer(15)]],
    device const uint       *overflow_offsets [[buffer(16)]],
    device const ushort     *overflow_lengths [[buffer(17)]],
    uint                     tid          [[thread_position_in_grid]],
    uint                     lid          [[thread_position_in_threadgroup]],
    uint                     tgsize       [[threads_per_threadgroup]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    threadgroup uint shared_M[8];
    threadgroup uint4 shared_prestate;
    threadgroup uint shared_word_idx;

    if (lid == 0) {
        shared_word_idx = word_idx;
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            shared_M[i] = mwords[i];

        uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;
        for (int i = 0; i < 8; i++) {
            uint f = (b & c) | (~b & d);
            f = f + a + K[i] + shared_M[i];
            a = d; d = c; c = b;
            b = b + ((f << S[i]) | (f >> (32 - S[i])));
        }
        shared_prestate = uint4(a, b, c, d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint M[16];
    bool use_shared = (word_idx == shared_word_idx);
    if (use_shared) {
        for (int i = 0; i < 8; i++)
            M[i] = shared_M[i];
    } else {
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 8; i++)
            M[i] = mwords[i];
    }

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;

    uint4 h2;
    if (slen <= 23) {
        for (int i = 8; i < 16; i++) M[i] = 0;
        thread uint8_t *mbytes = (thread uint8_t *)M;
        for (int i = 0; i < slen; i++)
            mbytes[32 + i] = salts[soff + i];
        mbytes[total_len] = 0x80;
        M[14] = total_len * 8;
        if (use_shared) {
            h2 = shared_prestate;
            md5_block_from8(h2, M);
        } else {
            h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
            md5_block(h2, M);
        }
    } else {
        uint8_t buf[128];
        thread uint *buf32 = (thread uint *)buf;
        for (int i = 0; i < 8; i++)
            buf32[i] = M[i];
        for (int i = 0; i < slen && i < 90; i++)
            buf[32 + i] = salts[soff + i];
        if (total_len <= 55)
            md5_short(buf, total_len, h2);
        else
            md5_two(buf, total_len, h2);
    }

    /* Iterate: probe at each iteration, then MD5(hex(h2)) for next */
    for (uint iter = 0; iter < params.max_iter; iter++) {
        /* If not first iteration, compute MD5(hex(h2)) */
        if (iter > 0) {
            uint Mi[16];
            hash_to_hex_M(h2, Mi);
            Mi[8] = 0x80;
            Mi[9] = 0; Mi[10] = 0; Mi[11] = 0;
            Mi[12] = 0; Mi[13] = 0;
            Mi[14] = 256; /* 32 * 8 bits */
            Mi[15] = 0;
            h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
            md5_block(h2, Mi);
        }

        /* Compact table probe — record hit but continue iterating */
        uint64_t key = ((thread const uint64_t *)&h2)[0];
        uint fp = (uint)(key >> 32);
        if (fp == 0) fp = 1;
        uint64_t pos = compact_mix(key) & params.compact_mask;
        bool found = false;

        for (int p = 0; p < (int)params.max_probe; p++) {
            uint cfp = compact_fp[pos];
            if (cfp == 0) break;
            if (cfp == fp) {
                uint idx = compact_idx[pos];
                if (idx < params.hash_data_count) {
                    uint64_t off = hash_data_off[idx];
                    device const uint *ref = (device const uint *)(hash_data_buf + off);
                    if (h2.x == ref[0] && h2.y == ref[1] &&
                        h2.z == ref[2] && h2.w == ref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 7;
                            hits[base]     = word_idx;
                            hits[base + 1] = salt_idx;
                            hits[base + 2] = iter + 1;
                            hits[base + 3] = h2.x;
                            hits[base + 4] = h2.y;
                            hits[base + 5] = h2.z;
                            hits[base + 6] = h2.w;
                        }
                        found = true;
                        break;
                    }
                }
            }
            pos = (pos + 1) & params.compact_mask;
        }

        /* Binary search overflow */
        if (!found && params.overflow_count > 0) {
            int lo = 0, hi = (int)params.overflow_count - 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                uint64_t mkey = overflow_keys[mid];
                if (key < mkey) hi = mid - 1;
                else if (key > mkey) lo = mid + 1;
                else {
                    uint ooff = overflow_offsets[mid];
                    device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 7;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = iter + 1;
                            hits[base+3] = h2.x; hits[base+4] = h2.y;
                            hits[base+5] = h2.z; hits[base+6] = h2.w;
                        }
                        break;
                    }
                    for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                        oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h2.x == oref[0] && h2.y == oref[1] &&
                            h2.z == oref[2] && h2.w == oref[3]) {
                            uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                            if (slot < params.max_hits) {
                                uint base = slot * 7;
                                hits[base] = word_idx; hits[base+1] = salt_idx;
                                hits[base+2] = iter + 1;
                                hits[base+3] = h2.x; hits[base+4] = h2.y;
                                hits[base+5] = h2.z; hits[base+6] = h2.w;
                            }
                            found = true; break;
                        }
                    }
                    if (!found) {
                        for (int d = mid + 1; d < (int)params.overflow_count && overflow_keys[d] == key; d++) {
                            oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                            if (h2.x == oref[0] && h2.y == oref[1] &&
                                h2.z == oref[2] && h2.w == oref[3]) {
                                uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                                if (slot < params.max_hits) {
                                    uint base = slot * 7;
                                    hits[base] = word_idx; hits[base+1] = salt_idx;
                                    hits[base+2] = iter + 1;
                                    hits[base+3] = h2.x; hits[base+4] = h2.y;
                                    hits[base+5] = h2.z; hits[base+6] = h2.w;
                                }
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
    } /* end iteration loop */
}

/* Sub8-24 batch kernel: hexlen=16 (chars 8-23 of MD5 hex), 4-word prestate.
 * Identical buffer layout to batch_prehashed, but constant 16-byte hex input. */
kernel void md5salt_batch_sub8_24(
    device const uint8_t    *hexhashes   [[buffer(0)]],
    device const ushort     *hex_lens    [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *salts       [[buffer(3)]],
    device const uint       *salt_offsets [[buffer(4)]],
    device const ushort     *salt_lens   [[buffer(5)]],
    device const uint       *compact_fp  [[buffer(6)]],
    device const uint       *compact_idx [[buffer(7)]],
    constant MetalParams    &params      [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits         [[buffer(12)]],
    device atomic_uint      *hit_count    [[buffer(13)]],
    device const uint64_t   *overflow_keys   [[buffer(14)]],
    device const uint8_t    *overflow_hashes [[buffer(15)]],
    device const uint       *overflow_offsets [[buffer(16)]],
    device const ushort     *overflow_lengths [[buffer(17)]],
    uint                     tid          [[thread_position_in_grid]],
    uint                     lid          [[thread_position_in_threadgroup]],
    uint                     tgsize       [[threads_per_threadgroup]])
{
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    /* Shared prestate: 4 M words, 4 prestate rounds for hexlen=16 */
    threadgroup uint shared_M[4];
    threadgroup uint4 shared_prestate;
    threadgroup uint shared_word_idx;

    if (lid == 0) {
        shared_word_idx = word_idx;
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 4; i++)
            shared_M[i] = mwords[i];

        uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;
        for (int i = 0; i < 4; i++) {
            uint f = (b & c) | (~b & d);
            f = f + a + K[i] + shared_M[i];
            a = d; d = c; c = b;
            b = b + ((f << S[i]) | (f >> (32 - S[i])));
        }
        shared_prestate = uint4(a, b, c, d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Load M[0..3] — use shared if same word, else load own */
    uint M[16];
    bool use_shared = (word_idx == shared_word_idx);
    if (use_shared) {
        for (int i = 0; i < 4; i++)
            M[i] = shared_M[i];
    } else {
        uint hoff = word_idx * 256;
        device const uint *mwords = (device const uint *)(hexhashes + hoff);
        for (int i = 0; i < 4; i++)
            M[i] = mwords[i];
    }

    /* Append salt at byte position 16 */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 16 + slen;

    uint4 h2;
    if (total_len <= 55) {
        /* Single block: fill M[4..15] with salt + padding + bitlen */
        for (int i = 4; i < 16; i++) M[i] = 0;
        thread uint8_t *mbytes = (thread uint8_t *)M;
        for (int i = 0; i < slen; i++)
            mbytes[16 + i] = salts[soff + i];
        mbytes[total_len] = 0x80;
        M[14] = total_len * 8;
        /* Use full md5_block for now — prestate optimization can be added later */
        h2 = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
        md5_block(h2, M);
    } else {
        /* Two-block: fall back to full computation */
        uint8_t buf[128];
        thread uint *buf32 = (thread uint *)buf;
        for (int i = 0; i < 4; i++)
            buf32[i] = M[i];
        for (int i = 0; i < slen && i < 106; i++)
            buf[16 + i] = salts[soff + i];
        if (total_len <= 55)
            md5_short(buf, total_len, h2);
        else
            md5_two(buf, total_len, h2);
    }

    /* Compact table probe — use wide compares */
    uint64_t key = ((thread const uint64_t *)&h2)[0];
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    uint64_t pos = compact_mix(key) & params.compact_mask;

    for (int p = 0; p < (int)params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                uint64_t off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h2.x == ref[0] && h2.y == ref[1] &&
                    h2.z == ref[2] && h2.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    /* Binary search the overflow table */
    if (params.overflow_count == 0) return;
    {
        int lo = 0, hi = (int)params.overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            uint64_t mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                bool match = (h2.x == oref[0] && h2.y == oref[1] &&
                              h2.z == oref[2] && h2.w == oref[3]);
                if (match) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base]     = word_idx;
                        hits[base + 1] = salt_idx;
                        hits[base + 2] = h2.x;
                        hits[base + 3] = h2.y;
                        hits[base + 4] = h2.z;
                        hits[base + 5] = h2.w;
                    }
                    return;
                }
                for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                for (int d = mid + 1; d < (int)params.overflow_count && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h2.x == oref[0] && h2.y == oref[1] &&
                        h2.z == oref[2] && h2.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = salt_idx;
                            hits[base+2] = h2.x; hits[base+3] = h2.y;
                            hits[base+4] = h2.z; hits[base+5] = h2.w;
                        }
                        return;
                    }
                }
                return;
            }
        }
    }
}
)MSL";

/* ---- Helper: wrap buffer with zero-copy if page-aligned, else copy ---- */
static id<MTLBuffer> wrap_buffer(void *ptr, size_t size) {
    if (size == 0) size = 4;
    /* Always copy for safety in PoC — zero-copy optimization later */
    return [mtl_device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
}

/* ---- Public API ---- */

int metal_md5salt_init(void) {
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

        /* Compile kernel from embedded source */
        NSError *error = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        id<MTLLibrary> lib = [mtl_device newLibraryWithSource:kernel_source
                                                      options:opts
                                                        error:&error];
        if (!lib) {
            fprintf(stderr, "Metal: kernel compile error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        id<MTLFunction> func = [lib newFunctionWithName:@"md5salt_probe"];
        if (!func) {
            fprintf(stderr, "Metal: kernel function not found\n");
            return -1;
        }

        mtl_pipeline = [mtl_device newComputePipelineStateWithFunction:func error:&error];
        if (!mtl_pipeline) {
            fprintf(stderr, "Metal: pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        id<MTLFunction> func2 = [lib newFunctionWithName:@"md5salt_salts_only"];
        if (func2) {
            mtl_pipeline_salts = [mtl_device newComputePipelineStateWithFunction:func2 error:&error];
        }

        id<MTLFunction> func3 = [lib newFunctionWithName:@"md5salt_batch_prehashed"];
        if (func3) {
            mtl_pipeline_batch = [mtl_device newComputePipelineStateWithFunction:func3 error:&error];
        }

        id<MTLFunction> func4 = [lib newFunctionWithName:@"md5salt_batch_iter"];
        if (func4) {
            mtl_pipeline_iter = [mtl_device newComputePipelineStateWithFunction:func4 error:&error];
        }

        id<MTLFunction> func5 = [lib newFunctionWithName:@"md5salt_batch_sub8_24"];
        if (func5) {
            mtl_pipeline_sub8 = [mtl_device newComputePipelineStateWithFunction:func5 error:&error];
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

void metal_md5salt_shutdown(void) {
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

int metal_md5salt_available(void) {
    return mtl_ready;
}

int metal_md5salt_set_compact_table(
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

int metal_md5salt_set_salts(
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

        _dispatch_bufs_ready = 1;
        return 0;
    }
}

uint32_t *metal_md5salt_probe_salts(
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

int metal_md5salt_set_overflow(
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

void metal_md5salt_set_max_iter(int max_iter) {
    _max_iter = (max_iter < 1) ? 1 : max_iter;
}

void metal_md5salt_set_op(int op) {
    _gpu_op = op;
}

uint32_t *metal_md5salt_dispatch_batch(
    const char *hexhashes,
    const uint16_t *hexlens,
    int num_words,
    int *nhits_out)
{
    @autoreleasepool {
        *nhits_out = 0;
        if (!mtl_ready || !_dispatch_bufs_ready || !mtl_pipeline_batch) return NULL;
        if (num_words <= 0 || _salts_count <= 0) return NULL;

        /* Create word data buffer: 256 bytes per word, packed */
        size_t words_size = (size_t)num_words * 256;
        id<MTLBuffer> b_hexhashes = [mtl_device newBufferWithBytes:hexhashes
                                                            length:words_size
                                                           options:MTLResourceStorageModeShared];
        /* Convert int hexlens to uint16_t for GPU */
        id<MTLBuffer> b_hexlens = [mtl_device newBufferWithBytes:hexlens
                                                          length:num_words * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];

        /* Params */
        MetalParams params;
        params.compact_mask = _compact_mask;
        params.num_words = num_words;
        params.num_salts = _salts_count;
        params.max_probe = 256;
        params.hash_data_count = _hash_data_count;
        params.max_hits = _max_hits;
        params.overflow_count = _overflow_count;
        params.max_iter = _max_iter;

        id<MTLBuffer> b_params = [mtl_device newBufferWithBytes:&params
                                                         length:sizeof(params)
                                                        options:MTLResourceStorageModeShared];

        /* Zero hit counter */
        *(uint32_t *)[buf_dispatch_hit_count contents] = 0;

        /* Encode and dispatch — use iter kernel when max_iter > 1 */
        id<MTLCommandBuffer> cmdbuf = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        id<MTLComputePipelineState> pipeline =
            (_gpu_op == 542 && mtl_pipeline_sub8) ? mtl_pipeline_sub8 :
            (_max_iter > 1 && mtl_pipeline_iter) ? mtl_pipeline_iter : mtl_pipeline_batch;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:b_hexhashes       offset:0 atIndex:0];
        [enc setBuffer:b_hexlens         offset:0 atIndex:1];
        [enc setBuffer:b_hexlens         offset:0 atIndex:2]; /* unused slot */
        [enc setBuffer:buf_salt_data     offset:0 atIndex:3];
        [enc setBuffer:buf_salt_off      offset:0 atIndex:4];
        [enc setBuffer:buf_salt_len      offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp    offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx   offset:0 atIndex:7];
        [enc setBuffer:b_params          offset:0 atIndex:8];
        [enc setBuffer:buf_hash_data     offset:0 atIndex:9];
        [enc setBuffer:buf_hash_data_off offset:0 atIndex:10];
        [enc setBuffer:buf_hash_data_len offset:0 atIndex:11];
        [enc setBuffer:buf_dispatch_hits offset:0 atIndex:12];
        [enc setBuffer:buf_dispatch_hit_count offset:0 atIndex:13];
        /* Overflow buffers — use dummy 4-byte buffers if no overflow */
        if (buf_overflow_keys) {
            [enc setBuffer:buf_overflow_keys    offset:0 atIndex:14];
            [enc setBuffer:buf_overflow_hashes  offset:0 atIndex:15];
            [enc setBuffer:buf_overflow_offsets  offset:0 atIndex:16];
            [enc setBuffer:buf_overflow_lengths  offset:0 atIndex:17];
        } else {
            id<MTLBuffer> dummy = [mtl_device newBufferWithLength:4
                                                          options:MTLResourceStorageModeShared];
            [enc setBuffer:dummy offset:0 atIndex:14];
            [enc setBuffer:dummy offset:0 atIndex:15];
            [enc setBuffer:dummy offset:0 atIndex:16];
            [enc setBuffer:dummy offset:0 atIndex:17];
        }

        uint64_t total_threads = (uint64_t)num_words * _salts_count;
        NSUInteger tpg = [pipeline maxTotalThreadsPerThreadgroup];
        if (tpg > 256) tpg = 256;
        MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
        MTLSize groupSize = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        /* Read back — return raw count and pointer */
        uint32_t raw_nhits = *(uint32_t *)[buf_dispatch_hit_count contents];
        *nhits_out = (int)raw_nhits;

        /* Release transient buffers */
        cmdbuf = nil; enc = nil;
        b_hexhashes = nil; b_hexlens = nil; b_params = nil;

        return (uint32_t *)[buf_dispatch_hits contents];
    }
}

/* ---- Double-buffer slot API ---- */

int metal_md5salt_init_slots(int max_salt_count, int max_salt_bytes) {
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
            if (!gpu_slots[i].buf_hexhashes || !gpu_slots[i].buf_salt_data ||
                !gpu_slots[i].buf_hits) return -1;
        }
        gpu_slots_ready = 1;
        fprintf(stderr, "Metal GPU: %d double-buffer slots initialized (%d max salts, %d max bytes)\n",
                GPU_NUM_SLOTS, max_salt_count, max_salt_bytes);
        return 0;
    }
}

int metal_md5salt_submit_slot(int slot,
    const char *hexhashes, const uint16_t *hexlens, int num_words,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts)
{
    @autoreleasepool {
        if (!gpu_slots_ready || !mtl_pipeline_batch) return -1;
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
        p->max_probe = 256;
        p->hash_data_count = _hash_data_count;
        p->max_hits = GPU_SLOT_MAX_HITS;
        p->overflow_count = _overflow_count;

        *(uint32_t *)[s->buf_hit_count contents] = 0;

        id<MTLCommandBuffer> cmdbuf = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        [enc setComputePipelineState:mtl_pipeline_batch];
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
            id<MTLBuffer> dummy = [mtl_device newBufferWithLength:4
                                                          options:MTLResourceStorageModeShared];
            [enc setBuffer:dummy offset:0 atIndex:14];
            [enc setBuffer:dummy offset:0 atIndex:15];
            [enc setBuffer:dummy offset:0 atIndex:16];
            [enc setBuffer:dummy offset:0 atIndex:17];
        }

        uint64_t total_threads = (uint64_t)num_words * num_salts;
        NSUInteger tpg = [mtl_pipeline_batch maxTotalThreadsPerThreadgroup];
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

uint32_t *metal_md5salt_wait_slot(int slot, int *nhits_out) {
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
