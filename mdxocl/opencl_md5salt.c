/*
 * opencl_md5salt.c — OpenCL GPU acceleration for MD5SALT
 *
 * Cross-vendor: NVIDIA, AMD, Intel, Apple (via OpenCL compatibility).
 * Kernel source compiled at runtime via clCreateProgramWithSource.
 */

#if defined(OPENCL_GPU)

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "opencl_dynload.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "opencl_md5salt.h"
#include "job_types.h"
#include "gpujob.h"

/* ---- Multi-GPU device state ---- */
#define MAX_GPU_DEVICES 64

struct gpu_device {
    cl_context       ctx;
    cl_command_queue  queue;
    cl_program        prog;
    cl_device_id      dev;
    cl_kernel         kern_batch;
    cl_kernel         kern_iter;
    cl_kernel         kern_sub8;
    cl_kernel         kern_md5_iter_lc;
    cl_kernel         kern_md5_iter_uc;
    char              name[256];

    /* Compact table (read-only, uploaded once) */
    cl_mem b_compact_fp, b_compact_idx;
    cl_mem b_hash_data, b_hash_data_off, b_hash_data_len;

    /* Salt buffers (updated per-snapshot) */
    cl_mem b_salt_data, b_salt_off, b_salt_len;
    size_t salt_data_cap;
    int salts_count;

    /* Overflow (uploaded once) */
    cl_mem b_overflow_keys, b_overflow_hashes, b_overflow_offsets, b_overflow_lengths;

    /* Per-dispatch buffers */
    cl_mem b_hits, b_hit_count;
    cl_mem b_hexhashes, b_hexlens, b_params;
    size_t hexhash_cap;
    uint32_t *h_hits;    /* host-side hit buffer */
};

static struct gpu_device gpu_devs[MAX_GPU_DEVICES];
static int num_gpu_devs = 0;
static int ocl_ready = 0;

/* Shared state (same across all devices) */
static uint64_t _compact_mask = 0;
static uint32_t _hash_data_count = 0;
static int _overflow_count = 0;
static int _max_iter = 1;
static int _gpu_op = 0;

/* ---- Per-kernel autotune state ---- */
#define TUNE_CANDIDATES 4
static const size_t tune_sizes[TUNE_CANDIDATES] = { 64, 128, 256, 512 };

#define MAX_GPU_KERNELS 2048   /* indexed by op type, same as JOB_DONE */

struct gpu_kern {
    cl_kernel kernel;        /* NULL if no GPU kernel for this op */
    size_t local_size;       /* current work group size */
    size_t max_local;        /* from CL_KERNEL_WORK_GROUP_SIZE */
    int tuned;               /* 1 = done tuning */
    int tune_candidate;      /* which tune_sizes[] we're testing */
    int tune_samples;        /* dispatches at current candidate */
    double tune_best_time;   /* best avg ms so far */
    size_t tune_best_size;   /* size that produced best_time */
    double tune_cur_total;   /* accumulated ms for current candidate */
    int dev_idx;             /* device index for status messages */
};

/* Per-device kernel table — each device has its own compiled kernels */
struct gpu_kern_table {
    struct gpu_kern kerns[MAX_GPU_KERNELS];
};
static struct gpu_kern_table dev_kerns[MAX_GPU_DEVICES];

#define TUNE_SAMPLES 3  /* dispatches per candidate before moving on */

static void kern_register(int di, int op, cl_kernel kernel) {
    if (di < 0 || di >= MAX_GPU_DEVICES) return;
    if (op < 0 || op >= MAX_GPU_KERNELS || !kernel) return;
    struct gpu_kern *k = &dev_kerns[di].kerns[op];
    size_t max_wg = 0;
    clGetKernelWorkGroupInfo(kernel, gpu_devs[di].dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    k->kernel = kernel;
    k->max_local = max_wg;
    k->dev_idx = di;
    k->tuned = 0;
    k->tune_candidate = 0;
    k->tune_samples = 0;
    k->tune_best_time = 1e30;
    k->tune_best_size = 64;
    k->tune_cur_total = 0;
    int c = 0;
    while (c < TUNE_CANDIDATES && tune_sizes[c] > max_wg) c++;
    k->tune_candidate = c;
    k->local_size = (c < TUNE_CANDIDATES) ? tune_sizes[c] : max_wg;
}

static size_t kern_get_local_size(struct gpu_kern *k) {
    if (k->tuned) return k->local_size;
    if (k->tune_candidate >= TUNE_CANDIDATES) {
        k->tuned = 1;
        k->local_size = k->tune_best_size;
        fprintf(stderr, "OpenCL GPU[%d]: autotuned work group size = %zu\n", k->dev_idx, k->local_size);
        return k->local_size;
    }
    return tune_sizes[k->tune_candidate];
}

static void kern_record_time(struct gpu_kern *k, double ms) {
    if (k->tuned) return;
    k->tune_cur_total += ms;
    k->tune_samples++;
    if (k->tune_samples >= TUNE_SAMPLES) {
        double avg = k->tune_cur_total / k->tune_samples;
        if (avg < k->tune_best_time) {
            k->tune_best_time = avg;
            k->tune_best_size = tune_sizes[k->tune_candidate];
        }
        k->tune_candidate++;
        while (k->tune_candidate < TUNE_CANDIDATES && tune_sizes[k->tune_candidate] > k->max_local)
            k->tune_candidate++;
        k->tune_samples = 0;
        k->tune_cur_total = 0;
        if (k->tune_candidate >= TUNE_CANDIDATES) {
            k->tuned = 1;
            k->local_size = k->tune_best_size;
            fprintf(stderr, "OpenCL GPU[%d]: autotuned work group size = %zu\n", k->dev_idx, k->local_size);
        }
    }
}

#define GPU_MAX_HITS 32768

static uint32_t *h_hits = NULL;

/* ---- Params struct (must match kernel) ---- */
typedef struct {
    uint64_t compact_mask;
    uint32_t num_words;
    uint32_t num_salts;
    uint32_t max_probe;
    uint32_t hash_data_count;
    uint32_t max_hits;
    uint32_t overflow_count;
    uint32_t max_iter;
} OCLParams;

/* ---- Load kernel source from file ---- */
static char *load_kernel_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        /* Try mdxocl/ prefix */
        char buf[512];
        snprintf(buf, sizeof(buf), "mdxocl/%s", path);
        f = fopen(buf, "r");
    }
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char *)malloc(sz + 1);
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);
    return src;
}

/* ---- Embedded kernel source (auto-generated from md5salt.cl) ---- */
#include "md5salt_kernel_str.h"
static const char *kernel_source_embedded_unused =
"typedef struct {\n"
"    ulong compact_mask;\n"
"    uint num_words;\n"
"    uint num_salts;\n"
"    uint max_probe;\n"
"    uint hash_data_count;\n"
"    uint max_hits;\n"
"    uint overflow_count;\n"
"    uint max_iter;\n"
"} OCLParams;\n"
"\n"
"__constant uint K[64] = {\n"
"    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,\n"
"    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,\n"
"    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,\n"
"    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,\n"
"    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,\n"
"    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,\n"
"    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,\n"
"    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391\n"
"};\n"
"__constant uint S[64] = {\n"
"    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,\n"
"    5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,\n"
"    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,\n"
"    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21\n"
"};\n"
"\n"
"#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + rotate(a,s); }\n"
"#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + rotate(a,s); }\n"
"#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + rotate(a,s); }\n"
"#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + rotate(a,s); }\n"
"\n"
"void md5_block(uint *h0, uint *h1, uint *h2, uint *h3, uint *M) {\n"
"    uint a = *h0, b = *h1, c = *h2, d = *h3;\n"
"    FF(a,b,c,d,M[0],(uint)7,0xd76aa478u);  FF(d,a,b,c,M[1],(uint)12,0xe8c7b756u);\n"
"    FF(c,d,a,b,M[2],(uint)17,0x242070dbu);  FF(b,c,d,a,M[3],(uint)22,0xc1bdceeeu);\n"
"    FF(a,b,c,d,M[4],(uint)7,0xf57c0fafu);   FF(d,a,b,c,M[5],(uint)12,0x4787c62au);\n"
"    FF(c,d,a,b,M[6],(uint)17,0xa8304613u);  FF(b,c,d,a,M[7],(uint)22,0xfd469501u);\n"
"    FF(a,b,c,d,M[8],(uint)7,0x698098d8u);   FF(d,a,b,c,M[9],(uint)12,0x8b44f7afu);\n"
"    FF(c,d,a,b,M[10],(uint)17,0xffff5bb1u); FF(b,c,d,a,M[11],(uint)22,0x895cd7beu);\n"
"    FF(a,b,c,d,M[12],(uint)7,0x6b901122u);  FF(d,a,b,c,M[13],(uint)12,0xfd987193u);\n"
"    FF(c,d,a,b,M[14],(uint)17,0xa679438eu); FF(b,c,d,a,M[15],(uint)22,0x49b40821u);\n"
"    GG(a,b,c,d,M[1],(uint)5,0xf61e2562u);   GG(d,a,b,c,M[6],(uint)9,0xc040b340u);\n"
"    GG(c,d,a,b,M[11],(uint)14,0x265e5a51u); GG(b,c,d,a,M[0],(uint)20,0xe9b6c7aau);\n"
"    GG(a,b,c,d,M[5],(uint)5,0xd62f105du);   GG(d,a,b,c,M[10],(uint)9,0x02441453u);\n"
"    GG(c,d,a,b,M[15],(uint)14,0xd8a1e681u); GG(b,c,d,a,M[4],(uint)20,0xe7d3fbc8u);\n"
"    GG(a,b,c,d,M[9],(uint)5,0x21e1cde6u);   GG(d,a,b,c,M[14],(uint)9,0xc33707d6u);\n"
"    GG(c,d,a,b,M[3],(uint)14,0xf4d50d87u);  GG(b,c,d,a,M[8],(uint)20,0x455a14edu);\n"
"    GG(a,b,c,d,M[13],(uint)5,0xa9e3e905u);  GG(d,a,b,c,M[2],(uint)9,0xfcefa3f8u);\n"
"    GG(c,d,a,b,M[7],(uint)14,0x676f02d9u);  GG(b,c,d,a,M[12],(uint)20,0x8d2a4c8au);\n"
"    HH(a,b,c,d,M[5],(uint)4,0xfffa3942u);   HH(d,a,b,c,M[8],(uint)11,0x8771f681u);\n"
"    HH(c,d,a,b,M[11],(uint)16,0x6d9d6122u); HH(b,c,d,a,M[14],(uint)23,0xfde5380cu);\n"
"    HH(a,b,c,d,M[1],(uint)4,0xa4beea44u);   HH(d,a,b,c,M[4],(uint)11,0x4bdecfa9u);\n"
"    HH(c,d,a,b,M[7],(uint)16,0xf6bb4b60u);  HH(b,c,d,a,M[10],(uint)23,0xbebfbc70u);\n"
"    HH(a,b,c,d,M[13],(uint)4,0x289b7ec6u);  HH(d,a,b,c,M[0],(uint)11,0xeaa127fau);\n"
"    HH(c,d,a,b,M[3],(uint)16,0xd4ef3085u);  HH(b,c,d,a,M[6],(uint)23,0x04881d05u);\n"
"    HH(a,b,c,d,M[9],(uint)4,0xd9d4d039u);   HH(d,a,b,c,M[12],(uint)11,0xe6db99e5u);\n"
"    HH(c,d,a,b,M[15],(uint)16,0x1fa27cf8u); HH(b,c,d,a,M[2],(uint)23,0xc4ac5665u);\n"
"    II(a,b,c,d,M[0],(uint)6,0xf4292244u);   II(d,a,b,c,M[7],(uint)10,0x432aff97u);\n"
"    II(c,d,a,b,M[14],(uint)15,0xab9423a7u); II(b,c,d,a,M[5],(uint)21,0xfc93a039u);\n"
"    II(a,b,c,d,M[12],(uint)6,0x655b59c3u);  II(d,a,b,c,M[3],(uint)10,0x8f0ccc92u);\n"
"    II(c,d,a,b,M[10],(uint)15,0xffeff47du); II(b,c,d,a,M[1],(uint)21,0x85845dd1u);\n"
"    II(a,b,c,d,M[8],(uint)6,0x6fa87e4fu);   II(d,a,b,c,M[15],(uint)10,0xfe2ce6e0u);\n"
"    II(c,d,a,b,M[6],(uint)15,0xa3014314u);  II(b,c,d,a,M[13],(uint)21,0x4e0811a1u);\n"
"    II(a,b,c,d,M[4],(uint)6,0xf7537e82u);   II(d,a,b,c,M[11],(uint)10,0xbd3af235u);\n"
"    II(c,d,a,b,M[2],(uint)15,0x2ad7d2bbu);  II(b,c,d,a,M[9],(uint)21,0xeb86d391u);\n"
"    *h0 += a; *h1 += b; *h2 += c; *h3 += d;\n"
"}\n"
"\n"
"ulong compact_mix(ulong k) {\n"
"    k ^= k >> 33; k *= 0xff51afd7ed558ccdUL;\n"
"    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53UL;\n"
"    k ^= k >> 33; return k;\n"
"}\n"
"\n"
"int probe_compact(uint hx, uint hy, uint hz, uint hw,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    ulong compact_mask, uint max_probe, uint hash_data_count,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, uint overflow_count)\n"
"{\n"
"    ulong key = ((ulong)hy << 32) | hx;\n"
"    uint fp = (uint)(key >> 32);\n"
"    if (fp == 0) fp = 1;\n"
"    ulong pos = compact_mix(key) & compact_mask;\n"
"    for (int p = 0; p < (int)max_probe; p++) {\n"
"        uint cfp = compact_fp[pos];\n"
"        if (cfp == 0) break;\n"
"        if (cfp == fp) {\n"
"            uint idx = compact_idx[pos];\n"
"            if (idx < hash_data_count) {\n"
"                ulong off = hash_data_off[idx];\n"
"                __global const uint *ref = (__global const uint *)(hash_data_buf + off);\n"
"                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3])\n"
"                    return 1;\n"
"            }\n"
"        }\n"
"        pos = (pos + 1) & compact_mask;\n"
"    }\n"
"    if (overflow_count > 0) {\n"
"        int lo = 0, hi = (int)overflow_count - 1;\n"
"        while (lo <= hi) {\n"
"            int mid = (lo + hi) / 2;\n"
"            ulong mkey = overflow_keys[mid];\n"
"            if (key < mkey) hi = mid - 1;\n"
"            else if (key > mkey) lo = mid + 1;\n"
"            else {\n"
"                uint ooff = overflow_offsets[mid];\n"
"                __global const uint *oref = (__global const uint *)(overflow_hashes + ooff);\n"
"                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;\n"
"                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {\n"
"                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);\n"
"                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;\n"
"                }\n"
"                for (int d = mid+1; d < (int)overflow_count && overflow_keys[d] == key; d++) {\n"
"                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);\n"
"                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;\n"
"                }\n"
"                break;\n"
"            }\n"
"        }\n"
"    }\n"
"    return 0;\n"
"}\n"
"\n"
"__kernel void md5salt_batch(\n"
"    __global const uchar *hexhashes, __global const ushort *hexlens,\n"
"    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    __global const OCLParams *params_buf,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,\n"
"    __global uint *hits, __global volatile uint *hit_count,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)\n"
"{\n"
"    OCLParams params = *params_buf;\n"
"    uint tid = get_global_id(0);\n"
"    uint word_idx = tid / params.num_salts;\n"
"    uint salt_idx = tid % params.num_salts;\n"
"    if (word_idx >= params.num_words) return;\n"
"\n"
"    uint M[16];\n"
"    uint hoff = word_idx * 256;\n"
"    __global const uint *mwords = (__global const uint *)(hexhashes + hoff);\n"
"    for (int i = 0; i < 8; i++) M[i] = mwords[i];\n"
"\n"
"    /* Prestate rounds 0-7 */\n"
"    uint a = 0x67452301, b = 0xEFCDAB89, c = 0x98BADCFE, d = 0x10325476;\n"
"    for (int i = 0; i < 8; i++) {\n"
"        uint f = (b & c) | (~b & d);\n"
"        f = f + a + K[i] + M[i];\n"
"        a = d; d = c; c = b;\n"
"        b = b + rotate(f, S[i]);\n"
"    }\n"
"\n"
"    uint soff = salt_offsets[salt_idx];\n"
"    int slen = salt_lens[salt_idx];\n"
"    int total_len = 32 + slen;\n"
"\n"
"    uint hx, hy, hz, hw;\n"
"    for (int i = 8; i < 16; i++) M[i] = 0;\n"
"    uchar *mbytes = (uchar *)M;\n"
"    for (int i = 0; i < slen; i++)\n"
"        mbytes[32 + i] = salts[soff + i];\n"
"    mbytes[total_len] = 0x80;\n"
"    M[14] = total_len * 8;\n"
"    /* Use full md5_block (prestate optimization deferred to tuning phase) */\n"
"    hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;\n"
"    md5_block(&hx, &hy, &hz, &hw, M);\n"
"\n"
"    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,\n"
"                      params.compact_mask, params.max_probe, params.hash_data_count,\n"
"                      hash_data_buf, hash_data_off,\n"
"                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {\n"
"        uint slot = atomic_add(hit_count, 1u);\n"
"        if (slot < params.max_hits) {\n"
"            uint base = slot * 6;\n"
"            hits[base] = word_idx; hits[base+1] = salt_idx;\n"
"            hits[base+2] = hx; hits[base+3] = hy;\n"
"            hits[base+4] = hz; hits[base+5] = hw;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__kernel void md5salt_sub8_24(\n"
"    __global const uchar *hexhashes, __global const ushort *hexlens,\n"
"    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    __global const OCLParams *params_buf,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,\n"
"    __global uint *hits, __global volatile uint *hit_count,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)\n"
"{\n"
"    OCLParams params = *params_buf;\n"
"    uint tid = get_global_id(0);\n"
"    uint word_idx = tid / params.num_salts;\n"
"    uint salt_idx = tid % params.num_salts;\n"
"    if (word_idx >= params.num_words) return;\n"
"\n"
"    uint M[16];\n"
"    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);\n"
"    for (int i = 0; i < 4; i++) M[i] = mwords[i];\n"
"    for (int i = 4; i < 16; i++) M[i] = 0;\n"
"\n"
"    uint soff = salt_offsets[salt_idx];\n"
"    int slen = salt_lens[salt_idx];\n"
"    int total_len = 16 + slen;\n"
"    uchar *mbytes = (uchar *)M;\n"
"    for (int i = 0; i < slen; i++) mbytes[16 + i] = salts[soff + i];\n"
"    mbytes[total_len] = 0x80;\n"
"    M[14] = total_len * 8;\n"
"\n"
"    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;\n"
"    md5_block(&hx, &hy, &hz, &hw, M);\n"
"\n"
"    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,\n"
"                      params.compact_mask, params.max_probe, params.hash_data_count,\n"
"                      hash_data_buf, hash_data_off,\n"
"                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {\n"
"        uint slot = atomic_add(hit_count, 1u);\n"
"        if (slot < params.max_hits) {\n"
"            uint base = slot * 6;\n"
"            hits[base] = word_idx; hits[base+1] = salt_idx;\n"
"            hits[base+2] = hx; hits[base+3] = hy;\n"
"            hits[base+4] = hz; hits[base+5] = hw;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__kernel void md5salt_iter(\n"
"    __global const uchar *hexhashes, __global const ushort *hexlens,\n"
"    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,\n"
"    __global const uint *compact_fp, __global const uint *compact_idx,\n"
"    __global const OCLParams *params_buf,\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,\n"
"    __global uint *hits, __global volatile uint *hit_count,\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,\n"
"    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)\n"
"{\n"
"    OCLParams params = *params_buf;\n"
"    uint tid = get_global_id(0);\n"
"    uint word_idx = tid / params.num_salts;\n"
"    uint salt_idx = tid % params.num_salts;\n"
"    if (word_idx >= params.num_words) return;\n"
"\n"
"    uint M[16];\n"
"    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);\n"
"    for (int i = 0; i < 8; i++) M[i] = mwords[i];\n"
"    for (int i = 8; i < 16; i++) M[i] = 0;\n"
"\n"
"    uint soff = salt_offsets[salt_idx];\n"
"    int slen = salt_lens[salt_idx];\n"
"    int total_len = 32 + slen;\n"
"    uchar *mbytes = (uchar *)M;\n"
"    for (int i = 0; i < slen; i++) mbytes[32 + i] = salts[soff + i];\n"
"    mbytes[total_len] = 0x80;\n"
"    M[14] = total_len * 8;\n"
"\n"
"    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;\n"
"    md5_block(&hx, &hy, &hz, &hw, M);\n"
"\n"
"    for (uint iter = 0; iter < params.max_iter; iter++) {\n"
"        if (iter > 0) {\n"
"            uint hwords[4]; hwords[0]=hx; hwords[1]=hy; hwords[2]=hz; hwords[3]=hw;\n"
"            uchar *hb = (uchar *)hwords;\n"
"            uint Mi[16];\n"
"            uchar *mb = (uchar *)Mi;\n"
"            for (int i = 0; i < 16; i++) {\n"
"                uchar hi = hb[i] >> 4, lo = hb[i] & 0xf;\n"
"                mb[i*2]   = hi + (hi < 10 ? '0' : 'a' - 10);\n"
"                mb[i*2+1] = lo + (lo < 10 ? '0' : 'a' - 10);\n"
"            }\n"
"            Mi[8] = 0x80; Mi[9]=0; Mi[10]=0; Mi[11]=0;\n"
"            Mi[12]=0; Mi[13]=0; Mi[14]=256; Mi[15]=0;\n"
"            hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;\n"
"            md5_block(&hx, &hy, &hz, &hw, Mi);\n"
"        }\n"
"        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,\n"
"                          params.compact_mask, params.max_probe, params.hash_data_count,\n"
"                          hash_data_buf, hash_data_off,\n"
"                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {\n"
"            uint slot = atomic_add(hit_count, 1u);\n"
"            if (slot < params.max_hits) {\n"
"                uint base = slot * 7;\n"
"                hits[base] = word_idx; hits[base+1] = salt_idx;\n"
"                hits[base+2] = iter + 1;\n"
"                hits[base+3] = hx; hits[base+4] = hy;\n"
"                hits[base+5] = hz; hits[base+6] = hw;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

/* ---- List GPU devices and exit (called early from option parsing) ---- */
void opencl_md5salt_list_devices(void) {
    if (opencl_dynload_init() != 0) {
        fprintf(stderr, "  OpenCL library not found.\n");
        exit(0);
    }
    cl_platform_id plats[8];
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(8, plats, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        fprintf(stderr, "  No OpenCL platforms found.\n");
        exit(0);
    }
    int idx = 0;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[64];
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 64, devs, &ndev);
        if (err != CL_SUCCESS || ndev == 0) continue;
        for (cl_uint d = 0; d < ndev; d++) {
            char dname[256];
            cl_ulong gmem = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
            fprintf(stderr, "  GPU[%d]: %s (%llu MB)\n", idx, dname,
                    (unsigned long long)(gmem / (1024*1024)));
            idx++;
        }
    }
    if (idx == 0) fprintf(stderr, "  No GPU devices found.\n");
    exit(0);
}

/* ---- GPU device filter ---- */
extern int gpu_device_filter_set;
extern int gpu_device_allowed[];

static int device_allowed(int idx) {
    if (!gpu_device_filter_set) return 1;
    if (idx < 0 || idx >= 64) return 0;
    return gpu_device_allowed[idx];
}

/* ---- Helper ---- */
#define OCL_CHECK(call, msg) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d: %s\n", _e, msg); return -1; } } while(0)

static cl_mem dev_buf(struct gpu_device *d, size_t size, cl_mem_flags flags) {
    cl_int err;
    return clCreateBuffer(d->ctx, flags, size ? size : 4, NULL, &err);
}

/* Initialize one GPU device */
static int init_device(int di, cl_device_id dev_id, const char *kernel_source) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    d->dev = dev_id;
    clGetDeviceInfo(dev_id, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);

    d->ctx = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &err);
    if (!d->ctx) return -1;

    d->queue = clCreateCommandQueue(d->ctx, dev_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!d->queue) return -1;

    d->prog = clCreateProgramWithSource(d->ctx, 1, &kernel_source, NULL, &err);
    if (!d->prog) return -1;

    err = clBuildProgram(d->prog, 1, &dev_id, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(d->prog, dev_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "OpenCL GPU[%d] kernel compile error:\n%s\n", di, log);
        return -1;
    }

    d->kern_batch = clCreateKernel(d->prog, "md5salt_batch", &err);
    d->kern_sub8  = clCreateKernel(d->prog, "md5salt_sub8_24", &err);
    d->kern_iter  = clCreateKernel(d->prog, "md5salt_iter", &err);
    d->kern_md5_iter_lc = clCreateKernel(d->prog, "md5_iter_lc", &err);
    d->kern_md5_iter_uc = clCreateKernel(d->prog, "md5_iter_uc", &err);

    /* Register kernels by op type for this device */
    memset(&dev_kerns[di], 0, sizeof(dev_kerns[di]));
    kern_register(di, JOB_MD5SALT, d->kern_batch);
    kern_register(di, JOB_MD5UCSALT, d->kern_batch);
    kern_register(di, JOB_MD5revMD5SALT, d->kern_batch);
    kern_register(di, JOB_MD5sub8_24SALT, d->kern_sub8);
    kern_register(di, JOB_MD5, d->kern_md5_iter_lc);
    kern_register(di, JOB_MD5UC, d->kern_md5_iter_uc);

    /* Per-device dispatch buffers */
    d->b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               GPU_MAX_HITS * 7 * sizeof(uint32_t), NULL, &err);
    d->b_hit_count = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t), NULL, &err);
    d->h_hits = (uint32_t *)malloc(GPU_MAX_HITS * 7 * sizeof(uint32_t));
    d->b_params = dev_buf(d, sizeof(OCLParams), CL_MEM_READ_ONLY);

    /* Dummy overflow buffers (replaced when overflow is loaded) */
    d->b_overflow_keys = dev_buf(d, 4, CL_MEM_READ_ONLY);
    d->b_overflow_hashes = dev_buf(d, 4, CL_MEM_READ_ONLY);
    d->b_overflow_offsets = dev_buf(d, 4, CL_MEM_READ_ONLY);
    d->b_overflow_lengths = dev_buf(d, 4, CL_MEM_READ_ONLY);

    d->salt_data_cap = 0;
    d->salts_count = 0;
    d->hexhash_cap = 0;

    return 0;
}

/* ---- Public API ---- */

int opencl_md5salt_init(void) {
    if (opencl_dynload_init() != 0) return -1;

    cl_uint nplat = 0;
    cl_platform_id plats[8];
    cl_int err;

    clGetPlatformIDs(8, plats, &nplat);
    if (nplat == 0) { fprintf(stderr, "OpenCL: no platforms\n"); return -1; }

    /* Load kernel source */
#ifdef DEBUG
    const char *kernel_source = load_kernel_file("md5salt.cl");
    if (kernel_source) {
        fprintf(stderr, "OpenCL GPU: loaded kernel from md5salt.cl\n");
    } else {
        kernel_source = md5salt_kernel_str;
        fprintf(stderr, "OpenCL GPU: using embedded kernel\n");
    }
#else
    const char *kernel_source = md5salt_kernel_str;
#endif

    /* Enumerate all GPU devices across all platforms.
     * -G 0,2,4 or -G 0-2: select specific devices.
     * (-G list is handled earlier in main via opencl_md5salt_list_devices) */
    int all_dev_idx = 0;

    num_gpu_devs = 0;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[64];
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 64,
                             devs, &ndev);
        if (err != CL_SUCCESS || ndev == 0) continue;

        for (cl_uint d = 0; d < ndev; d++) {
            char dname[256];
            cl_ulong gmem = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);

            if (!device_allowed(all_dev_idx)) {
                fprintf(stderr, "OpenCL GPU[%d]: %s - skipped\n", all_dev_idx, dname);
                all_dev_idx++;
                continue;
            }

            if (num_gpu_devs < MAX_GPU_DEVICES) {
                if (init_device(num_gpu_devs, devs[d], kernel_source) == 0) {
                    fprintf(stderr, "OpenCL GPU[%d]: %s (%llu MB)\n", all_dev_idx, dname,
                            (unsigned long long)(gmem / (1024*1024)));
                    num_gpu_devs++;
                }
            }
            all_dev_idx++;
        }
    }

    if (num_gpu_devs == 0) {
        fprintf(stderr, "OpenCL: no GPU devices found\n");
        return -1;
    }

    fprintf(stderr, "OpenCL GPU: %d device%s initialized\n",
            num_gpu_devs, num_gpu_devs > 1 ? "s" : "");
    ocl_ready = 1;
    return 0;
}

void opencl_md5salt_shutdown(void) {
    if (!ocl_ready) return;
    for (int i = 0; i < num_gpu_devs; i++) {
        struct gpu_device *d = &gpu_devs[i];
        if (d->b_compact_fp) clReleaseMemObject(d->b_compact_fp);
        if (d->b_compact_idx) clReleaseMemObject(d->b_compact_idx);
        if (d->b_hash_data) clReleaseMemObject(d->b_hash_data);
        if (d->b_hash_data_off) clReleaseMemObject(d->b_hash_data_off);
        if (d->b_hash_data_len) clReleaseMemObject(d->b_hash_data_len);
        if (d->b_salt_data) clReleaseMemObject(d->b_salt_data);
        if (d->b_salt_off) clReleaseMemObject(d->b_salt_off);
        if (d->b_salt_len) clReleaseMemObject(d->b_salt_len);
        if (d->b_hits) clReleaseMemObject(d->b_hits);
        if (d->b_hit_count) clReleaseMemObject(d->b_hit_count);
        if (d->b_params) clReleaseMemObject(d->b_params);
        if (d->b_overflow_keys) clReleaseMemObject(d->b_overflow_keys);
        if (d->b_overflow_hashes) clReleaseMemObject(d->b_overflow_hashes);
        if (d->b_overflow_offsets) clReleaseMemObject(d->b_overflow_offsets);
        if (d->b_overflow_lengths) clReleaseMemObject(d->b_overflow_lengths);
        if (d->b_hexhashes) clReleaseMemObject(d->b_hexhashes);
        if (d->b_hexlens) clReleaseMemObject(d->b_hexlens);
        if (d->kern_batch) clReleaseKernel(d->kern_batch);
        if (d->kern_sub8) clReleaseKernel(d->kern_sub8);
        if (d->kern_iter) clReleaseKernel(d->kern_iter);
        if (d->prog) clReleaseProgram(d->prog);
        if (d->queue) clReleaseCommandQueue(d->queue);
        if (d->ctx) clReleaseContext(d->ctx);
        free(d->h_hits);
    }
    ocl_ready = 0;
}

int opencl_md5salt_available(void) { return ocl_ready; }
int opencl_md5salt_num_devices(void) { return num_gpu_devs; }

int opencl_md5salt_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;
    _compact_mask = compact_mask;
    _hash_data_count = hash_data_count;

    d->b_compact_fp = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     compact_size * sizeof(uint32_t), compact_fp, &err);
    d->b_compact_idx = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      compact_size * sizeof(uint32_t), compact_idx, &err);
    d->b_hash_data = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    hash_data_buf_size, hash_data_buf, &err);

    uint64_t *off64 = (uint64_t *)malloc(hash_data_count * sizeof(uint64_t));
    for (size_t i = 0; i < hash_data_count; i++) off64[i] = hash_data_off[i];
    d->b_hash_data_off = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        hash_data_count * sizeof(uint64_t), off64, &err);
    free(off64);

    d->b_hash_data_len = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        hash_data_count * sizeof(uint16_t), hash_data_len, &err);

    fprintf(stderr, "OpenCL GPU[%d]: compact table registered (%llu slots, %u hashes)\n",
            dev_idx, (unsigned long long)compact_size, (unsigned)hash_data_count);
    return 0;
}

int opencl_md5salt_set_salts(int dev_idx,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || num_salts <= 0) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;
    size_t salts_size = salt_offsets[num_salts - 1] + salt_lens[num_salts - 1];

    if (salts_size > d->salt_data_cap) {
        if (d->b_salt_data) clReleaseMemObject(d->b_salt_data);
        d->b_salt_data = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, salts_size + 4096, NULL, &err);
        d->salt_data_cap = salts_size + 4096;
    }
    if (d->b_salt_off) clReleaseMemObject(d->b_salt_off);
    if (d->b_salt_len) clReleaseMemObject(d->b_salt_len);
    d->b_salt_off = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_salts * sizeof(uint32_t), NULL, &err);
    d->b_salt_len = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_salts * sizeof(uint16_t), NULL, &err);

    clEnqueueWriteBuffer(d->queue, d->b_salt_data, CL_TRUE, 0, salts_size, salts, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_salt_off, CL_TRUE, 0, num_salts * sizeof(uint32_t), salt_offsets, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_salt_len, CL_TRUE, 0, num_salts * sizeof(uint16_t), salt_lens, 0, NULL, NULL);
    d->salts_count = num_salts;
    return 0;
}

int opencl_md5salt_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || count <= 0) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;
    size_t total = offsets[count - 1] + lengths[count - 1];

    clReleaseMemObject(d->b_overflow_keys);
    clReleaseMemObject(d->b_overflow_hashes);
    clReleaseMemObject(d->b_overflow_offsets);
    clReleaseMemObject(d->b_overflow_lengths);

    d->b_overflow_keys = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        count * sizeof(uint64_t), (void*)keys, &err);
    d->b_overflow_hashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          total, (void*)hashes, &err);
    d->b_overflow_offsets = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           count * sizeof(uint32_t), (void*)offsets, &err);
    d->b_overflow_lengths = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            count * sizeof(uint16_t), (void*)lengths, &err);
    _overflow_count = count;
    fprintf(stderr, "OpenCL GPU[%d]: %d overflow entries loaded\n", dev_idx, count);
    return 0;
}

void opencl_md5salt_set_max_iter(int max_iter) { _max_iter = (max_iter < 1) ? 1 : max_iter; }
void opencl_md5salt_set_op(int op) { _gpu_op = op; }

uint32_t *opencl_md5salt_dispatch_batch(int dev_idx,
    const char *hexhashes, const uint16_t *hexlens,
    int num_words, int *nhits_out)
{
    *nhits_out = 0;
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || num_words <= 0) return NULL;
    struct gpu_device *d = &gpu_devs[dev_idx];
    /* Look up kernel for current op */
    int op = _gpu_op;
    int cat = gpu_op_category(op);
    struct gpu_kern *gk = (op >= 0 && op < MAX_GPU_KERNELS) ? &dev_kerns[dev_idx].kerns[op] : NULL;
    if (!gk || !gk->kernel) return NULL;

    /* Salted types need salts uploaded; iteration types do not */
    if (cat == GPU_CAT_SALTED && d->salts_count <= 0) return NULL;

    /* For salted iteration, use the salt-iter kernel; otherwise use the registered kernel */
    cl_kernel kern = (cat == GPU_CAT_SALTED && _max_iter > 1 && d->kern_iter) ? d->kern_iter : gk->kernel;
    cl_int err;

    /* Upload words */
    size_t words_size = (size_t)num_words * 256;
    if (words_size > d->hexhash_cap) {
        if (d->b_hexhashes) clReleaseMemObject(d->b_hexhashes);
        if (d->b_hexlens) clReleaseMemObject(d->b_hexlens);
        d->b_hexhashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, words_size, NULL, &err);
        d->b_hexlens = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_words * sizeof(uint16_t), NULL, &err);
        d->hexhash_cap = words_size;
    }
    clEnqueueWriteBuffer(d->queue, d->b_hexhashes, CL_TRUE, 0, words_size, hexhashes, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_hexlens, CL_TRUE, 0, num_words * sizeof(uint16_t), hexlens, 0, NULL, NULL);

    /* Params */
    OCLParams params;
    params.compact_mask = _compact_mask;
    params.num_words = num_words;
    params.num_salts = d->salts_count;
    params.max_probe = 256;
    params.hash_data_count = _hash_data_count;
    params.max_hits = GPU_MAX_HITS;
    params.overflow_count = _overflow_count;
    /* For iteration types, CPU did iter 1; GPU does the remaining */
    params.max_iter = (cat == GPU_CAT_ITER) ? (_max_iter - 1) : _max_iter;
    clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);

    /* Zero hit counter */
    uint32_t zero = 0;
    clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero, sizeof(zero), 0, sizeof(zero), 0, NULL, NULL);
    clFinish(d->queue);

    /* Set kernel args */
    {
        int a = 0;
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexhashes);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexlens);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_salt_data);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_salt_off);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_salt_len);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_compact_fp);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_compact_idx);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_params);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hash_data);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hash_data_off);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hash_data_len);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hits);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hit_count);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_keys);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_hashes);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_offsets);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_overflow_lengths);
    }

    /* Dispatch with autotune */
    size_t local = kern_get_local_size(gk);
    size_t global = (cat == GPU_CAT_SALTED)
        ? (size_t)num_words * d->salts_count
        : (size_t)num_words;
    global = ((global + local - 1) / local) * local;

    struct timespec t0, t1;
    if (!gk->tuned) clock_gettime(CLOCK_MONOTONIC, &t0);

    err = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL GPU[%d] dispatch error: %d (global=%zu local=%zu)\n",
                dev_idx, err, global, local);
        return NULL;
    }
    clFinish(d->queue);

    if (!gk->tuned) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        kern_record_time(gk, ms);
    }

    /* Read back */
    uint32_t raw_nhits;
    clEnqueueReadBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(raw_nhits), &raw_nhits, 0, NULL, NULL);

    *nhits_out = (int)raw_nhits;
    if (raw_nhits > 0) {
        if (raw_nhits > GPU_MAX_HITS) raw_nhits = GPU_MAX_HITS;
        int stride = (_max_iter > 1) ? 7 : 6;
        clEnqueueReadBuffer(d->queue, d->b_hits, CL_TRUE, 0, raw_nhits * stride * sizeof(uint32_t), d->h_hits, 0, NULL, NULL);
    }
    return d->h_hits;
}

#endif /* OPENCL_GPU */
