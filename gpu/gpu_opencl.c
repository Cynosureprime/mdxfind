/*
 * gpu_opencl.c — OpenCL GPU acceleration for mdxfind
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
#include "gpu_opencl.h"
#include "job_types.h"
#include "gpujob.h"

/* ---- Multi-GPU device state ---- */
#define MAX_GPU_DEVICES 64

struct gpu_device {
    cl_context       ctx;
    cl_command_queue  queue;
    cl_program        prog;           /* selftest program (common only) */
    cl_program        fam_prog[FAM_COUNT]; /* per-family compiled programs */
    cl_device_id      dev;
    cl_kernel         kern_salt_iter; /* special: salted iteration override for Maxiter>1 */
    char              name[256];
    int               max_batch;     /* per-device batch limit (words per dispatch) */
    int               max_dispatch;  /* max work items per dispatch (0 = unlimited) */

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

    /* Mask mode */
    cl_mem bgpu_mask_desc;  /* mask descriptor: charset IDs per position */

    /* Per-family timing-based dispatch sizing */
    uint32_t fam_max_items[FAM_COUNT];  /* timed max work items per dispatch */
    int      fam_timed[FAM_COUNT];      /* 1 = probed for this family */
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
static uint64_t _mask_resume = 0;  /* mask_start override for overflow retry */
static uint32_t _salt_resume = 0;  /* salt_start override for overflow retry */

/* ---- Timing-based dispatch sizing constants ---- */
#define TIMING_BUDGET_MIN_MS  200.0
#define TIMING_BUDGET_MAX_MS  400.0
#define TIMING_INITIAL_SIZE   65536   /* 64K work items */
#define TIMING_WATCHDOG_MS    5000.0  /* never risk exceeding this */

extern int gpu_op_family(int op);

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
    /* Check for reqd_work_group_size — if set, the required size overrides max_wg */
    size_t req_wg[3] = {0, 0, 0};
    clGetKernelWorkGroupInfo(kernel, gpu_devs[di].dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(req_wg), req_wg, NULL);
    if (req_wg[0] > 0) max_wg = req_wg[0];
    k->kernel = kernel;
    k->max_local = max_wg;
    k->dev_idx = di;
    k->tuned = 0;
    k->tune_candidate = 0;
    k->tune_samples = 0;
    k->tune_best_time = 1e30;
    k->tune_best_size = max_wg ? max_wg : 64;
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
    uint32_t salt_start;
    uint32_t max_probe;
    uint32_t hash_data_count;
    uint32_t max_hits;
    uint32_t overflow_count;
    uint32_t max_iter;
    uint32_t num_masks;
    uint64_t mask_start;
    uint32_t n_prepend;
    uint32_t n_append;
    uint64_t mask_base0;   /* pre-decomposed mask_start: positions 0-7 packed as bytes */
    uint64_t mask_base1;   /* positions 8-15 packed as bytes */
} OCLParams;

/* ---- Load kernel source from file ---- */
static char *load_kernel_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        /* Try gpu/ prefix */
        char buf[512];
        snprintf(buf, sizeof(buf), "gpu/%s", path);
        f = fopen(buf, "r");
    }
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char *)malloc_lock(sz + 1,"load_kernel");
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);
    return src;
}

/* ---- Embedded kernel sources (auto-generated from gpu_*.cl via cl2str.py --all) ---- */
#include "gpu_common_str.h"
#include "gpu_md5salt_str.h"
#include "gpu_md5saltpass_str.h"
#include "gpu_md5iter_str.h"
#include "gpu_phpbb3_str.h"
#include "gpu_md5crypt_str.h"
#include "gpu_md5_md5saltmd5pass_str.h"
#include "gpu_sha1_str.h"
#include "gpu_sha256_str.h"
#include "gpu_md5mask_str.h"
#include "gpu_descrypt_str.h"
#include "gpu_md5unsalted_str.h"
#include "gpu_md4unsalted_str.h"
#include "gpu_sha1unsalted_str.h"
#include "gpu_sha256unsalted_str.h"
#include "gpu_sha512unsalted_str.h"
#include "gpu_wrlunsalted_str.h"
#include "gpu_md6256unsalted_str.h"
#include "gpu_keccakunsalted_str.h"
/* HMAC-SHA256 kernels are in gpu_sha256.cl (FAM_SHA256) */
#include "gpu_hmac_sha512_str.h"
#include "gpu_mysql3unsalted_str.h"
#include "gpu_hmac_rmd160_str.h"
#include "gpu_hmac_rmd320_str.h"
#include "gpu_hmac_blake2s_str.h"
#include "gpu_streebog_str.h"
#include "gpu_sha512crypt_str.h"
#include "gpu_sha256crypt_str.h"
#include "gpu_rmd160unsalted_str.h"
#include "gpu_blake2s256unsalted_str.h"
#include "gpu_bcrypt_str.h"
/* Old monolithic kernel source removed — per-family compilation now.
 * See gpu_common_str.h + gpu_*_str.h
 *
 * REMOVED: ~300 lines of inline kernel string (kernel_source_embedded_unused)
"}\n";
#endif

/* ---- List GPU devices and exit (called early from option parsing) ---- */
void gpu_opencl_list_devices(void) {
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
extern int GPUForce;
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

/* Per-family kernel source (concatenated with gpu_common_str at compile time) */
static const char *family_source[FAM_COUNT] = {
    [FAM_MD5SALT]            = gpu_md5salt_str,
    [FAM_MD5SALTPASS]        = gpu_md5saltpass_str,
    [FAM_MD5ITER]            = gpu_md5iter_str,
    [FAM_PHPBB3]             = gpu_phpbb3_str,
    [FAM_MD5CRYPT]           = gpu_md5crypt_str,
    [FAM_MD5_MD5SALTMD5PASS] = gpu_md5_md5saltmd5pass_str,
    [FAM_SHA1]               = gpu_sha1_str,
    [FAM_SHA256]             = gpu_sha256_str,
    [FAM_MD5MASK]            = gpu_md5mask_str,
    [FAM_DESCRYPT]           = gpu_descrypt_str,
    [FAM_MD5UNSALTED]        = gpu_md5unsalted_str,
    [FAM_MD4UNSALTED]        = gpu_md4unsalted_str,
    [FAM_SHA1UNSALTED]       = gpu_sha1unsalted_str,
    [FAM_SHA256UNSALTED]     = gpu_sha256unsalted_str,
    [FAM_SHA512UNSALTED]     = gpu_sha512unsalted_str,
    [FAM_WRLUNSALTED]        = gpu_wrlunsalted_str,
    [FAM_MD6256UNSALTED]     = gpu_md6256unsalted_str,
    [FAM_KECCAKUNSALTED]     = gpu_keccakunsalted_str,
    [FAM_HMAC_SHA512]        = gpu_hmac_sha512_str,
    [FAM_MYSQL3UNSALTED]     = gpu_mysql3unsalted_str,
    [FAM_HMAC_RMD160]       = gpu_hmac_rmd160_str,
    [FAM_HMAC_RMD320]       = gpu_hmac_rmd320_str,
    [FAM_HMAC_BLAKE2S]      = gpu_hmac_blake2s_str,
    [FAM_STREEBOG]          = gpu_streebog_str,
    [FAM_SHA512CRYPT]       = gpu_sha512crypt_str,
    [FAM_SHA256CRYPT]       = gpu_sha256crypt_str,
    [FAM_RMD160UNSALTED]    = gpu_rmd160unsalted_str,
    [FAM_BLAKE2S256UNSALTED] = gpu_blake2s256unsalted_str,
    [FAM_BCRYPT]            = gpu_bcrypt_str,
};

/* Kernel-to-op mapping table. Each entry: kernel function name, ops it serves, family.
 * Adding a new GPU algorithm = one line here + a .cl file in gpu/. */
static const struct {
    const char *name;
    int ops[8];      /* -1 terminated */
    int family;
} kernel_map[] = {
    {"md5salt_batch",       {JOB_MD5SALT, JOB_MD5UCSALT, JOB_MD5revMD5SALT, -1}, FAM_MD5SALT},
    {"md5salt_sub8_24",     {JOB_MD5sub8_24SALT, -1}, FAM_MD5SALT},
    {"md5salt_iter",        {-1}, FAM_MD5SALT},  /* special: salted iteration override */
    {"md5saltpass_batch",   {JOB_MD5SALTPASS, -1}, FAM_MD5SALTPASS},
    {"md5passsalt_batch",   {JOB_MD5PASSSALT, -1}, FAM_MD5SALTPASS},
    {"md5_iter_lc",         {JOB_MD5, -1}, FAM_MD5ITER},
    {"md5_iter_uc",         {JOB_MD5UC, -1}, FAM_MD5ITER},
    {"sha256passsalt_batch", {JOB_SHA256PASSSALT, -1}, FAM_SHA256},
    {"sha256saltpass_batch", {JOB_SHA256SALTPASS, -1}, FAM_SHA256},
    {"md5_mask_batch",        {JOB_MD5, -1}, FAM_MD5MASK},
    {"sha1dru_batch",         {JOB_SHA1DRU, -1}, FAM_SHA1},
    {"sha1passsalt_batch",    {JOB_SHA1PASSSALT, -1}, FAM_SHA1},
    {"sha1saltpass_batch",   {JOB_SHA1SALTPASS, -1}, FAM_SHA1},
    {"phpbb3_batch",          {JOB_PHPBB3, -1}, FAM_PHPBB3},
    {"md5_md5saltmd5pass_batch", {JOB_MD5_MD5SALTMD5PASS, -1}, FAM_MD5_MD5SALTMD5PASS},
    {"md5crypt_batch",       {JOB_MD5CRYPT, -1}, FAM_MD5CRYPT},
    {"descrypt_batch",       {JOB_DESCRYPT, -1}, FAM_DESCRYPT},
    {"md5_unsalted_batch",   {JOB_MD5, -1}, FAM_MD5UNSALTED},
    {"md4_unsalted_batch",   {JOB_MD4, -1}, FAM_MD4UNSALTED},
    {"md4utf16_unsalted_batch", {JOB_NTLMH, -1}, FAM_MD4UNSALTED},
    {"sha1_unsalted_batch",  {JOB_SHA1, -1}, FAM_SHA1UNSALTED},
    {"sha256_unsalted_batch", {JOB_SHA256, -1}, FAM_SHA256UNSALTED},
    {"sha224_unsalted_batch", {JOB_SHA224, -1}, FAM_SHA256UNSALTED},
    {"sha256raw_unsalted_batch", {JOB_SHA256RAW, -1}, FAM_SHA256UNSALTED},
    {"sha512_unsalted_batch", {JOB_SHA512, -1}, FAM_SHA512UNSALTED},
    {"sha384_unsalted_batch", {JOB_SHA384, -1}, FAM_SHA512UNSALTED},
    {"wrl_unsalted_batch",   {JOB_WRL, -1}, FAM_WRLUNSALTED},
    {"md6_256_unsalted_batch", {JOB_MD6256, -1}, FAM_MD6256UNSALTED},
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
    {"hmac_sha1_ksalt_batch",   {JOB_HMAC_SHA1, -1}, FAM_SHA1},
    {"hmac_sha1_kpass_batch",   {JOB_HMAC_SHA1_KPASS, -1}, FAM_SHA1},
    {"sha512passsalt_batch",    {JOB_SHA512PASSSALT, -1}, FAM_HMAC_SHA512},
    {"sha512saltpass_batch",   {JOB_SHA512SALTPASS, -1}, FAM_HMAC_SHA512},
    {"hmac_sha512_ksalt_batch", {JOB_HMAC_SHA512, -1}, FAM_HMAC_SHA512},
    {"hmac_sha512_kpass_batch", {JOB_HMAC_SHA512_KPASS, -1}, FAM_HMAC_SHA512},
    {"hmac_sha384_ksalt_batch", {JOB_HMAC_SHA384, -1}, FAM_HMAC_SHA512},
    {"hmac_sha384_kpass_batch", {JOB_HMAC_SHA384_KPASS, -1}, FAM_HMAC_SHA512},
    {"sql5_unsalted_batch",    {JOB_SQL5, -1}, FAM_SHA1UNSALTED},
    {"sha1raw_unsalted_batch", {JOB_SHA1RAW, -1}, FAM_SHA1UNSALTED},
    {"md5raw_unsalted_batch",  {JOB_MD5RAW, -1}, FAM_MD5UNSALTED},
    {"sha384raw_unsalted_batch", {JOB_SHA384RAW, -1}, FAM_SHA512UNSALTED},
    {"sha512raw_unsalted_batch", {JOB_SHA512RAW, -1}, FAM_SHA512UNSALTED},
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
    {"sha512crypt_batch",              {JOB_SHA512CRYPT, JOB_SHA512CRYPTMD5, -1}, FAM_SHA512CRYPT},
    {"sha256crypt_batch",              {JOB_SHA256CRYPT, -1}, FAM_SHA256CRYPT},
    {"rmd160_unsalted_batch",          {JOB_RMD160, -1}, FAM_RMD160UNSALTED},
    {"blake2s256_unsalted_batch",      {JOB_BLAKE2S256, -1}, FAM_BLAKE2S256UNSALTED},
    {"bcrypt_batch",                   {JOB_BCRYPT, -1}, FAM_BCRYPT},
    {NULL, {-1}, 0}
};
#define KERN_ITER_IDX 2  /* index of md5salt_iter in kernel_map (for Maxiter override) */

/* OpenCL async error callback — called by the driver on compute errors */
static void CL_CALLBACK ocl_error_callback(const char *errinfo,
    const void *private_info, size_t cb, void *user_data) {
    fprintf(stderr, "OpenCL GPU ASYNC ERROR: %s\n", errinfo);
}

/* Initialize one GPU device */
static int init_device(int di, cl_device_id dev_id) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    d->dev = dev_id;
    clGetDeviceInfo(dev_id, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);

    d->ctx = clCreateContext(NULL, 1, &dev_id, ocl_error_callback, NULL, &err);
    if (!d->ctx) return -1;

    d->queue = clCreateCommandQueue(d->ctx, dev_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!d->queue) return -1;

    /* Compile common-only program (warms up NVIDIA driver state) */
    d->prog = clCreateProgramWithSource(d->ctx, 1, (const char *[]){gpu_common_str}, NULL, &err);
    if (!d->prog) return -1;
    fprintf(stderr, "OpenCL GPU[%d]: kernel build\n", di);
    err = clBuildProgram(d->prog, 1, &dev_id, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(d->prog, dev_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "OpenCL GPU[%d] common kernel compile error:\n%s\n", di, log);
        return -1;
    }

    /* Compile only the md5salt family at init (needed for selftest/probe).
     * Other families are compiled later by gpu_opencl_compile_families(). */
    memset(d->fam_prog, 0, sizeof(d->fam_prog));
    memset(&dev_kerns[di], 0, sizeof(dev_kerns[di]));
    d->kern_salt_iter = NULL;
    {
        const char *sources[2] = { gpu_common_str, family_source[FAM_MD5SALT] };
        d->fam_prog[FAM_MD5SALT] = clCreateProgramWithSource(d->ctx, 2, sources, NULL, &err);
        if (d->fam_prog[FAM_MD5SALT]) {
            err = clBuildProgram(d->fam_prog[FAM_MD5SALT], 1, &dev_id, "-cl-std=CL1.2", NULL, NULL);
            if (err != CL_SUCCESS) {
                char log[4096];
                clGetProgramBuildInfo(d->fam_prog[FAM_MD5SALT], dev_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
                fprintf(stderr, "OpenCL GPU[%d] md5salt compile error:\n%s\n", di, log);
                clReleaseProgram(d->fam_prog[FAM_MD5SALT]);
                d->fam_prog[FAM_MD5SALT] = NULL;
            }
        }
        /* Register md5salt kernels */
        for (int k = 0; kernel_map[k].name; k++) {
            if (kernel_map[k].family != FAM_MD5SALT) continue;
            if (!d->fam_prog[FAM_MD5SALT]) continue;
            cl_kernel kern = clCreateKernel(d->fam_prog[FAM_MD5SALT], kernel_map[k].name, &err);
            if (!kern) continue;
            if (k == KERN_ITER_IDX)
                d->kern_salt_iter = kern;
            for (int j = 0; kernel_map[k].ops[j] >= 0; j++)
                kern_register(di, kernel_map[k].ops[j], kern);
        }
    }

    /* Scale batch size to GPU capability */
    cl_uint compute_units = 0;
    clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    if (compute_units < 8)
        d->max_batch = compute_units * 8;   /* tiny GPU: 4 CU -> 32 words */
    else
        d->max_batch = GPUBATCH_MAX;        /* 8+ CU: full 512 */
    if (d->max_batch < 16) d->max_batch = 16;
    if (d->max_batch > GPUBATCH_MAX) d->max_batch = GPUBATCH_MAX;
    fprintf(stderr, "OpenCL GPU[%d]: %u compute units, batch size %d\n",
            di, compute_units, d->max_batch);

    /* Per-device dispatch buffers */
    d->b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               GPU_MAX_HITS * 19 * sizeof(uint32_t), NULL, &err);
    d->b_hit_count = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t), NULL, &err);
    d->h_hits = (uint32_t *)malloc_lock(GPU_MAX_HITS * 19 * sizeof(uint32_t),"device_init");
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

/* Probe maximum reliable dispatch size for a GPU device.
 * Uses the actual md5salt_batch kernel with synthetic test data at increasing
 * power-of-2 salt counts. Each test uses 1 word with a known MD5SALT result;
 * the number of salts scales up. When the GPU misses the known hit, we've
 * found the dispatch limit.
 * Returns 0 if all sizes pass (no limit needed). */
static int probe_max_dispatch(int di) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    cl_kernel test_kern = d->fam_prog[FAM_MD5SALT]
        ? clCreateKernel(d->fam_prog[FAM_MD5SALT], "md5salt_batch", &err) : NULL;
    if (!test_kern) {
        fprintf(stderr, "OpenCL GPU[%d]: probe kernel not found (err=%d)\n", di, err);
        return 0;
    }

    /* Test word: "a" → hex = "61" (2 hex chars) padded to 256 bytes.
     * Test salt at index 0: "x" (1 byte).
     * MD5("61" + "x") = MD5("61x") — we'll compute this on the CPU to get the expected hash,
     * then build a tiny 2-slot compact table containing just this one hash.
     * The test: dispatch with 1 word and N salts (salt 0 = "x", salts 1..N-1 = "y").
     * If the GPU correctly processes all N work items, it finds exactly 1 hit (salt 0).
     * If the dispatch is too large and corrupts computation, it finds 0 hits. */

    /* Kernel computes MD5(hexhash[0..31] + salt) where hexhash = "61" + 30 NULs, salt = "x"
     * = MD5("61\0\0...\0x") (33 bytes) = 1d7f6bd1126bd87f5f5a4cdb6502cc5e */
    uint32_t exp_hx = 0xd16b7f1du, exp_hy = 0x7fd86b12u;

    /* Build a 4-slot compact table (mask=3) with just this one hash */
    uint32_t compact_fp[4] = {0, 0, 0, 0};
    uint32_t compact_idx[4] = {0, 0, 0, 0};
    uint32_t fp = exp_hy;
    if (fp == 0) fp = 1;
    uint32_t pos = (exp_hx ^ exp_hy) & 3;
    compact_fp[pos] = fp;
    compact_idx[pos] = 0;

    /* hash_data: 16-byte hash at offset 0 */
    uint32_t hash_data[4] = { exp_hx, exp_hy, 0xdb4c5a5fu, 0x5ecc0265u };
    uint64_t hash_off = 0;
    uint16_t hash_len = 16;

    /* Word: "61" in hex, padded to 256 bytes. First 2 bytes = 0x36,0x31 = "61" */
    /* 32 copies of the same word "61" in the hexhash buffer */
    unsigned char hexwords[32 * 256];
    uint16_t hexlens[32];
    memset(hexwords, 0, sizeof(hexwords));
    for (int w = 0; w < 32; w++) {
        hexwords[w * 256] = '6';
        hexwords[w * 256 + 1] = '1';
        hexlens[w] = 2;
    }

    /* Create GPU buffers for the test */
    cl_mem b_hexhash = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hexwords), hexwords, &err);
    cl_mem b_hexlen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hexlens), hexlens, &err);
    cl_mem b_cfp = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(compact_fp), compact_fp, &err);
    cl_mem b_cidx = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(compact_idx), compact_idx, &err);
    cl_mem b_hdata = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_data), hash_data, &err);
    cl_mem b_hoff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_off), &hash_off, &err);
    cl_mem b_hlen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_len), &hash_len, &err);

    /* Params buffer */
    OCLParams params;
    memset(&params, 0, sizeof(params));
    params.compact_mask = 3;
    params.num_words = 32;
    // params.salt_start = 0;
    params.max_probe = 4;
    params.hash_data_count = 1;
    params.max_hits = 256;
    params.overflow_count = 0;
    params.max_iter = 1;
    cl_mem b_params = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, sizeof(params), NULL, &err);

    /* Hits */
    cl_mem b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, 256 * 6 * sizeof(uint32_t), NULL, &err);
    cl_mem b_hitcnt = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &err);

    /* Dummy overflow buffers (empty) */
    uint64_t dummy64 = 0;
    uint32_t dummy32 = 0;
    uint16_t dummy16 = 0;
    cl_mem b_okeys = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8, &dummy64, &err);
    cl_mem b_ohash = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &dummy32, &err);
    cl_mem b_ooff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &dummy32, &err);
    cl_mem b_olen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2, &dummy16, &err);

    /* Build salt buffer: salt[0] = "x", salt[1..max] = "y" */
    int max_salts = 4194304;
    char *salt_data = (char *)malloc_lock(max_salts,"salt_data");
    uint32_t *salt_off = (uint32_t *)malloc_lock(max_salts * sizeof(uint32_t),"Salt_off");
    uint16_t *salt_len = (uint16_t *)malloc_lock(max_salts * sizeof(uint16_t),"salt_len");
    salt_data[0] = 'x';
    for (int i = 1; i < max_salts; i++) salt_data[i] = 'y';
    for (int i = 0; i < max_salts; i++) { salt_off[i] = i; salt_len[i] = 1; }

    cl_mem b_sdata = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts, NULL, &err);
    cl_mem b_soff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts * sizeof(uint32_t), NULL, &err);
    cl_mem b_slen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts * sizeof(uint16_t), NULL, &err);
    clEnqueueWriteBuffer(d->queue, b_sdata, CL_TRUE, 0, max_salts, salt_data, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, b_soff, CL_TRUE, 0, max_salts * sizeof(uint32_t), salt_off, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, b_slen, CL_TRUE, 0, max_salts * sizeof(uint16_t), salt_len, 0, NULL, NULL);
    clFinish(d->queue);
    free(salt_data); free(salt_off); free(salt_len);

    /* Set kernel args */
    int a = 0;
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hexhash);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hexlen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_sdata);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_soff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_slen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_cfp);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_cidx);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_params);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hdata);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hoff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hlen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hits);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hitcnt);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_okeys);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_ohash);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_ooff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_olen);

    /* Test at increasing power-of-2 salt counts */
    static const int test_sizes[] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 0
    };

    int max_good = 0;
    for (int t = 0; test_sizes[t]; t++) {
        int nsalts = test_sizes[t];

        params.num_salts = nsalts;
        // params.salt_start = 0;
        clEnqueueWriteBuffer(d->queue, b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);

        /* Zero hit counter */
        uint32_t zero = 0;
        clEnqueueWriteBuffer(d->queue, b_hitcnt, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
        clFinish(d->queue);

        size_t global = (size_t)32 * nsalts;  /* 32 words * nsalts */
        size_t local = 128;
        global = ((global + local - 1) / local) * local;

        err = clEnqueueNDRangeKernel(d->queue, test_kern, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL GPU[%d]: probe dispatch error %d at %d salts\n", di, err, nsalts);
            break;
        }
        cl_int finish_err = clFinish(d->queue);

        /* Read hit count — should be exactly 32 (salt 0 matches for each of 32 words) */
        uint32_t nhits;
        clEnqueueReadBuffer(d->queue, b_hitcnt, CL_TRUE, 0, sizeof(nhits), &nhits, 0, NULL, NULL);

        if (nhits == 32 && finish_err == CL_SUCCESS) {
            max_good = nsalts;
        } else {
            fprintf(stderr, "OpenCL GPU[%d]: probe FAIL at %d salts (%d items)"
                    " — got %u hits, expected 32, clFinish=%d\n",
                    di, nsalts, (int)global, nhits, finish_err);
            break;
        }
    }

    /* Cleanup */
    clReleaseKernel(test_kern);
    clReleaseMemObject(b_hexhash); clReleaseMemObject(b_hexlen);
    clReleaseMemObject(b_cfp); clReleaseMemObject(b_cidx);
    clReleaseMemObject(b_hdata); clReleaseMemObject(b_hoff); clReleaseMemObject(b_hlen);
    clReleaseMemObject(b_params); clReleaseMemObject(b_hits); clReleaseMemObject(b_hitcnt);
    clReleaseMemObject(b_sdata); clReleaseMemObject(b_soff); clReleaseMemObject(b_slen);
    clReleaseMemObject(b_okeys); clReleaseMemObject(b_ohash); clReleaseMemObject(b_ooff);
    clReleaseMemObject(b_olen);

    int last_tested = 0;
    for (int t = 0; test_sizes[t]; t++) last_tested = test_sizes[t];
    if (max_good >= last_tested)
        return 0;  /* all passed — no dispatch limit */
    return max_good;
}

int gpu_opencl_init(void) {
    if (opencl_dynload_init() != 0) return -1;

    cl_uint nplat = 0;
    cl_platform_id plats[8];
    cl_int err;

    clGetPlatformIDs(8, plats, &nplat);
    if (nplat == 0) { fprintf(stderr, "OpenCL: no platforms\n"); return -1; }

    /* Kernel sources are per-family, compiled in init_device */

    /* Enumerate all GPU devices across all platforms.
     * -G 0,2,4 or -G 0-2: select specific devices.
     * (-G list is handled earlier in main via gpu_opencl_list_devices) */
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
            cl_uint mhz = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, NULL);

            if (!device_allowed(all_dev_idx)) {
                fprintf(stderr, "OpenCL GPU[%d]: %s - skipped\n", all_dev_idx, dname);
                all_dev_idx++;
                continue;
            }

            if (num_gpu_devs < MAX_GPU_DEVICES) {
                if (init_device(num_gpu_devs, devs[d]) == 0) {
                    fprintf(stderr, "OpenCL GPU[%d]: %s (%llu MB, %u MHz)\n", all_dev_idx, dname,
                            (unsigned long long)(gmem / (1024*1024)), mhz);
                    int max_disp = probe_max_dispatch(num_gpu_devs);
                    /* Mali GPUs have a 17-bit salt dimension limit —
                     * silently drop work items beyond 2^17 salts.
                     * Probe returns max_good in salts (not work items),
                     * so convert: limit = max_salts * 32 test words.
                     * If probe didn't catch it, force the limit. */
                    if (strstr(dname, "Mali")) {
                        if (max_disp > 0)
                            max_disp *= 32;  /* probe result is in salts */
                        else
                            max_disp = 4194304;  /* 2^17 * 32 */
                    }
                    gpu_devs[num_gpu_devs].max_dispatch = max_disp;
                    if (max_disp == 0)
                        fprintf(stderr, "OpenCL GPU[%d]: selftest passed all sizes (no dispatch limit)\n", all_dev_idx);
                    else
                        fprintf(stderr, "OpenCL GPU[%d]: dispatch limit = %d work items\n", all_dev_idx, max_disp);
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

/* Compile additional kernel families on demand. Called from main() after
 * hash types are known. fam_mask is a bitmask of FAM_* values to compile. */
void gpu_opencl_compile_families(unsigned int fam_mask) {
    if (!ocl_ready) return;
    for (int di = 0; di < num_gpu_devs; di++) {
        struct gpu_device *d = &gpu_devs[di];
        cl_int err;
        for (int f = 0; f < FAM_COUNT; f++) {
            if (d->fam_prog[f]) continue;  /* already compiled */
            if (!(fam_mask & (1u << f))) continue;  /* not requested */
            const char *sources[2] = { gpu_common_str, family_source[f] };
            d->fam_prog[f] = clCreateProgramWithSource(d->ctx, 2, sources, NULL, &err);
            if (!d->fam_prog[f]) continue;
            err = clBuildProgram(d->fam_prog[f], 1, &d->dev, "-cl-std=CL1.2", NULL, NULL);
            if (err != CL_SUCCESS) {
                char log[4096];
                clGetProgramBuildInfo(d->fam_prog[f], d->dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
                fprintf(stderr, "OpenCL GPU[%d] family %d compile error:\n%s\n", di, f, log);
                clReleaseProgram(d->fam_prog[f]);
                d->fam_prog[f] = NULL;
                continue;
            }
            /* Register kernels from this family */
            for (int k = 0; kernel_map[k].name; k++) {
                if (kernel_map[k].family != f) continue;
                cl_kernel kern = clCreateKernel(d->fam_prog[f], kernel_map[k].name, &err);
                if (!kern) continue;
                if (k == KERN_ITER_IDX)
                    d->kern_salt_iter = kern;
                for (int j = 0; kernel_map[k].ops[j] >= 0; j++)
                    kern_register(di, kernel_map[k].ops[j], kern);
            }
        }
    }
}

void gpu_opencl_shutdown(void) {
    if (!ocl_ready) return;
    ocl_ready = 0;  /* prevent re-entry */
    for (int i = 0; i < num_gpu_devs; i++) {
        struct gpu_device *d = &gpu_devs[i];
        if (d->queue) clFinish(d->queue);  /* drain any pending GPU work */
        /* Skip CL object releases — NVIDIA driver may have already torn
         * down internal state by this point, causing NULL dereference
         * inside clReleaseKernel.  Process exit reclaims everything. */
        free(d->h_hits);
    }
}

int gpu_opencl_available(void) { return ocl_ready; }
int gpu_opencl_num_devices(void) { return num_gpu_devs; }

/* Brute-force multi-GPU: shared atomic cursor into mask keyspace. */
volatile uint64_t _bf_cursor = 0;
volatile int _bf_active = 0;

void gpu_opencl_bf_start(void) {
    _bf_cursor = 0;
    _bf_active = 1;
    __sync_synchronize();
}

void gpu_opencl_bf_stop(void) {
    _bf_active = 0;
    __sync_synchronize();
}

int gpu_opencl_bf_active(void) { __sync_synchronize(); return _bf_active; }
int gpu_opencl_max_batch(int dev_idx) {
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return GPUBATCH_MAX;
    return gpu_devs[dev_idx].max_batch;
}

int gpu_opencl_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;

    /* Check if GPU has enough memory for the compact table + hash data */
    size_t needed = compact_size * sizeof(uint32_t) * 2   /* compact_fp + compact_idx */
                  + hash_data_buf_size                      /* hash_data_buf */
                  + hash_data_count * sizeof(uint64_t)      /* hash_data_off */
                  + hash_data_count * sizeof(uint16_t)      /* hash_data_len */
                  + GPU_MAX_HITS * 11 * sizeof(uint32_t)     /* hits buffer */
                  + 512 * 256;                              /* hexhash buffer */
    cl_ulong gpu_mem = 0;
    clGetDeviceInfo(d->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gpu_mem), &gpu_mem, NULL);
    if (needed > (size_t)(gpu_mem * 0.8)) {
        fprintf(stderr, "OpenCL GPU[%d]: insufficient memory (%zuMB needed, %lluMB available) - skipping\n",
                dev_idx, needed / (1024*1024), (unsigned long long)(gpu_mem / (1024*1024)));
        return -1;
    }

    _compact_mask = compact_mask;
    _hash_data_count = hash_data_count;

    /* Allocate + upload in one call (CL_MEM_COPY_HOST_PTR) */
    d->b_compact_fp = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     compact_size * sizeof(uint32_t), compact_fp, &err);
    if (err != CL_SUCCESS) goto alloc_fail;
    d->b_compact_idx = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      compact_size * sizeof(uint32_t), compact_idx, &err);
    if (err != CL_SUCCESS) goto alloc_fail;
    d->b_hash_data = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    hash_data_buf_size, hash_data_buf, &err);
    if (err != CL_SUCCESS) goto alloc_fail;

    uint64_t *off64 = (uint64_t *)malloc_lock(hash_data_count * sizeof(uint64_t),"gpu temp");
    for (size_t i = 0; i < hash_data_count; i++) off64[i] = hash_data_off[i];
    d->b_hash_data_off = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        hash_data_count * sizeof(uint64_t), off64, &err);
    if (err != CL_SUCCESS) { free(off64); goto alloc_fail; }

    d->b_hash_data_len = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        hash_data_count * sizeof(uint16_t), (void*)hash_data_len, &err);
    if (err != CL_SUCCESS) { free(off64); goto alloc_fail; }

    free(off64);

    fprintf(stderr, "OpenCL GPU[%d]: compact table registered (%llu slots, %u hashes, %zuMB)\n",
            dev_idx, (unsigned long long)compact_size, (unsigned)hash_data_count,
            needed / (1024*1024));
    return 0;

alloc_fail:
    fprintf(stderr, "OpenCL GPU[%d]: buffer allocation failed (err=%d) - skipping\n", dev_idx, err);
    if (d->b_compact_fp) { clReleaseMemObject(d->b_compact_fp); d->b_compact_fp = NULL; }
    if (d->b_compact_idx) { clReleaseMemObject(d->b_compact_idx); d->b_compact_idx = NULL; }
    if (d->b_hash_data) { clReleaseMemObject(d->b_hash_data); d->b_hash_data = NULL; }
    if (d->b_hash_data_off) { clReleaseMemObject(d->b_hash_data_off); d->b_hash_data_off = NULL; }
    if (d->b_hash_data_len) { clReleaseMemObject(d->b_hash_data_len); d->b_hash_data_len = NULL; }
    return -1;
}

int gpu_opencl_set_salts(int dev_idx,
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
    clFinish(d->queue);
    d->salts_count = num_salts;
    return 0;
}

int gpu_opencl_set_overflow(int dev_idx,
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

void gpu_opencl_set_max_iter(int max_iter) { _max_iter = (max_iter < 1) ? 1 : max_iter; }
void gpu_opencl_set_mask_resume(uint32_t start) { _mask_resume = start; }
int gpu_opencl_has_resume(void) { return _mask_resume > 0 || _salt_resume > 0; }
static uint64_t _last_mask_start = 0;
uint64_t gpu_opencl_last_mask_start(void) { return _last_mask_start; }
void gpu_opencl_set_salt_resume(uint32_t start) { _salt_resume = start; }
void gpu_opencl_set_op(int op) { _gpu_op = op; }

/* Mask mode state — accessed by gpujob for hit reconstruction.
 * gpu_mask_desc layout: [sizes[n_total], tables[n_total][256]]
 * sizes[i] = character count for position i
 * tables[i] = 256-byte character table for position i
 * Kernel indexes: ch = mask_desc[n_total + i*256 + (idx % mask_desc[i])] */
#ifndef MAX_MASK_POS
#define MAX_MASK_POS 16
#endif
uint8_t gpu_mask_desc[MAX_MASK_POS + MAX_MASK_POS * 256];
uint8_t gpu_mask_sizes[MAX_MASK_POS];  /* for hit reconstruction */
int gpu_mask_n_prepend = 0;
int gpu_mask_n_append = 0;
uint64_t gpu_mask_total = 0;

int gpu_opencl_set_mask(const uint8_t *sizes, const uint8_t tables[][256],
                        int npre, int napp) {
    int ntotal = npre + napp;
    gpu_mask_n_prepend = npre;
    gpu_mask_n_append = napp;
    /* Pack: sizes first, then tables */
    memcpy(gpu_mask_desc, sizes, ntotal);
    memcpy(gpu_mask_sizes, sizes, ntotal);
    for (int i = 0; i < ntotal; i++)
        memcpy(gpu_mask_desc + ntotal + i * 256, tables[i], 256);
    gpu_mask_total = 1;
    for (int i = 0; i < ntotal; i++)
        gpu_mask_total *= sizes[i];
    /* Upload mask descriptor to all devices */
    int bufsize = ntotal + ntotal * 256;
    for (int i = 0; i < num_gpu_devs; i++) {
        struct gpu_device *d = &gpu_devs[i];
        if (d->bgpu_mask_desc) clReleaseMemObject(d->bgpu_mask_desc);
        cl_int err;
        d->bgpu_mask_desc = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         bufsize, gpu_mask_desc, &err);
    }
    fprintf(stderr, "OpenCL GPU: mask mode: %d prepend + %d append = %llu combinations\n",
            npre, napp, (unsigned long long)gpu_mask_total);
    return 0;
}

uint32_t *gpu_opencl_dispatch_batch(int dev_idx,
    const char *hexhashes, const uint16_t *hexlens,
    int num_words, int *nhits_out)
{
    *nhits_out = 0;
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || num_words <= 0) return NULL;
    struct gpu_device *d = &gpu_devs[dev_idx];
    if (!d->b_compact_fp) return NULL; /* compact table not loaded (insufficient memory) */
    /* Look up kernel for current op */
    int op = _gpu_op;
    int cat = gpu_op_category(op);
    struct gpu_kern *gk = (op >= 0 && op < MAX_GPU_KERNELS) ? &dev_kerns[dev_idx].kerns[op] : NULL;
    if (!gk || !gk->kernel) return NULL;

    /* Salted types need salts uploaded; iteration types do not */
    if ((cat == GPU_CAT_SALTED || cat == GPU_CAT_SALTPASS) && d->salts_count <= 0) return NULL;

    /* For salted iteration, use the salt-iter kernel; otherwise use the registered kernel */
    cl_kernel kern = (cat == GPU_CAT_SALTED && _max_iter > 1 && d->kern_salt_iter) ? d->kern_salt_iter : gk->kernel;
    cl_int err;

    /* Upload words — determine stride based on op type.
     * Unsalted pre-padded data from gpu_try_pack_unsalted uses per-type strides.
     * Salted password data uses 256-byte stride. */
    int word_stride;
    if ((cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED) &&
        (num_words > GPUBATCH_MAX || GPUForce)) {
        /* Pre-padded data at per-type stride (from gpu_try_pack_unsalted) */
        if (_gpu_op == JOB_SHA512 || _gpu_op == JOB_SHA384 ||
            _gpu_op == JOB_SHA512RAW || _gpu_op == JOB_SHA384RAW)
            word_stride = 128;
        else if (_gpu_op == JOB_MD6256)
            word_stride = 712;
        else if (_gpu_op == JOB_KECCAK224 || _gpu_op == JOB_SHA3_224)
            word_stride = 152;
        else if (_gpu_op == JOB_KECCAK256 || _gpu_op == JOB_SHA3_256)
            word_stride = 144;
        else if (_gpu_op == JOB_KECCAK384 || _gpu_op == JOB_SHA3_384)
            word_stride = 112;
        else if (_gpu_op == JOB_KECCAK512 || _gpu_op == JOB_SHA3_512)
            word_stride = 80;
        else
            word_stride = 64;
    } else {
        word_stride = 256;
    }
    size_t words_size = (size_t)num_words * word_stride;
    if (words_size > d->hexhash_cap) {
        if (d->b_hexhashes) clReleaseMemObject(d->b_hexhashes);
        if (d->b_hexlens) clReleaseMemObject(d->b_hexlens);
        d->b_hexhashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, words_size, NULL, &err);
        d->b_hexlens = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, num_words * sizeof(uint16_t), NULL, &err);
        d->hexhash_cap = words_size;
    }
    clEnqueueWriteBuffer(d->queue, d->b_hexhashes, CL_TRUE, 0, words_size, hexhashes, 0, NULL, NULL);
    /* For unsalted pre-padded batches (num_words > GPUBATCH_MAX), hexlens is not
     * used by the kernel — upload only what's available to keep the buffer valid */
    size_t hexlens_upload = ((num_words > GPUBATCH_MAX) ? GPUBATCH_MAX : num_words) * sizeof(uint16_t);
    clEnqueueWriteBuffer(d->queue, d->b_hexlens, CL_TRUE, 0, hexlens_upload, hexlens, 0, NULL, NULL);

    /* Params */
    OCLParams params;
    params.compact_mask = _compact_mask;
    params.num_words = num_words;
    params.num_salts = d->salts_count;
    params.salt_start = 0;
    params.max_probe = 256;
    params.hash_data_count = _hash_data_count;
    params.max_hits = GPU_MAX_HITS;
    params.overflow_count = _overflow_count;
    params.max_iter = (cat == GPU_CAT_ITER) ? (_max_iter - 1) : _max_iter;
    /* num_masks: 0 for salted types (unused), >0 for mask/unsalted.
     * Iteration-only (no mask) dispatches with num_masks=1. */
    params.num_masks = (cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED) ? 1 : 0;
    params.mask_start = 0;
    params.n_prepend = gpu_mask_n_prepend;
    params.n_append = gpu_mask_n_append;

    /* Zero hit counter */
    uint32_t zero = 0;
    if (p_clEnqueueFillBuffer)
        clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero, sizeof(zero), 0, sizeof(zero), 0, NULL, NULL);
    else
        clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
    clFinish(d->queue);

    /* Set kernel args */
    {
        int a = 0;
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexhashes);
        clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexlens);
        if (cat == GPU_CAT_MASK)
            clSetKernelArg(kern, a++, sizeof(cl_mem), &d->bgpu_mask_desc);
        else
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
    /* Dispatch — chunk salts/masks in blocks for GPUs with limited work capacity */
    size_t local = kern_get_local_size(gk);
    int is_salted = (cat == GPU_CAT_SALTED || cat == GPU_CAT_SALTPASS);
    int is_mask = ((cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED) && gpu_mask_total > 0);
    int total_salts = is_salted ? d->salts_count : 0;

    int salt_start_base = is_salted ? _salt_resume : 0;
    total_salts = is_salted ? (total_salts - salt_start_base) : 0;
    _salt_resume = 0;  /* consumed — reset for next dispatch */

    /* --- Timing-based dispatch sizing probe --- */
    int fam = gpu_op_family(op);
    int is_bcrypt = (fam == FAM_BCRYPT);
    if (fam >= 0 && fam < FAM_COUNT && !d->fam_timed[fam] && (is_mask || is_salted)) {
        /* First dispatch for this family: probe at increasing sizes to find
         * the chunk size that fills 200-400ms of GPU time. */
        uint32_t probe_size = is_bcrypt ? 1 : TIMING_INITIAL_SIZE;
        uint32_t best_size = probe_size;
        double best_ms = 0;

        for (int probe = 0; probe < 20; probe++) {
            /* Set up a probe dispatch */
            uint32_t zero_hit = 0;
            if (p_clEnqueueFillBuffer)
                clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero_hit, sizeof(zero_hit), 0, sizeof(zero_hit), 0, NULL, NULL);
            else
                clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero_hit), &zero_hit, 0, NULL, NULL);

            OCLParams probe_params = params;
            if (is_mask) {
                uint32_t pm = probe_size / (num_words > 0 ? num_words : 1);
                if (pm < 1) pm = 1;
                if ((uint64_t)pm > gpu_mask_total) pm = (gpu_mask_total < 0xFFFFFFFF) ? (uint32_t)gpu_mask_total : 0xFFFFFFFF;
                probe_params.num_masks = pm;
                probe_params.mask_start = 0;
            } else if (is_salted) {
                int ps = (int)(probe_size / (num_words > 0 ? num_words : 1));
                if (ps < 1) ps = 1;
                if (ps > d->salts_count) ps = d->salts_count;
                probe_params.num_salts = ps;
                probe_params.salt_start = salt_start_base;
            }
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(probe_params), &probe_params, 0, NULL, NULL);

            size_t probe_global = is_mask
                ? (size_t)num_words * probe_params.num_masks
                : (size_t)num_words * probe_params.num_salts;
            probe_global = ((probe_global + local - 1) / local) * local;

            struct timespec pt0, pt1;
            clock_gettime(CLOCK_MONOTONIC, &pt0);
            cl_int perr = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL, &probe_global, &local, 0, NULL, NULL);
            if (perr != CL_SUCCESS) break;
            clFinish(d->queue);
            clock_gettime(CLOCK_MONOTONIC, &pt1);
            double pms = (pt1.tv_sec - pt0.tv_sec) * 1e3 + (pt1.tv_nsec - pt0.tv_nsec) / 1e6;

            best_size = probe_size;
            best_ms = pms;

            if (is_bcrypt) {
                /* bcrypt: extrapolate from single-item timing */
                if (pms > 0) {
                    double per_item = pms;
                    uint32_t target = (uint32_t)(TIMING_BUDGET_MIN_MS / per_item);
                    if (target < 1) target = 1;
                    best_size = target;
                }
                break;
            }

            if (pms >= TIMING_BUDGET_MIN_MS && pms <= TIMING_BUDGET_MAX_MS)
                break;  /* in range */
            if (pms > TIMING_BUDGET_MAX_MS) {
                /* overshoot — halve and stop */
                best_size = probe_size / 2;
                if (best_size < TIMING_INITIAL_SIZE) best_size = TIMING_INITIAL_SIZE;
                break;
            }
            if (pms >= TIMING_WATCHDOG_MS)
                break;

            /* Below budget: double and try again */
            uint64_t next = (uint64_t)probe_size * 2;
            if (next > 0x7FFFFFFF) next = 0x7FFFFFFF;
            probe_size = (uint32_t)next;
        }

        d->fam_max_items[fam] = best_size;
        d->fam_timed[fam] = 1;
        fprintf(stderr, "OpenCL GPU[%d]: timed family %d: max_items=%u (%.1fms)\n",
                dev_idx, fam, best_size, best_ms);

        /* Reset hit counter after probe */
        uint32_t zero2 = 0;
        if (p_clEnqueueFillBuffer)
            clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero2, sizeof(zero2), 0, sizeof(zero2), 0, NULL, NULL);
        else
            clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero2), &zero2, 0, NULL, NULL);
        clFinish(d->queue);
    }

    /* Use timed chunk size if available, otherwise fall back to defaults */
    uint32_t timed_max = (fam >= 0 && fam < FAM_COUNT && d->fam_timed[fam])
        ? d->fam_max_items[fam] : 0;

    int salt_chunk = total_salts;
    if (timed_max > 0 && is_salted && num_words > 0) {
        salt_chunk = (int)(timed_max / num_words);
        if (salt_chunk < 1024) salt_chunk = 1024;
    } else if (d->max_dispatch > 0 && is_salted && num_words > 0) {
        salt_chunk = d->max_dispatch / num_words;
        if (salt_chunk < 1024) salt_chunk = 1024;
    }

#define MASK_CHUNK_DEFAULT 4194304  /* 4M mask combinations per dispatch (fallback) */
    uint64_t mask_start_base = is_mask ? _mask_resume : 0;
    uint64_t mask_total = is_mask ? (gpu_mask_total - mask_start_base) : 0;
    uint64_t mask_chunk;
    if (is_mask) {
        if (timed_max > 0 && num_words > 0)
            mask_chunk = timed_max / num_words;
        else
            mask_chunk = MASK_CHUNK_DEFAULT;
        if (mask_chunk < 1) mask_chunk = 1;
    } else {
        mask_chunk = 0;
    }
    if (mask_chunk > mask_total) mask_chunk = mask_total;
    _mask_resume = 0;  /* consumed — reset for next dispatch */

    /* Brute-force multi-GPU: use atomic cursor instead of sequential chunks.
     * Each GPU claims mask_chunk-sized pieces from _bf_cursor until keyspace
     * is exhausted.  Non-brute-force dispatches use the original chunk loop. */
    int bf_mode = (__sync_add_and_fetch(&_bf_active, 0) && is_mask);

    int num_chunks;
    if (bf_mode)
        num_chunks = 0;  /* not used — loop controlled by atomic cursor */
    else if (is_mask)
        num_chunks = (int)((mask_total + mask_chunk - 1) / mask_chunk);
    else if (is_salted)
        num_chunks = (total_salts + salt_chunk - 1) / salt_chunk;
    else
        num_chunks = 1;

    struct timespec t0, t1;
    if (!gk->tuned) clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int chunk = 0; bf_mode ? 1 : (chunk < num_chunks); chunk++) {
        if (bf_mode) {
            /* Atomically claim next chunk from shared cursor */
            uint64_t my_start = __sync_fetch_and_add(&_bf_cursor, mask_chunk);
            if (my_start >= gpu_mask_total) break;  /* keyspace exhausted */
            uint64_t remaining = gpu_mask_total - my_start;
            if (remaining > mask_chunk) remaining = mask_chunk;
            params.mask_start = my_start;
            params.num_masks = (uint32_t)remaining;
            /* Pre-decompose mask_start into per-position base offsets */
            { uint64_t ms = my_start;
              uint64_t b0 = 0, b1 = 0;
              int ntot = gpu_mask_n_prepend + gpu_mask_n_append;
              for (int bi = ntot - 1; bi >= 0; bi--) {
                uint8_t sz = gpu_mask_sizes[bi];
                uint8_t digit = (uint8_t)(ms % sz);
                ms /= sz;
                if (bi < 8) b0 |= (uint64_t)digit << (bi * 8);
                else        b1 |= (uint64_t)digit << ((bi - 8) * 8);
              }
              params.mask_base0 = b0;
              params.mask_base1 = b1;
            }
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);
        } else if (is_mask) {
            params.mask_start = mask_start_base + (uint64_t)chunk * mask_chunk;
            { uint64_t remaining = mask_total - (uint64_t)chunk * mask_chunk;
              if (remaining > mask_chunk) remaining = mask_chunk;
              params.num_masks = (uint32_t)remaining;
            }
            /* Pre-decompose mask_start into per-position base offsets */
            { uint64_t ms = params.mask_start;
              uint64_t b0 = 0, b1 = 0;
              int ntot = gpu_mask_n_prepend + gpu_mask_n_append;
              for (int bi = ntot - 1; bi >= 0; bi--) {
                uint8_t sz = gpu_mask_sizes[bi];
                uint8_t digit = (uint8_t)(ms % sz);
                ms /= sz;
                if (bi < 8) b0 |= (uint64_t)digit << (bi * 8);
                else        b1 |= (uint64_t)digit << ((bi - 8) * 8);
              }
              params.mask_base0 = b0;
              params.mask_base1 = b1;
            }
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);
        } else if (is_salted) {
            params.salt_start = salt_start_base + chunk * salt_chunk;
            params.num_salts = total_salts - chunk * salt_chunk;
            if (params.num_salts > salt_chunk) params.num_salts = salt_chunk;
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);
        } else if (chunk == 0) {
            clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);
        }

        size_t global = is_mask
            ? (size_t)num_words * params.num_masks
            : is_salted
            ? (size_t)num_words * params.num_salts
            : (size_t)num_words;
        global = ((global + local - 1) / local) * local;

        /* Zero hit counter before each chunk */
        { uint32_t zero = 0;
          if (p_clEnqueueFillBuffer)
            clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero, sizeof(zero), 0, sizeof(zero), 0, NULL, NULL);
          else
            clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
        }

        err = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL GPU[%d] dispatch error: %d (chunk %d/%d global=%zu)\n",
                    dev_idx, err, chunk, num_chunks, global);
            return NULL;
        }
        { cl_int ferr = clFinish(d->queue);
          if (ferr != CL_SUCCESS) {
            fprintf(stderr, "OpenCL GPU[%d] clFinish error: %d (chunk %d/%d)\n",
                    dev_idx, ferr, chunk, num_chunks);
            break;
          }
        }

        /* Update Tothash for progress tracking */
        { extern unsigned long long Tothash;
          size_t actual = is_mask ? (size_t)num_words * params.num_masks
                        : is_salted ? (size_t)num_words * params.num_salts
                        : (size_t)num_words;
          __sync_fetch_and_add(&Tothash, actual);
        }

        /* Check for hits — return immediately if any found */
        { uint32_t chunk_hits;
          clEnqueueReadBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(chunk_hits), &chunk_hits, 0, NULL, NULL);
          if (chunk_hits > 0) {
            *nhits_out = (int)chunk_hits;
            if (chunk_hits > GPU_MAX_HITS) chunk_hits = GPU_MAX_HITS;
            int cat = gpu_op_category(_gpu_op);
            int is_md5crypt = (_gpu_op == JOB_MD5CRYPT || _gpu_op == JOB_PHPBB3);
            int is_bcrypt = (_gpu_op == JOB_BCRYPT);
            int hw = is_bcrypt ? 6 : gpu_hash_words(_gpu_op);
            int stride = is_md5crypt ? 6
                : is_bcrypt ? 9
                : (_max_iter > 1 || cat == GPU_CAT_ITER || cat == GPU_CAT_SALTPASS)
                    ? (3 + hw) : (2 + hw);
            clEnqueueReadBuffer(d->queue, d->b_hits, CL_TRUE, 0,
                chunk_hits * stride * sizeof(uint32_t), d->h_hits, 0, NULL, NULL);
            /* Set resume point and record mask_start for hit reconstruction */
            _last_mask_start = params.mask_start;
            if (bf_mode) {
                /* Brute-force: cursor already advanced; signal caller to
                 * re-enter dispatch_batch (cursor handles continuation). */
                _mask_resume = _bf_cursor < gpu_mask_total ? _bf_cursor : 0;
            } else if (is_mask) {
                uint64_t next = mask_start_base + (uint64_t)(chunk + 1) * mask_chunk;
                if (next < gpu_mask_total)
                    _mask_resume = next;
            }
            return d->h_hits;
          }
        }
    }

    if (!gk->tuned) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        kern_record_time(gk, ms);
    }

    /* All chunks completed — check final hit count (salted single-chunk etc.) */
    *nhits_out = 0;
    uint32_t raw_nhits;
    clEnqueueReadBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(raw_nhits), &raw_nhits, 0, NULL, NULL);
    *nhits_out = (int)raw_nhits;
    if (raw_nhits > 0) {
        if (raw_nhits > GPU_MAX_HITS) raw_nhits = GPU_MAX_HITS;
        int cat = gpu_op_category(_gpu_op);
        int is_md5crypt = (_gpu_op == JOB_MD5CRYPT || _gpu_op == JOB_PHPBB3);
        int is_bcrypt = (_gpu_op == JOB_BCRYPT);
        int hw = is_bcrypt ? 6 : gpu_hash_words(_gpu_op);
        int stride = is_md5crypt ? 6
            : is_bcrypt ? 9
            : (_max_iter > 1 || cat == GPU_CAT_ITER || cat == GPU_CAT_SALTPASS)
                ? (3 + hw) : (2 + hw);
        clEnqueueReadBuffer(d->queue, d->b_hits, CL_TRUE, 0, raw_nhits * stride * sizeof(uint32_t), d->h_hits, 0, NULL, NULL);
    }
    return d->h_hits;
}

#endif /* OPENCL_GPU */
