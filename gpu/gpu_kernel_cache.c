/* $Header: /Users/dlr/src/mdfind/gpu/RCS/gpu_kernel_cache.c,v 1.6 2026/05/04 07:54:59 dlr Exp dlr $
 *
 * Implementation. See gpu_kernel_cache.h for the design overview.
 */

/* opencl_dynload.h must be included BEFORE gpu_kernel_cache.h: the
 * cache header's cl_program / cl_context declarations are gated on
 * cl.h having already been seen (see gpu_kernel_cache.h). dynload.h
 * pulls in cl.h, satisfying that. dynload.h also redirects cl* calls
 * through the runtime dlopen'd function-pointer table that gpu_opencl.c
 * uses, avoiding "undefined reference to clGetDeviceInfo" etc. */
#include "opencl_dynload.h"
#include "gpu_kernel_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "openssl/evp.h"

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#include <windows.h>
/* mingw provides mkdir(path) as the POSIX-flavored single-arg wrapper;
 * <direct.h> provides _mkdir(path) for the MSVC name. Use the explicitly
 * Windows-named form to avoid clashing with POSIX mkdir(path,mode). */
#define MKDIR(p)        _mkdir(p)
#define CLOSE(fd)       _close(fd)
#define UNLINK(p)       _unlink(p)
#define ACCESS(p, m)    _access(p, m)
#define RENAME_REPL(a, b) (MoveFileExA((a), (b), MOVEFILE_REPLACE_EXISTING) ? 0 : -1)
#define DIRSEP_NATIVE   '\\'
#else
#include <sys/file.h>
#include <unistd.h>
#include <dirent.h>
#define MKDIR(p)        mkdir((p), 0755)
#define CLOSE(fd)       close(fd)
#define UNLINK(p)       unlink(p)
#define ACCESS(p, m)    access(p, m)
#define RENAME_REPL(a, b) rename((a), (b))
#define DIRSEP_NATIVE   '/'
#endif

/* ============================================================
 * Path helpers
 * ============================================================ */

/* Returns length of directory prefix in `path`, NOT including the
 * trailing separator. 0 means "no directory component — use cwd".
 * Handles POSIX, Windows, mixed, drive-letter, and UNC paths. */
static size_t path_dir_len(const char *path) {
    if (!path || !*path) return 0;
    const char *fwd = strrchr(path, '/');
    const char *bck = strrchr(path, '\\');
    const char *sep = (fwd && bck) ? (fwd > bck ? fwd : bck)
                    : (fwd ? fwd : bck);
    if (!sep) return 0;
    if (sep == path) return 1;
    return (size_t)(sep - path);
}

/* ============================================================
 * SHA-256 helpers (OpenSSL EVP)
 * ============================================================ */

/* Compute SHA-256 over a sequence of (ptr, len) chunks. `out` is 32 bytes. */
static void sha256_chunks(const void * const *ptrs, const size_t *lens,
                          int n_chunks, unsigned char out[32]) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    for (int i = 0; i < n_chunks; i++)
        EVP_DigestUpdate(ctx, ptrs[i], lens[i]);
    unsigned int outlen = 32;
    EVP_DigestFinal_ex(ctx, out, &outlen);
    EVP_MD_CTX_free(ctx);
}

static void hex_encode(const unsigned char *bin, size_t binlen, char *out) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < binlen; i++) {
        out[i*2]     = hex[(bin[i] >> 4) & 0xf];
        out[i*2 + 1] = hex[bin[i] & 0xf];
    }
    out[binlen*2] = 0;
}

static int hex_decode(const char *hex, unsigned char *out, size_t outlen) {
    for (size_t i = 0; i < outlen; i++) {
        char hi = hex[i*2], lo = hex[i*2 + 1];
        int hv = (hi >= '0' && hi <= '9') ? hi - '0'
               : (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10
               : (hi >= 'A' && hi <= 'F') ? hi - 'A' + 10
               : -1;
        int lv = (lo >= '0' && lo <= '9') ? lo - '0'
               : (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10
               : (lo >= 'A' && lo <= 'F') ? lo - 'A' + 10
               : -1;
        if (hv < 0 || lv < 0) return -1;
        out[i] = (unsigned char)((hv << 4) | lv);
    }
    return 0;
}

/* ============================================================
 * Cross-platform file lock
 * ============================================================ */

/* Acquire exclusive lock on `path` (creating if missing). Returns the
 * fd/handle (cast to int) on success, -1 on failure. Pass to
 * lock_release() to release + close. */
static int lock_acquire(const char *path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov = {0};
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0,
                    MAXDWORD, MAXDWORD, &ov)) {
        CloseHandle(h);
        return -1;
    }
    /* Stash HANDLE in an int slot — Win64 HANDLEs fit in 32 bits in
     * practice; we round-trip via intptr_t. */
    return (int)(intptr_t)h;
#else
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        return -1;
    }
    return fd;
#endif
}

static void lock_release(int slot) {
    if (slot < 0) return;
#ifdef _WIN32
    HANDLE h = (HANDLE)(intptr_t)slot;
    OVERLAPPED ov = {0};
    UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov);
    CloseHandle(h);
#else
    flock(slot, LOCK_UN);
    close(slot);
#endif
}

/* ============================================================
 * Cache state (init-time)
 * ============================================================ */

static int  CacheInited   = 0;
static int  CacheEnabled  = 0;
static char CacheDir[4096] = "";   /* <dir>/gpu-kernels (the leaf dir) */
static char MdxRev[64]     = "";   /* mdxfind binary rev (for the key + cache.version) */

/* Build the kernel-cache directory path from MDXFIND_CACHE. Stores into
 * CacheDir[]. Returns 0 on success, -1 if MDXFIND_CACHE unset/empty. */
static int derive_cache_dir(void) {
    const char *e = getenv("MDXFIND_CACHE");
    if (!e || !*e) return -1;

    size_t dlen = path_dir_len(e);
    if (dlen == 0) {
        /* No directory in MDXFIND_CACHE — kernel cache would land in cwd
         * which is the dangerous case the user explicitly rejected.
         * Treat as disabled. */
        return -1;
    }

    char sep = e[dlen];                     /* the separator we found */
    int n = snprintf(CacheDir, sizeof(CacheDir), "%.*s%cgpu-kernels",
                     (int)dlen, e, sep);
    if (n <= 0 || n >= (int)sizeof(CacheDir)) return -1;
    return 0;
}

/* mkdir -p style: creates a single leaf, ignores EEXIST. */
static int ensure_dir(const char *path) {
    if (MKDIR(path) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

/* Read first line of a file into buf (NUL-trimmed, no newline).
 * Returns 0 if read OK and buf populated, -1 otherwise. */
static int read_first_line(const char *path, char *buf, size_t buflen) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    if (!fgets(buf, (int)buflen, fp)) { fclose(fp); return -1; }
    fclose(fp);
    /* Trim trailing newline / whitespace */
    size_t len = strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'
                       || buf[len-1] == ' ' || buf[len-1] == '\t')) {
        buf[--len] = 0;
    }
    return 0;
}

/* Write a single line to `path` atomically (.tmp + rename). */
static int write_line_atomic(const char *path, const char *line) {
    char tmp[4096];
    int n = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    if (n <= 0 || n >= (int)sizeof(tmp)) return -1;
    FILE *fp = fopen(tmp, "w");
    if (!fp) return -1;
    fprintf(fp, "%s\n", line);
    fflush(fp);
    fclose(fp);
    if (RENAME_REPL(tmp, path) != 0) {
        UNLINK(tmp);
        return -1;
    }
    return 0;
}

/* Build a cache-relative path: "<CacheDir><DIRSEP><name>". */
static void cache_path(char *out, size_t outlen, const char *name) {
    snprintf(out, outlen, "%s%c%s", CacheDir, DIRSEP_NATIVE, name);
}

/* Under .cache.lock, compare on-disk cache.version with MdxRev.
 * If different (or missing), unlink all *.bin and *.meta in CacheDir
 * and write fresh cache.version. */
static void cache_version_check_and_invalidate(void) {
    char lockpath[4096], verpath[4096];
    cache_path(lockpath, sizeof(lockpath), ".cache.lock");
    cache_path(verpath,  sizeof(verpath),  "cache.version");

    int lk = lock_acquire(lockpath);
    if (lk < 0) {
        fprintf(stderr,
            "Warning: GPU kernel cache: cannot acquire lock %s — cache disabled this session\n",
            lockpath);
        CacheEnabled = 0;
        return;
    }

    char on_disk[64] = "";
    int have = (read_first_line(verpath, on_disk, sizeof(on_disk)) == 0);

    if (!have || strcmp(on_disk, MdxRev) != 0) {
        /* Mismatch (or first-ever cache here). Sweep entries and rewrite
         * version. We only delete *.bin and *.meta (not the lock file
         * we hold, not anything else a peer might have stashed). */
#ifdef _WIN32
        char glob[4096];
        snprintf(glob, sizeof(glob), "%s\\*", CacheDir);
        WIN32_FIND_DATAA fd;
        HANDLE hf = FindFirstFileA(glob, &fd);
        if (hf != INVALID_HANDLE_VALUE) {
            do {
                size_t nlen = strlen(fd.cFileName);
                int is_bin  = (nlen >= 4 && strcmp(fd.cFileName + nlen - 4, ".bin")  == 0);
                int is_meta = (nlen >= 5 && strcmp(fd.cFileName + nlen - 5, ".meta") == 0);
                int is_lock = (nlen >= 5 && strcmp(fd.cFileName + nlen - 5, ".lock") == 0
                               && strcmp(fd.cFileName, ".cache.lock") != 0);
                if (is_bin || is_meta || is_lock) {
                    char full[4096];
                    cache_path(full, sizeof(full), fd.cFileName);
                    UNLINK(full);
                }
            } while (FindNextFileA(hf, &fd));
            FindClose(hf);
        }
#else
        DIR *dp = opendir(CacheDir);
        if (dp) {
            struct dirent *de;
            while ((de = readdir(dp)) != NULL) {
                size_t nlen = strlen(de->d_name);
                int is_bin  = (nlen >= 4 && strcmp(de->d_name + nlen - 4, ".bin")  == 0);
                int is_meta = (nlen >= 5 && strcmp(de->d_name + nlen - 5, ".meta") == 0);
                int is_lock = (nlen >= 5 && strcmp(de->d_name + nlen - 5, ".lock") == 0
                               && strcmp(de->d_name, ".cache.lock") != 0);
                if (is_bin || is_meta || is_lock) {
                    char full[4096];
                    cache_path(full, sizeof(full), de->d_name);
                    UNLINK(full);
                }
            }
            closedir(dp);
        }
#endif
        if (write_line_atomic(verpath, MdxRev) == 0) {
            fprintf(stderr,
                "GPU kernel cache: rev %s%s — invalidated, will recompile\n",
                have ? "changed from " : "first session at ",
                have ? on_disk : MdxRev);
        } else {
            fprintf(stderr,
                "Warning: GPU kernel cache: cannot write %s — cache disabled this session\n",
                verpath);
            CacheEnabled = 0;
        }
    }

    lock_release(lk);
}

int gpu_kernel_cache_init(const char *mdxfind_rev) {
    if (CacheInited) return CacheEnabled ? 0 : -1;
    CacheInited = 1;

    if (!mdxfind_rev || !*mdxfind_rev) {
        fprintf(stderr,
            "Warning: GPU kernel cache: empty mdxfind rev — cache disabled\n");
        CacheEnabled = 0;
        return -1;
    }
    snprintf(MdxRev, sizeof(MdxRev), "%s", mdxfind_rev);

    if (derive_cache_dir() != 0) {
        fprintf(stderr,
            "Warning: MDXFIND_CACHE not set (or no directory component) — "
            "GPU kernel cache disabled (will JIT every session). "
            "Set MDXFIND_CACHE=/path/to/mdxfind.db to enable.\n");
        CacheEnabled = 0;
        return -1;
    }

    if (ensure_dir(CacheDir) != 0) {
        fprintf(stderr,
            "Warning: GPU kernel cache: cannot create %s (%s) — cache disabled\n",
            CacheDir, strerror(errno));
        CacheEnabled = 0;
        return -1;
    }

    CacheEnabled = 1;
    cache_version_check_and_invalidate();
    if (!CacheEnabled) return -1;

    fprintf(stderr, "GPU kernel cache: enabled at %s (rev %s)\n",
            CacheDir, MdxRev);
    return 0;
}

int gpu_kernel_cache_enabled(void) {
    return CacheInited && CacheEnabled;
}

/* ============================================================
 * Per-entry: key derivation, store, load, evict
 * ============================================================ */

/* Compute the 24-hex-char cache key for the given inputs.
 *
 * Memo B B2 R3 mitigation: `defines_str` is hashed in alongside the
 * sources. Pass NULL to mean "no defines" -- in that case the chunk
 * is omitted entirely so the resulting key is bit-identical to the
 * pre-B2 key (existing cache entries do not need to be invalidated).
 * Pass "" to also produce the bit-identical key (zero-length string
 * contributes nothing to the SHA-256 transcript). */
static void compute_key(cl_uint n_sources, const char **sources,
                        const char *defines_str,
                        const char *device_name, const char *driver_version,
                        const char *cl_version, const char *mdx_rev,
                        char key_out[25]) {
    /* Upper bound: n_sources + 5 metadata chunks (was 4, +1 for defines_str). */
    enum { MAX_CHUNKS = 32 };
    const void *ptrs[MAX_CHUNKS];
    size_t      lens[MAX_CHUNKS];
    int         nc = 0;
    if ((cl_uint)MAX_CHUNKS - 5 < n_sources) {
        /* Should never happen in practice (we have 2-4 sources max). */
        n_sources = MAX_CHUNKS - 5;
    }
    for (cl_uint i = 0; i < n_sources; i++) {
        ptrs[nc] = sources[i];
        lens[nc] = strlen(sources[i]);
        nc++;
    }
    /* defines_str: only mix in if non-NULL AND non-empty. The key
     * stability rule (NULL/"" -> pre-B2-identical key) is what allows
     * shipping B2's cache-key change without invalidating existing
     * cache entries for non-template builds. */
    if (defines_str && defines_str[0] != '\0') {
        ptrs[nc] = defines_str;
        lens[nc] = strlen(defines_str);
        nc++;
    }
    ptrs[nc] = device_name;     lens[nc] = strlen(device_name);     nc++;
    ptrs[nc] = driver_version;  lens[nc] = strlen(driver_version);  nc++;
    ptrs[nc] = cl_version;      lens[nc] = strlen(cl_version);      nc++;
    ptrs[nc] = mdx_rev;         lens[nc] = strlen(mdx_rev);         nc++;

    unsigned char digest[32];
    sha256_chunks(ptrs, lens, nc, digest);
    hex_encode(digest, 12, key_out);   /* 12 bytes -> 24 hex chars */
}

/* Read entire file into a malloc'd buffer. Returns 0 on success and
 * sets *out_buf, *out_len. -1 on error. */
static int read_file_all(const char *path, void **out_buf, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return -1; }
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return -1; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return -1; }
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(fp); return -1; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) { free(buf); return -1; }
    *out_buf = buf;
    *out_len = (size_t)sz;
    return 0;
}

/* Parse a "key=value\n..." text into a value for the given key. Writes
 * up to outlen-1 bytes + NUL into out. Returns 0 on found, -1 on miss. */
static int meta_get_value(const char *meta, const char *key,
                          char *out, size_t outlen) {
    size_t klen = strlen(key);
    const char *p = meta;
    while (*p) {
        const char *eol = strchr(p, '\n');
        size_t llen = eol ? (size_t)(eol - p) : strlen(p);
        if (llen > klen + 1 && strncmp(p, key, klen) == 0 && p[klen] == '=') {
            size_t vlen = llen - klen - 1;
            if (vlen >= outlen) vlen = outlen - 1;
            memcpy(out, p + klen + 1, vlen);
            out[vlen] = 0;
            return 0;
        }
        if (!eol) break;
        p = eol + 1;
    }
    return -1;
}

/* Unlink both <key>.bin and <key>.meta. Logs to stderr. The <key>.lock
 * (if present) is intentionally left in place — it is the persistent
 * lock-target and never goes away outside cache-wide invalidation. */
static void evict_entry(const char *key, const char *reason) {
    char binp[4096], metp[4096];
    snprintf(binp, sizeof(binp), "%s%c%s.bin",  CacheDir, DIRSEP_NATIVE, key);
    snprintf(metp, sizeof(metp), "%s%c%s.meta", CacheDir, DIRSEP_NATIVE, key);
    UNLINK(binp);
    UNLINK(metp);
    fprintf(stderr, "GPU kernel cache: evicted %s (%s)\n", key, reason);
}

/* Try to load + verify + clCreateProgramWithBinary for `key`.
 * On success: returns the cl_program, *err_out = CL_SUCCESS.
 * On any failure: evicts the entry, returns NULL.
 * On miss (no file): returns NULL, *err_out = CL_INVALID_PROGRAM. */
static cl_program try_load(const char *key, cl_context ctx, cl_device_id dev,
                           const char *build_opts, cl_int *err_out) {
    char binp[4096], metp[4096];
    snprintf(binp, sizeof(binp), "%s%c%s.bin",  CacheDir, DIRSEP_NATIVE, key);
    snprintf(metp, sizeof(metp), "%s%c%s.meta", CacheDir, DIRSEP_NATIVE, key);

    if (ACCESS(binp, 0) != 0) {
        if (err_out) *err_out = CL_INVALID_PROGRAM;
        return NULL;
    }

    void *bin = NULL; size_t binlen = 0;
    if (read_file_all(binp, &bin, &binlen) != 0 || binlen == 0) {
        evict_entry(key, "binary file unreadable");
        if (err_out) *err_out = CL_INVALID_BINARY;
        return NULL;
    }

    /* Verify SHA-256 from .meta (if present). Missing meta is treated
     * as a load failure to keep .bin and .meta paired. */
    void *meta_buf = NULL; size_t meta_len = 0;
    if (read_file_all(metp, &meta_buf, &meta_len) != 0) {
        free(bin);
        evict_entry(key, "meta missing");
        if (err_out) *err_out = CL_INVALID_BINARY;
        return NULL;
    }
    /* Ensure NUL-terminated for meta_get_value. */
    char *meta_str = (char *)realloc(meta_buf, meta_len + 1);
    if (!meta_str) { free(bin); free(meta_buf); evict_entry(key, "oom on meta"); if (err_out) *err_out = CL_OUT_OF_HOST_MEMORY; return NULL; }
    meta_str[meta_len] = 0;
    meta_buf = meta_str;

    char stored_hex[65] = "";
    if (meta_get_value(meta_str, "binary_sha256", stored_hex, sizeof(stored_hex)) != 0
        || strlen(stored_hex) != 64) {
        free(bin); free(meta_buf);
        evict_entry(key, "meta missing binary_sha256");
        if (err_out) *err_out = CL_INVALID_BINARY;
        return NULL;
    }
    unsigned char stored_dig[32];
    if (hex_decode(stored_hex, stored_dig, 32) != 0) {
        free(bin); free(meta_buf);
        evict_entry(key, "meta sha256 hex malformed");
        if (err_out) *err_out = CL_INVALID_BINARY;
        return NULL;
    }
    free(meta_buf);

    unsigned char actual_dig[32];
    const void *bin_chunks[1] = { bin };
    size_t       bin_lens[1]  = { binlen };
    sha256_chunks(bin_chunks, bin_lens, 1, actual_dig);
    if (memcmp(stored_dig, actual_dig, 32) != 0) {
        free(bin);
        evict_entry(key, "binary sha256 mismatch");
        if (err_out) *err_out = CL_INVALID_BINARY;
        return NULL;
    }

    cl_int berr = CL_SUCCESS, err = CL_SUCCESS;
    const unsigned char *bin_ptr = (const unsigned char *)bin;
    cl_program prog = clCreateProgramWithBinary(ctx, 1, &dev,
                                                &binlen, &bin_ptr,
                                                &berr, &err);
    free(bin);
    if (err != CL_SUCCESS || berr != CL_SUCCESS || !prog) {
        if (prog) clReleaseProgram(prog);
        evict_entry(key, "clCreateProgramWithBinary rejected entry");
        if (err_out) *err_out = err;
        return NULL;
    }

    err = clBuildProgram(prog, 1, &dev, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        evict_entry(key, "clBuildProgram on cached binary failed");
        if (err_out) *err_out = err;
        return NULL;
    }

    if (err_out) *err_out = CL_SUCCESS;
    return prog;
}

/* After successful source-compile + clBuildProgram, store the binary
 * to <key>.bin + sidecar to <key>.meta. */
static void store_entry(const char *key, cl_program prog, cl_device_id dev,
                        const char *device_name, const char *driver_version,
                        const char *cl_version) {
    /* Get binary size + bytes for our (single) device. */
    size_t bin_size = 0;
    cl_int err = clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                                   sizeof(bin_size), &bin_size, NULL);
    if (err != CL_SUCCESS || bin_size == 0) {
        fprintf(stderr,
            "GPU kernel cache: store %s skipped (binary size 0 / err=%d)\n",
            key, err);
        return;
    }
    unsigned char *bin = (unsigned char *)malloc(bin_size);
    if (!bin) return;
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bin), &bin, NULL);
    if (err != CL_SUCCESS) {
        free(bin);
        fprintf(stderr,
            "GPU kernel cache: store %s skipped (CL_PROGRAM_BINARIES err=%d)\n",
            key, err);
        return;
    }

    /* Atomic write of .bin via .tmp + rename. */
    char binp[4096], binp_tmp[4096];
    snprintf(binp,     sizeof(binp),     "%s%c%s.bin",     CacheDir, DIRSEP_NATIVE, key);
    snprintf(binp_tmp, sizeof(binp_tmp), "%s%c%s.bin.tmp", CacheDir, DIRSEP_NATIVE, key);
    FILE *fp = fopen(binp_tmp, "wb");
    if (!fp) { free(bin); return; }
    size_t wr = fwrite(bin, 1, bin_size, fp);
    fflush(fp);
    fclose(fp);
    if (wr != bin_size) { UNLINK(binp_tmp); free(bin); return; }
    if (RENAME_REPL(binp_tmp, binp) != 0) { UNLINK(binp_tmp); free(bin); return; }

    /* Compute SHA-256 of the binary for .meta. */
    unsigned char dig[32];
    char dighex[65];
    const void *chunks[1] = { bin }; size_t clens[1] = { bin_size };
    sha256_chunks(chunks, clens, 1, dig);
    hex_encode(dig, 32, dighex);
    free(bin);

    /* Write .meta atomically. */
    char metp[4096], metp_tmp[4096];
    snprintf(metp,     sizeof(metp),     "%s%c%s.meta",     CacheDir, DIRSEP_NATIVE, key);
    snprintf(metp_tmp, sizeof(metp_tmp), "%s%c%s.meta.tmp", CacheDir, DIRSEP_NATIVE, key);
    fp = fopen(metp_tmp, "w");
    if (!fp) return;
    fprintf(fp,
        "key=%s\n"
        "mdxfind_rev=%s\n"
        "device=%s\n"
        "driver=%s\n"
        "cl_version=%s\n"
        "binary_size=%zu\n"
        "binary_sha256=%s\n",
        key, MdxRev, device_name, driver_version, cl_version, bin_size, dighex);
    fflush(fp);
    fclose(fp);
    if (RENAME_REPL(metp_tmp, metp) != 0) { UNLINK(metp_tmp); return; }
}

/* ============================================================
 * Public: build (or load from cache) a program
 * ============================================================ */

static void get_str_info(cl_device_id dev, cl_device_info what, char *out, size_t outlen) {
    out[0] = 0;
    clGetDeviceInfo(dev, what, outlen, out, NULL);
    out[outlen - 1] = 0;
}

cl_program gpu_kernel_cache_build_program(
    cl_context ctx, cl_device_id dev,
    cl_uint n_sources, const char **sources,
    const char *build_opts,
    cl_int *err_out)
{
    /* Thin wrapper: pre-B2 callers get NULL defines_str, which means
     * "no defines" and produces a cache key bit-identical to before
     * the R3 fix landed. New template-instantiation callers use the
     * _ex entry point with their own defines_str. */
    return gpu_kernel_cache_build_program_ex(
        ctx, dev, n_sources, sources, build_opts, NULL, err_out);
}

cl_program gpu_kernel_cache_build_program_ex(
    cl_context ctx, cl_device_id dev,
    cl_uint n_sources, const char **sources,
    const char *build_opts,
    const char *defines_str,
    cl_int *err_out)
{
    cl_int err = CL_SUCCESS;
    cl_program prog = NULL;

    /* Disabled-cache path: plain compile-from-source. Caller checks
     * *err_out and pulls the build log from `prog` on non-CL_SUCCESS. */
    if (!gpu_kernel_cache_enabled()) {
        prog = clCreateProgramWithSource(ctx, n_sources, sources, NULL, &err);
        if (!prog) { if (err_out) *err_out = err; return NULL; }
        err = clBuildProgram(prog, 1, &dev, build_opts ? build_opts : "", NULL, NULL);
        if (err_out) *err_out = err;
        return prog;
    }

    /* Cache-enabled path. Compute key + names. */
    char device_name[256] = "", driver_version[128] = "", cl_version[128] = "";
    get_str_info(dev, CL_DEVICE_NAME,    device_name,    sizeof(device_name));
    get_str_info(dev, CL_DRIVER_VERSION, driver_version, sizeof(driver_version));
    get_str_info(dev, CL_DEVICE_VERSION, cl_version,     sizeof(cl_version));

    char key[25];
    compute_key(n_sources, sources, defines_str,
                device_name, driver_version, cl_version,
                MdxRev, key);

    /* Phase 1: lock-free cache load attempt (warm path). */
    prog = try_load(key, ctx, dev, build_opts ? build_opts : "", &err);
    if (prog) {
        if (err_out) *err_out = CL_SUCCESS;
        return prog;
    }

    /* Phase 2: cache miss (or failure-evicted). Take per-entry lock,
     * recheck (a peer may have just finished compiling), and either
     * load-or-compile.
     *
     * Lock target is <key>.lock — a dedicated never-unlinked sentinel
     * file. Using .meta would have a subtle race: store_entry's atomic
     * rename of .meta orphans the lock holder's fd onto the old inode
     * while peers see the new inode at the same path, so flocks would
     * land on different inodes and not serialize. .lock dodges this. */
    char lockp[4096];
    snprintf(lockp, sizeof(lockp), "%s%c%s.lock", CacheDir, DIRSEP_NATIVE, key);
    int lk = lock_acquire(lockp);
    if (lk < 0) {
        /* Couldn't lock — fall through to source compile without storing. */
        fprintf(stderr,
            "GPU kernel cache: lock %s failed — compiling without cache update\n",
            lockp);
        prog = clCreateProgramWithSource(ctx, n_sources, sources, NULL, &err);
        if (!prog) { if (err_out) *err_out = err; return NULL; }
        err = clBuildProgram(prog, 1, &dev, build_opts ? build_opts : "", NULL, NULL);
        if (err_out) *err_out = err;
        return prog;
    }

    /* Inside lock: recheck. */
    cl_int recheck_err = CL_SUCCESS;
    prog = try_load(key, ctx, dev, build_opts ? build_opts : "", &recheck_err);
    if (prog) {
        lock_release(lk);
        if (err_out) *err_out = CL_SUCCESS;
        return prog;
    }

    /* We're the compiler. Source-compile + store. */
    prog = clCreateProgramWithSource(ctx, n_sources, sources, NULL, &err);
    if (!prog) {
        lock_release(lk);
        if (err_out) *err_out = err;
        return NULL;
    }
    err = clBuildProgram(prog, 1, &dev, build_opts ? build_opts : "", NULL, NULL);
    if (err == CL_SUCCESS) {
        store_entry(key, prog, dev, device_name, driver_version, cl_version);
    }
    lock_release(lk);
    if (err_out) *err_out = err;
    return prog;
}
