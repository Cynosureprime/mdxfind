/*
 * hx_dump.c -- env-flag-driven dump of emitted hx codegen source.
 *
 * If the environment variable named by env_var_name is set, it names a
 * DIRECTORY, and the source string is written verbatim to <dir>/<name>.
 * One run emits a kernel per type and device, so a single file path could
 * not hold them.  Unset or empty: no-op.  Every codegen site, OpenCL and
 * Metal, dumps through here so the variable has one meaning.
 *
 * Per feedback_external_failures_are_fatal.md: if the env var IS set
 * but the open/write fails, that is a real I/O failure -- we report
 * a negative return so the caller can exit(1) with a full diagnostic.
 * (We do NOT exit(1) here directly because this helper is reusable
 * and the calling context owns the hostname / dev_idx fields needed
 * for the fatal message.)
 *
 * $Revision: 1.2 $
 * $Log: hx_dump.c,v $
 * Revision 1.2  2026/11/14 22:14:35  dlr
 * The GPU rules engine now covers every GPU type on OpenCL and Metal, including the crypt types, MD5MD5SALT and the MD5PASS family, with masks and iteration. The S rule is removed and the v rule runs on the GPU. Each GPU job covers exactly its own slice of the mask, and Metal mask jobs are chunked below 2^32. Unused Kernel-A test code and profiling knobs are removed. GPU dispatch and replay failures are fatal, and hash counts are accounted on every GPU path.
 *
 * Revision 1.1  2026/05/21 21:31:32  dlr
 * sub-phase 2a.1 initial: MDXFIND_HX_CODEGEN_DUMP path writes emitted source verbatim; no-op when env var unset; I-O failure returns negative for caller to fatal
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "hx_walker.h"

int hx_dump_source(const char *src, const char *env_var_name,
                   const char *name, char *path_out, size_t path_cap)
{
    if (path_out && path_cap) snprintf(path_out, path_cap, "(none)");
    if (!src || !env_var_name || !name) return -1;

    const char *dir = getenv(env_var_name);
    if (!dir || !*dir) return 0;   /* unset = no-op */

    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) {
        fprintf(stderr, "hx codegen: %s=%s: path too long for %s\n",
                env_var_name, dir, name);
        return -1;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "hx codegen: %s: fopen %s failed (errno=%d %s)\n",
                env_var_name, path, errno, strerror(errno));
        return -1;
    }
    size_t n = strlen(src);
    size_t w = fwrite(src, 1, n, fp);
    int closerc = fclose(fp);
    if (w != n || closerc != 0) {
        fprintf(stderr, "hx codegen: %s: write %s failed (%zu/%zu bytes, "
                "fclose=%d errno=%d %s)\n", env_var_name, path, w, n,
                closerc, errno, strerror(errno));
        return -1;
    }
    if (path_out && path_cap) snprintf(path_out, path_cap, "%s", path);
    fprintf(stderr, "hx codegen: dumped %zu bytes of emitted source to %s\n",
            n, path);
    return 1;
}
