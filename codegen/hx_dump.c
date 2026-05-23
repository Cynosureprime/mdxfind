/*
 * hx_dump.c -- env-flag-driven dump of emitted hx codegen source.
 *
 * If the environment variable named by env_var_name is set, write the
 * passed source string verbatim to the path it names. If the env var
 * is unset, the call is a no-op and returns success. Used by the
 * sub-phase 2a.1 harness in mdxfind.c to surface emitted source for
 * inspection without complicating the walker proper.
 *
 * Per feedback_external_failures_are_fatal.md: if the env var IS set
 * but the open/write fails, that is a real I/O failure -- we report
 * a negative return so the caller can exit(1) with a full diagnostic.
 * (We do NOT exit(1) here directly because this helper is reusable
 * and the calling context owns the hostname / dev_idx fields needed
 * for the fatal message.)
 *
 * $Revision: 1.1 $
 * $Log: hx_dump.c,v $
 * Revision 1.1  2026/05/21 21:31:32  dlr
 * sub-phase 2a.1 initial: MDXFIND_HX_CODEGEN_DUMP path writes emitted source verbatim; no-op when env var unset; I-O failure returns negative for caller to fatal
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hx_walker.h"

int hx_dump_source(const char *src, const char *env_var_name)
{
    if (!src || !env_var_name) return -1;

    const char *path = getenv(env_var_name);
    if (!path || !*path) return 0;   /* unset = no-op, success */

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr,
                "hx codegen: %s=%s: fopen failed\n",
                env_var_name, path);
        return -1;
    }
    size_t n = strlen(src);
    size_t w = fwrite(src, 1, n, fp);
    int closerc = fclose(fp);
    if (w != n) {
        fprintf(stderr,
                "hx codegen: %s=%s: short write %zu/%zu bytes\n",
                env_var_name, path, w, n);
        return -1;
    }
    if (closerc != 0) {
        fprintf(stderr,
                "hx codegen: %s=%s: fclose failed\n",
                env_var_name, path);
        return -1;
    }
    fprintf(stderr,
            "hx codegen: dumped %zu bytes of emitted source to %s\n",
            n, path);
    return 0;
}
