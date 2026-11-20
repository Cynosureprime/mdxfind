/*
 * hx_gen_emit.c -- write the template core of every catalog type the
 * generator handles into one C file, through the C dialect, with a table of
 * their entry points, for hx_gen_check to compile and run.
 *
 *   hx_gen_emit out.c          the cores, and a skip summary on stderr
 *   hx_gen_emit -v out.c       also every skipped type and why
 *   hx_gen_emit -b opencl|metal -j N out
 *                              one type's core for that backend, dialect
 *                              first: the text a GPU program is built from
 *
 * $Revision: 1.3 $
 * $Log: hx_gen_emit.c,v $
 * Revision 1.3  2026/11/20 05:22:13  dlr
 * Generic hx walker stage 1c: user-defined types run on the GPU through the generated walker. A user type is served when it takes no salt or user, declares no stored form, and its result is the hex of its final digest; rounds past x01 re-run the whole program on the previous round's hex text, keeping its case, as the userdef CPU arm does. The selected user type's status line says whether it runs on the GPU, or why not, in place of the former not-yet-enabled line. Verified: the C mirror agrees for 529 of 529 cores, including 10 user cores whose later rounds are checked against the hx VM re-running the program, and a planted case defect is caught in exactly the 8 case-sensitive ones; the 519 catalog cores are byte-identical; on dev3 (M2 Max, Metal) and gp1 (Tahiti, OpenCL) 10 user types match -G none at x01 and x02 and deeper with rules, a -n mask and a mixed run with a catalog type, except one type whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
 *
 * Revision 1.2  2026/11/20 02:52:03  dlr
 * Generic hx walker stage 2: the plan and emitter take a salt or user channel, HMAC, round-1-only types and peppers. A salt or user variable reads the 6-argument salted template_finalize; HMAC is generated over any existing body following mhash: block from pblock with 0 as 64, MD2 16, HAVAL and SHA-384/512 128, a key longer than the block replaced by its digest, the inner digest entering the outer hash at the body width, Tiger in mhash byte order; HX_NO_ITERATE marks a type compared at round 1 only. A pepper rides the salt channel: the host packs salt . pepper . the pepper length in 2 bytes, and the core splits it. Verified: the C mirror agrees for 519 of 519 cores over 67548 finalize and 64194 iterate vectors, including the 14 pepper cores over 8 salts by 6 peppers up to 300 bytes; planted defects in the split and in the length high byte are caught in exactly those 14; the 505 earlier cores are byte-identical.
 *
 * Revision 1.1  2026/11/18 04:08:38  dlr
 * Initial revision.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hx_vm.h"
#include "codegen/hx_spec_entry.h"
#include "codegen/hx_plan.h"
#include "codegen/hx_gen.h"

#define MAXR 128

/* User-defined types in the shape stage 1c serves: unsalted, the result the
 * hex of its final digest, iterated by re-running the whole program on the
 * previous round's text. Emitted as jobs 1100 + index; the checker reads the
 * expressions back from the table this writes. */
static const char *const user_expr[] = {
    "md5(pass)", "sha1(md5(pass))", "md5(pass . pass)", "upper(md5(pass))",
    "sha256(md5(pass) . sha1(pass))", "md5(rev(pass))", "sha1(pass . \":\" . pass)",
    "md4(utf16le(pass))", "sha512(pass)", "md5(sha256(upper(pass)))", NULL
};
#define USER_BASE 1100

int main(int argc, char **argv)
{
    int verbose = 0, i, n = 0, nr = 0, counts[MAXR];
    char reasons[MAXR][160], why[160];
    static struct hx_plan plan;
    static int jobs[4096];
    static int iter[4096];
    static int chan[4096];
    FILE *f;
    int backend = HXG_C, only = -1;
    for (;;) {
        if (argc > 1 && strcmp(argv[1], "-v") == 0) { verbose = 1; argv++; argc--; continue; }
        if (argc > 2 && strcmp(argv[1], "-b") == 0) {
            if (strcmp(argv[2], "opencl") == 0) backend = HXG_OPENCL;
            else if (strcmp(argv[2], "metal") == 0) backend = HXG_METAL;
            else { fprintf(stderr, "hx_gen_emit: unknown backend %s\n", argv[2]); return 2; }
            argv += 2; argc -= 2; continue;
        }
        if (argc > 2 && strcmp(argv[1], "-j") == 0) { only = atoi(argv[2]); argv += 2; argc -= 2; continue; }
        break;
    }
    if (argc != 2 || (backend != HXG_C) != (only >= 0)) {
        fprintf(stderr, "usage: hx_gen_emit [-v] out.c | hx_gen_emit -b opencl|metal -j N out\n");
        return 2;
    }
    if (!(f = fopen(argv[1], "w"))) { perror(argv[1]); return 1; }
    fputs(hx_gen_dialect(backend), f);
    for (i = 0; user_expr[i]; i++) {
        hx_program *up;
        char *core;
        if (only >= 0 && only != USER_BASE + i) continue;
        up = hx_compile_expr(user_expr[i], NULL);
        if (!up) { fprintf(stderr, "hx_gen_emit: user expression %s does not compile\n", user_expr[i]); return 1; }
        hx_plan_build(up, NULL, 256, &plan);
        core = hx_gen_core_whole(&plan, why, sizeof why);
        if (!core) { fprintf(stderr, "hx_gen_emit: user expression %s: %s\n", user_expr[i], why); return 1; }
        if (only >= 0) {
            fprintf(f, "\n/* user %d */\n%s", only, core);
            fclose(f);
            return 0;
        }
        fprintf(f, "\n/* user %d: %s */\n#define HXID %d\n%s#undef HXID\n",
                USER_BASE + i, user_expr[i], USER_BASE + i, core);
        free(core);
        jobs[n] = USER_BASE + i;
        iter[n] = 2;            /* whole-program iterate */
        chan[n] = 0;
        n++;
    }
    for (i = 0; i < hx_specs_count; i++) {
        const struct hx_spec_entry *e = &hx_specs_data[i];
        char *core;
        int j;
        if (e->is_outlier || !e->expression) continue;
        if (only >= 0 && e->job_enum != only) continue;
        hx_plan_build(e->program, e->call_names, 256, &plan);
        core = hx_gen_core(&plan, 0, why, sizeof why);
        if (core && backend == HXG_C && !hx_gen_c_mirror(&plan)) {
            /* checked on the device against the CPU instead */
            free(core); core = NULL;
            snprintf(why, sizeof why, "a body with no C mirror (stage 1b)");
        }
        if (!core) {
            if (verbose) fprintf(stderr, "e%d\t%s\t%s\n", e->job_enum, e->name, why);
            for (j = 0; j < nr; j++) if (strcmp(reasons[j], why) == 0) break;
            if (j == nr && nr < MAXR) { snprintf(reasons[nr], 160, "%s", why); counts[nr++] = 0; }
            if (j < MAXR) counts[j]++;
            continue;
        }
        if (only >= 0) {
            /* hx expressions hold no comment terminator, but be sure */
            fprintf(f, "\n/* e%d %s */\n%s", e->job_enum, e->name, core);
            free(core);
            fclose(f);
            return 0;
        }
        fprintf(f, "\n/* e%d %s: %s */\n#define HXID %d\n%s#undef HXID\n#undef HX_X01_SILENT\n#undef HX_NO_ITERATE\n",
                e->job_enum, e->name, e->expression, e->job_enum, core);
        free(core);
        jobs[n] = e->job_enum;
        iter[n] = plan.iter_body >= 0;
        chan[n] = plan.pepper ? 3 : plan.channel;   /* 3: salt and pepper */
        n++;
    }
    if (only >= 0) {
        fprintf(stderr, "hx_gen_emit: e%d is not generated%s%s\n", only, nr ? ": " : "", nr ? reasons[nr - 1] : "");
        fclose(f);
        return 1;
    }
    fprintf(f, "\nstruct hx_gen_fn { int job; void (*fin)(template_state *, const uchar *, int);\n"
               "    void (*fin_s)(template_state *, const uchar *, int, const uchar *, uint, uint);\n"
               "    void (*iter)(template_state *); int has_iter; int chan; };\n"
               "const struct hx_gen_fn hx_gen_table[] = {\n");
    for (i = 0; i < n; i++) {
        if (chan[i])
            fprintf(f, "    { %d, 0, hx_fin_%d, hx_iter_%d, %d, %d },\n", jobs[i], jobs[i], jobs[i], iter[i], chan[i]);
        else
            fprintf(f, "    { %d, hx_fin_%d, 0, hx_iter_%d, %d, 0 },\n", jobs[i], jobs[i], jobs[i], iter[i]);
    }
    fprintf(f, "};\nconst int hx_gen_count = %d;\n", n);
    fprintf(f, "const char *const hx_gen_user_expr[] = {");
    for (i = 0; user_expr[i]; i++) {
        const char *c;
        fputs("\n    \"", f);
        for (c = user_expr[i]; *c; c++) { if (*c == '"' || *c == '\\') fputc('\\', f); fputc(*c, f); }
        fputs("\",", f);
    }
    fprintf(f, " 0 };\n");
    fclose(f);
    for (i = 0; i < nr; i++) fprintf(stderr, "%5d  skipped: %s\n", counts[i], reasons[i]);
    fprintf(stderr, "%d cores written to %s\n", n, argv[1]);
    return 0;
}
