/*
 * hx_plan_scan.c -- build the generic walker's plan for every catalog row and
 * report which are eligible, and why the others are not.
 *
 *   hx_plan_scan            one summary line per reason, with counts
 *   hx_plan_scan -v         also every row: eN, verdict, plan or reason
 *
 * Each row's precompiled catalog program is used, with its call-name
 * sidecar, exactly as the walker sees it. Host-side conditions (compact table, TypeDigest, the drift ledger,
 * hand routing) are not judged here.
 *
 * $Revision: 1.1 $
 * $Log: hx_plan_scan.c,v $
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

#define MAXR 256

int main(int argc, char **argv)
{
    int verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    char reasons[MAXR][160];
    int counts[MAXR], nr = 0, i, ok = 0, rows = 0;
    static struct hx_plan plan;
    char buf[1024];

    for (i = 0; i < hx_specs_count; i++) {
        const struct hx_spec_entry *e = &hx_specs_data[i];
        const char *why;
        int j;
        if (e->is_outlier || !e->expression) continue;
        rows++;
        hx_plan_build(e->program, e->call_names, 256, &plan);
        if (plan.eligible) {
            ok++;
            why = "ELIGIBLE";
            hx_plan_render(&plan, plan.root, buf, sizeof buf);
        } else {
            why = plan.reason;
            snprintf(buf, sizeof buf, "%s", e->expression);
        }
        if (verbose)
            printf("e%d\t%s\t%s\t%s\n", e->job_enum, e->name, why, buf);
        for (j = 0; j < nr; j++) if (strcmp(reasons[j], why) == 0) break;
        if (j == nr && nr < MAXR) { snprintf(reasons[nr], 160, "%s", why); counts[nr++] = 0; }
        if (j < MAXR) counts[j]++;
    }
    for (i = 0; i < nr; i++) printf("%5d  %s\n", counts[i], reasons[i]);
    printf("%d of %d rows eligible by expression\n", ok, rows);
    return 0;
}
