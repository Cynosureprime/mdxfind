#!/bin/sh
# run_validation_family_md5pass.sh -- driver for the 5a.6 MAKE_MD5PASS
#                                     family byte-exact validation harness.
#
# Sub-phase 5a.6 (2026-05-23). Ships the (unsalted) fixture to the target
# host then invokes mdxfind with MDXFIND_HX_CODEGEN_VALIDATE=1 for one or
# all 9 GPU-eligible family JOB enums (e120/e122/e157/e159/e161/e163/e165/
# e167/e169). e120 (MD2MD5PASS) added in sub-phase 5b.1a (2026-05-27)
# Tier 1; e157 (RMD128MD5PASS) added in sub-phase 5b.1b (2026-05-27)
# Tier 1.
#
# $Revision: 1.4 $
# $Log: run_validation_family_md5pass.sh,v $
# Revision 1.4  2026/05/27 17:51:39  dlr
# sub-phase 5b1b4 widen FAMILY_JOBS from 8 to 9 entries 120 122 157 159 161 163 165 167 169 numeric sorted RMD128MD5PASS e157 added Tier 1 shipped 5b1b 2026-05-27 case validation widened comment header updated to 9 GPU-eligible family JOB enums per-eN comments include 157 RMD128MD5PASS line
#
# Revision 1.3  2026/05/27 17:06:12  dlr
# sub-phase 5b1a4 widen FAMILY_JOBS from 7 to 8 entries 120 122 159 161 163 165 167 169 numeric sorted MD2MD5PASS e120 added Tier 1 shipped 5b1a 2026-05-27 case validation widened comment header updated to 8 GPU-eligible family JOB enums
#
# Revision 1.2  2026/05/23 18:41:18  dlr
# fix stale draft comment line in eNNN enum list header e122 is MD4MD5PASS not RMD160
#
# Revision 1.1  2026/05/23 18:41:00  dlr
# sub-phase 5a6 family validation runner script for MAKE_MD5PASS family driver ships fixture invokes mdxfind with MDXFIND_HX_CODEGEN_VALIDATE for one or all 7 GPU-eligible family JOBs default host fpga local for OpenCL Pascal use dev3 local for Metal Apple M aggregates per-cell PASS FAIL plus wall time into clean summary table
#
#
# Usage:
#   ./run_validation_family_md5pass.sh <fixture_file> <job_enum|all> [host]
#
# job_enum is the eNNN internal index without the "e" prefix:
#   120  MD2MD5PASS    (Tier 1 -- shipped 5b.1a 2026-05-27)
#   122  MD4MD5PASS
#   157  RMD128MD5PASS (Tier 1 -- shipped 5b.1b 2026-05-27)
#   159  RMD160MD5PASS
#   161  SHA1MD5PASS
#   163  SHA224MD5PASS
#   165  SHA256MD5PASS
#   167  SHA384MD5PASS
#   169  SHA512MD5PASS
# "all" loops through all 9 sequentially.
#
# Default host: fpga.local (Pascal GTX 1080 OpenCL). For Metal use
# dev3.local (Apple M2 Max).
#
# Exit code 0 = all selected cells PASS; non-zero = at least one FAIL.

set -e

FIXTURE=${1:-/Users/dlr/src/mdfind/codegen/tests/family_md5pass/family_smoke.txt}
JOB_ENUM=${2:-all}
HOST=${3:-fpga.local}

if [ ! -f "$FIXTURE" ]; then
    echo "error: fixture '$FIXTURE' not found" >&2
    exit 2
fi

# Family JOB enums (e120/e122/e157/e159/e161/e163/e165/e167/e169 -- numeric sort).
# e120 (MD2MD5PASS) added in sub-phase 5b.1a (2026-05-27) Tier 1.
# e157 (RMD128MD5PASS) added in sub-phase 5b.1b (2026-05-27) Tier 1.
FAMILY_JOBS="120 122 157 159 161 163 165 167 169"

# Validate JOB_ENUM is either "all" or one of the 9 family members.
case "$JOB_ENUM" in
    all|120|122|157|159|161|163|165|167|169) ;;
    *)
        echo "error: job_enum '$JOB_ENUM' not in family set: all $FAMILY_JOBS" >&2
        exit 2
        ;;
esac

# Per-host fixture path on remote.
REMOTE_FIXTURE=/tmp/hx_family_md5pass_fixture.txt

echo ">> shipping fixture $FIXTURE -> $HOST:$REMOTE_FIXTURE" >&2
rsync -av "$FIXTURE" "$HOST:$REMOTE_FIXTURE" >/dev/null

# Detect dummy hash + wordlist files (needed by mdxfind so it doesn't
# bail before reaching build_compact_table where the harness fires).
ssh -o StrictHostKeyChecking=no "$HOST" \
    "touch /tmp/hx_dummy_hashes.txt /tmp/hx_dummy_wordlist.txt"

# Determine which JOBs to run.
if [ "$JOB_ENUM" = "all" ]; then
    JOBS_TO_RUN="$FAMILY_JOBS"
else
    JOBS_TO_RUN="$JOB_ENUM"
fi

# Per-cell driver. Captures the single RESULT line, returns PASS/FAIL
# plus wall time. Format of RESULT line:
#   hx codegen 5a.2 (<backend>) e<N> RESULT on <host>: PASS  n_pass=<N> \
#       vn_hits=<N> matched=<N> missing=<N> extras=<N> digest_mismatches=<N>
run_one_cell() {
    job=$1
    label="e${job}"
    fixture_name=$(basename "$FIXTURE")
    start=$(date +%s)
    # Capture entire harness output for diagnosis on FAIL; tail to result.
    out=$(ssh -o StrictHostKeyChecking=no "$HOST" \
        "cd ~/src/mdfind && \
        MDXFIND_HX_CODEGEN=1 \
        MDXFIND_HX_CODEGEN_JOB=$job \
        MDXFIND_HX_CODEGEN_VALIDATE=1 \
        MDXFIND_HX_CODEGEN_FIXTURE=$REMOTE_FIXTURE \
        ./mdxfind -m e$job -f /tmp/hx_dummy_hashes.txt \
                  /tmp/hx_dummy_wordlist.txt 2>&1" || true)
    end=$(date +%s)
    elapsed=$((end - start))
    result_line=$(echo "$out" | grep "RESULT on" | tail -1)
    if echo "$result_line" | grep -q " PASS "; then
        status=PASS
    else
        status=FAIL
    fi
    # Extract count tuple for matrix display.
    counts=$(echo "$result_line" | sed -E 's/.*RESULT on [^:]*: [A-Z]+ +//; s/[[:space:]]+/ /g')
    printf "  %-10s %-6s %4ds  %s\n" "$label" "$status" "$elapsed" "$counts"
    if [ "$status" = "FAIL" ]; then
        echo "    --- full harness output (FAIL) ---" >&2
        echo "$out" | tail -40 >&2
        echo "    ---" >&2
        return 1
    fi
    return 0
}

echo "" >&2
echo "============================================================" >&2
echo "5a.6 family_md5pass validation: $(basename "$FIXTURE") on $HOST" >&2
echo "============================================================" >&2
n_pass=0
n_fail=0
fail_list=""
for j in $JOBS_TO_RUN; do
    if run_one_cell "$j"; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
        fail_list="$fail_list e$j"
    fi
done
echo "" >&2
echo "----- summary on $HOST ($(basename "$FIXTURE")) -----" >&2
echo "PASS: $n_pass    FAIL: $n_fail" >&2
if [ $n_fail -gt 0 ]; then
    echo "FAILED cells:$fail_list" >&2
    exit 1
fi
exit 0
