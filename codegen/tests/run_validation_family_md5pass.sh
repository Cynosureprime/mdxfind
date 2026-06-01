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
# $Revision: 1.12 $
# $Log: run_validation_family_md5pass.sh,v $
# Revision 1.12  2026/05/28 06:12:50  dlr
# sub-phase 5c.2.3 widen FAMILY_JOBS 29 to 30 add 123 MD5MD5PASS e123 numeric-sorted after 122 before 125 the FIRST multi-emit family member closes MAKE_MD5PASS family at 30 of 30 GPU-eligible; e123 emits TWO digests per password canonical plus colon variant harness plants both requires GPU to emit both cracks G1b dual-hash canary; update JOB_ENUM validation case parallel and header comment
#
# Revision 1.11  2026/05/28 04:51:16  dlr
# 5b.4b.4 widen FAMILY_JOBS 28 to 29 add 125 GOSTMD5PASS e125 numeric sorted after 122 before 127 the FINAL GPU-eligible MAKE_MD5PASS member update JOB_ENUM case and comment Tier 4 COMPLETE 29 of 30 only e123 multi-emit remains CPU-only
#
# Revision 1.10  2026/05/28 04:32:31  dlr
# sub-phase 5b4a4 widen FAMILY_JOBS from 26 to 28 entries add 175 177 the 2 Snefru family members SNE128MD5PASS e175 SNE256MD5PASS e177 numeric sorted after 173 WRLMD5PASS update JOB_ENUM validation case parallel and comment Tier 4 Snefru pair ships 28 of 30 MAKE_MD5PASS coverage gost e125 follows in 5b4b
#
# Revision 1.9  2026/05/28 03:53:05  dlr
# sub-phase 5b3c4 widen FAMILY_JOBS from 21 to 26 entries add 131 137 143 149 155 the 5 5-pass HAVAL family members HAV128_5 HAV160_5 HAV192_5 HAV224_5 HAV256_5 numeric sorted interleaved with 3-pass and 4-pass entries update JOB_ENUM validation case parallel and header comment Tier 3 COMPLETE all 15 HAVAL variants e127 through e155 GPU-eligible 26 of 30 MAKE_MD5PASS family coverage 86.7 percent
#
# Revision 1.8  2026/05/28 03:20:11  dlr
# sub-phase 5b3b4 widen FAMILY_JOBS from 16 to 21 entries add 129 135 141 147 153 the 5 4-pass HAVAL family members HAV128_4 HAV160_4 HAV192_4 HAV224_4 HAV256_4 numeric sorted interleaved with 3-pass entries update JOB_ENUM validation case parallel and header comment Tier 3 sub-phase 5b3b ships 4-pass HAVAL e129 e135 e141 e147 e153
#
# Revision 1.7  2026/05/28 02:25:16  dlr
# sub-phase 5b3a4 widen FAMILY_JOBS list from 11 to 16 entries add 127 133 139 145 151 the 5 3-pass HAVAL family members HAV128 HAV160_3 HAV192_3 HAV224_3 HAV256 numeric sorted between 122 MD4MD5PASS and 157 RMD128MD5PASS update header comment and JOB_ENUM validation case parallel Tier 3 sub-phase 5b3a ships 3-pass HAVAL e127 e133 e139 e145 e151
#
# Revision 1.6  2026/05/27 23:14:17  dlr
# sub-phase 5b2b4 widen FAMILY_JOBS list from 10 to 11 entries add 171 JOB_TIGERMD5PASS e171 for Tier 2 Tiger ship 5b2b numeric-sorted between 169 JOB_SHA512MD5PASS and 173 JOB_WRLMD5PASS update header comment and JOB_ENUM validation case parallel both Tier 2 ships complete
#
# Revision 1.5  2026/05/27 22:31:10  dlr
# sub-phase 5b2a4 widen FAMILY_JOBS list from 9 to 10 entries add 173 JOB_WRLMD5PASS e173 for Tier 2 Whirlpool ship 5b.2a numeric-sorted after 169 JOB_SHA512MD5PASS update header comment and JOB_ENUM validation case parallel
#
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
#   171  TIGERMD5PASS  (Tier 2 -- shipped 5b.2b 2026-05-27)
#   173  WRLMD5PASS    (Tier 2 -- shipped 5b.2a 2026-05-27)
# "all" loops through all 11 sequentially.
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

# Family JOB enums (numeric sort).
# e120 (MD2MD5PASS) added in sub-phase 5b.1a (2026-05-27) Tier 1.
# e157 (RMD128MD5PASS) added in sub-phase 5b.1b (2026-05-27) Tier 1.
# e171 (TIGERMD5PASS) added in sub-phase 5b.2b (2026-05-27) Tier 2.
# e173 (WRLMD5PASS) added in sub-phase 5b.2a (2026-05-27) Tier 2.
# e127/e133/e139/e145/e151 (3-pass HAVAL) added in sub-phase 5b.3a
# (2026-05-27) Tier 3: HAV128/160/192/224/256 _3.
# e129/e135/e141/e147/e153 (4-pass HAVAL) added in sub-phase 5b.3b
# (2026-05-27) Tier 3: HAV128/160/192/224/256 _4.
# e131/e137/e143/e149/e155 (5-pass HAVAL) added in sub-phase 5b.3c
# (2026-05-27) Tier 3: HAV128/160/192/224/256 _5. Completes the 15-variant
# HAVAL family; Tier 3 closes at 26/30 MAKE_MD5PASS members GPU-eligible.
# e175/e177 (Snefru-128/256) added in sub-phase 5b.4a (2026-05-27) Tier 4.
# e125 (GOSTMD5PASS, GOST R 34.11-94) added in sub-phase 5b.4b (2026-05-27)
# Tier 4. e123 (MD5MD5PASS) added in sub-phase 5c.2 (2026-05-27) -- the
# FIRST multi-emit member; closes the MAKE_MD5PASS family at 30/30
# GPU-eligible. e123 is multi-emit: each password emits TWO digests
# (canonical + colon variant); the harness plants both and requires the
# GPU to emit both cracks (the G1b dual-hash canary).
FAMILY_JOBS="120 122 123 125 127 129 131 133 135 137 139 141 143 145 147 149 151 153 155 157 159 161 163 165 167 169 171 173 175 177"

# Validate JOB_ENUM is either "all" or one of the 30 family members.
case "$JOB_ENUM" in
    all|120|122|123|125|127|129|131|133|135|137|139|141|143|145|147|149|151|153|155|157|159|161|163|165|167|169|171|173|175|177) ;;
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
