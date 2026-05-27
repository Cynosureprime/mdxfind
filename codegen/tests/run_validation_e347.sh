#!/bin/sh
# run_validation_e347.sh -- driver for the 2a.5 e347 byte-exact harness.
#
# Sub-phase 2a.5 (2026-05-21). Ships the fixture to the target host then
# invokes mdxfind with MDXFIND_HX_CODEGEN_VALIDATE=1.
#
# $Revision: 1.2 $
# $Log: run_validation_e347.sh,v $
# Revision 1.2  2026/05/23 18:55:10  dlr
# sub-phase 5a7 cleanup fix latent harness invocation bug mdxfind dash V exits at version print before build compact table runs so the e347 harness body was never actually exercised by this script switch to -m e347 -f empty -f empty wordlist pattern matching run validation family md5pass dot sh which reaches build compact table and fires the harness body adds dummy hash and wordlist touch on remote noted by 5a6 agent
#
# Revision 1.1  2026/05/22 03:35:05  dlr
# initial check-in sub-phase 2a.5 e347 validation harness driver script
#
#
# Usage:
#   ./run_validation_e347.sh <fixture_file> [host]
#
# Default host: fpga.local (Pascal GTX 1080).
#
# Exit code 0 = PASS (byte-exact); 1 = FAIL (diffs reported on stderr).

set -e

FIXTURE=${1:-/Users/dlr/src/mdfind/codegen/tests/fixture_smoke.txt}
HOST=${2:-fpga.local}

if [ ! -f "$FIXTURE" ]; then
    echo "error: fixture '$FIXTURE' not found" >&2
    exit 2
fi

REMOTE_FIXTURE=/tmp/hx_e347_fixture.txt
REMOTE_DUMP=/tmp/hx_e347_validate.cl

echo ">> shipping fixture $FIXTURE -> $HOST:$REMOTE_FIXTURE" >&2
rsync -av "$FIXTURE" "$HOST:$REMOTE_FIXTURE"

echo ">> running mdxfind validation harness on $HOST" >&2
# Pre-create empty dummy hash + wordlist on the remote so mdxfind reaches
# build_compact_table where the validation harness fires. The previous
# invocation used ./mdxfind -V which exits at the version-print site
# (mdxfind.c around line 46256) BEFORE the harness body runs, so the
# fixture was never actually exercised. Mirrors run_validation_family_md5pass.sh.
ssh -o StrictHostKeyChecking=no "$HOST" \
    "touch /tmp/hx_dummy_hashes.txt /tmp/hx_dummy_wordlist.txt"

ssh "$HOST" "cd ~/src/mdfind && \
    MDXFIND_HX_CODEGEN=1 \
    MDXFIND_HX_CODEGEN_JOB=347 \
    MDXFIND_HX_CODEGEN_VALIDATE=1 \
    MDXFIND_HX_CODEGEN_FIXTURE=$REMOTE_FIXTURE \
    MDXFIND_HX_CODEGEN_DUMP=$REMOTE_DUMP \
    ./mdxfind -m e347 -f /tmp/hx_dummy_hashes.txt \
              /tmp/hx_dummy_wordlist.txt 2>&1 | tail -80"
