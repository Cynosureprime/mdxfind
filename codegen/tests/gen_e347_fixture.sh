#!/bin/sh
# gen_e347_fixture.sh -- generate a deterministic fixture file for the
#                        2a.5+2a.6 e347 byte-exact validation harness.
#
# Sub-phase 2a.5 (2026-05-21). Output format (line-oriented):
#
#   # header comments
#   PASS:<plaintext>
#   ...
#   SALT:<text>
#   ...
#
# Cartesian product: n_pass x n_salt pairs. Each PASS/SALT line value
# is a deterministic pseudo-random ASCII string derived from a fixed-seed
# awk RNG, so the fixture is reproducible byte-for-byte across runs and
# across machines.
#
# Sub-phase 2a.6 (2026-05-22) extension: optional length-range args so
# edge fixtures can exercise the slen 1 single-block path AND the slen
# >=56 multi-block path that the 2a.5 padding bug surfaced through
# inference but no shipped fixture actually reached.
#
# $Revision: 1.3 $
# $Log: gen_e347_fixture.sh,v $
# Revision 1.3  2026/05/22 04:12:26  dlr
# sub-phase 2a6 followup add rejection retry uniqueness enforcement to gen_e347_fixture sh prior version dropped the index suffix when length range was too tight which allowed deterministic awk RNG to emit duplicate single char passwords on the edge_empty 1 1 1 4 path with seed 42 producing duplicate V and I that the validation harness then reported as 8 missing pairs even though the GPU dedup behavior was correct emit_unique helper now rejects and retries up to alphabet times 10 plus 100 attempts per item and dies loud if uniqueness is mathematically impossible for the requested range edge_empty fixture regenerated with 16 unique single char passwords no longer fails validation
#
# Revision 1.2  2026/05/22 04:00:42  dlr
# sub-phase 2a6 extend gen_e347_fixture sh to accept optional length range args for pass and salt so edge fixtures can exercise the slen 1 single block path and the slen 56 through 128 multi block path that 2a5 only inferred but never reached defaults preserve 4 to 20 char pass and 12 to 25 char salt original behavior underscore numeric suffix dropped when min_len equals max_len equals 1 to avoid blowing past max_len on very short pass paths
#
# Revision 1.2  2026/05/22 04:30:00  dlr
# sub-phase 2a.6 extend gen_e347_fixture.sh to accept optional length-range args for pass and salt so edge fixtures can exercise the slen 1 single block path and the slen 56 through 128 multi block path that 2a.5 only inferred but never reached with a fixture; defaults preserve 4 to 20 char pass and 12 to 25 char salt original behavior; underscore-numeric suffix dropped from output strings when min_len equals max_len equals 1 to avoid blowing past max_len on very short pass paths
#
# Revision 1.1  2026/05/22 03:35:05  dlr
# initial check-in sub-phase 2a.5 deterministic e347 fixture generator
#
#
# Usage:
#   ./gen_e347_fixture.sh <n_pass> <n_salt> <output_file>
#       [pass_minlen] [pass_maxlen] [salt_minlen] [salt_maxlen]
#
# Defaults: pass_minlen=4 pass_maxlen=20 salt_minlen=12 salt_maxlen=25
#
# Examples:
#   ./gen_e347_fixture.sh 64 16 fixture_medium.txt
#   ./gen_e347_fixture.sh 16 8 fixture_edge_maxlen.txt 1 20 56 128

set -e

if [ $# -lt 3 ] || [ $# -gt 7 ]; then
    echo "usage: $0 <n_pass> <n_salt> <output_file> [pass_min] [pass_max] [salt_min] [salt_max]" >&2
    exit 1
fi

N_PASS=$1
N_SALT=$2
OUT=$3
PASS_MIN=${4:-4}
PASS_MAX=${5:-20}
SALT_MIN=${6:-12}
SALT_MAX=${7:-25}

# awk emits both passwords and salts. Single deterministic seed (42)
# guarantees byte-identical output across runs / hosts. Each generated
# string is alphanumeric; index suffix ensures uniqueness when there's
# room, otherwise the loop rejects-and-retries until a duplicate-free
# string is produced (and dies hard if uniqueness is mathematically
# impossible for the requested range, e.g. n_pass=100 with pmax=1).
awk -v np="$N_PASS" -v ns="$N_SALT" \
    -v pmin="$PASS_MIN" -v pmax="$PASS_MAX" \
    -v smin="$SALT_MIN" -v smax="$SALT_MAX" \
    'function gen_random_str(min_l, max_l, alphabet, alen,    L, s, j, c, span) {
        span = max_l - min_l + 1
        if (span < 1) span = 1
        L = min_l + int(rand() * span)
        if (L < min_l) L = min_l
        if (L > max_l) L = max_l
        s = ""
        for (j = 0; j < L; j++) {
            c = substr(alphabet, 1 + int(rand() * alen), 1)
            s = s c
        }
        return s
    }
    function emit_unique(side_tag, suffix_template, n_items, min_l, max_l,
                         alphabet, alen, seen, MAX_RETRY,    i, s, key, tries) {
        for (i = 0; i < n_items; i++) {
            tries = 0
            while (1) {
                s = gen_random_str(min_l, max_l, alphabet, alen)
                if (suffix_template == "_%d") {
                    key = s "_" i
                } else if (suffix_template == "_s%d") {
                    key = s "_s" i
                } else {
                    key = s
                }
                if (!(key in seen)) break
                tries++
                if (tries > MAX_RETRY) {
                    printf("FATAL: gen_e347_fixture.sh: cannot generate "    \
                           "unique %s after %d retries at index %d "        \
                           "(min=%d max=%d alphabet=%d) -- requested count "\
                           "exceeds the unique combinatorial space\n",      \
                           side_tag, MAX_RETRY, i, min_l, max_l, alen)      \
                          > "/dev/stderr"
                    exit 2
                }
            }
            seen[key] = 1
            printf("%s:%s\n", side_tag, key)
        }
    }
    BEGIN {
    srand(42)
    printf("# e347 validation fixture, generated by gen_e347_fixture.sh\n")
    printf("# n_pass=%d n_salt=%d n_pairs=%d\n", np, ns, np * ns)
    printf("# pass_min=%d pass_max=%d salt_min=%d salt_max=%d\n",
           pmin, pmax, smin, smax)
    printf("# seed=42 deterministic; do not hand-edit\n")
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    alen = length(alphabet)

    # Decide once whether to emit the _N index suffix per side.
    # When pmax is too small to accommodate "_<int>" we drop the suffix
    # and rely on the rejection-retry loop in emit_unique() for uniqueness.
    pass_suffix = (pmax >= 6 && pmax - pmin >= 2) ? "_%d" : ""
    salt_suffix = (smax >= 8 && smax - smin >= 2) ? "_s%d" : ""

    # Rejection-retry cap: ~10x the alphabet should be plenty when the
    # combinatorial space exists; if exhausted the gen fails loud rather
    # than silently emitting duplicates that the validation harness will
    # then surface as missing-pair "FAIL".
    MAX_RETRY = 10 * alen + 100

    emit_unique("PASS", pass_suffix, np, pmin, pmax,
                alphabet, alen, pass_seen, MAX_RETRY)
    emit_unique("SALT", salt_suffix, ns, smin, smax,
                alphabet, alen, salt_seen, MAX_RETRY)
}' > "$OUT"

echo "wrote fixture $OUT (n_pass=$N_PASS n_salt=$N_SALT n_pairs=$((N_PASS * N_SALT)) pass=$PASS_MIN..$PASS_MAX salt=$SALT_MIN..$SALT_MAX)" >&2
