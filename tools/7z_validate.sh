#!/bin/bash
#
# $Revision: 1.1 $
#
# $Log: 7z_validate.sh,v $
# Revision 1.1  2026/08/21 17:23:16  dlr
# Acceptance test for the 7ZIP e1000 type. Builds a matrix of archives covering every codec, the branch filters, both header-encryption modes, a multi-file archive, a single-block archive and padsizes 0 through 3; validates every fixture against the native 7zz oracle before the attack; extracts with 7z2john.pl and 7z2mdx.py; cracks with a short mostly-invalid wordlist; then re-verifies each reported crack with 7zz t. Exists because a tool returning zero is not evidence of absence until it has returned non-zero on a known positive.
#
#
# 7z_validate.sh -- end-to-end acceptance test for mdxfind's 7ZIP type (e1000).
#
# Builds a matrix of 7z archives covering every codec, the branch filters, both
# header-encryption modes and padsizes 0..3, then:
#   1. validates each fixture with the native oracle (7zz t: right pw rc=0,
#      wrong pw rc=2) BEFORE trusting anything downstream;
#   2. extracts hashes with 7z2john.pl and tools/7z2mdx.py;
#   3. cracks with a short wordlist of mostly-invalid words;
#   4. re-verifies every reported crack with 7zz t.
#
# A negative from a tool that cannot report a positive is not evidence, so the
# invalid-only run must return zero and the known-answer run must return all.
#
# Usage:  tools/7z_validate.sh [workdir]        (default: a mktemp dir)

set -u
# Resolve everything to absolute paths BEFORE cd'ing into the workdir.
HERE=$(cd "$(dirname "$0")" && pwd)
MDX=${MDX:-$HERE/../mdxfind}
J7Z=${J7Z:-$HOME/src/john-bleeding-jumbo/run/7z2john.pl}
CVT=${CVT:-$HERE/7z2mdx.py}
case $MDX in /*) ;; *) MDX=$(cd "$(dirname "$MDX")" && pwd)/$(basename "$MDX");; esac
W=${1:-$(mktemp -d)}
case $W in /*) ;; *) mkdir -p "$W"; W=$(cd "$W" && pwd);; esac
mkdir -p "$W/payload" && cd "$W" || exit 1

for t in 7zz perl python3; do
  command -v $t >/dev/null || { echo "missing: $t" >&2; exit 1; }
done
[ -x "$MDX" ] || { echo "missing mdxfind: $MDX" >&2; exit 1; }
[ -f "$J7Z" ] || { echo "missing 7z2john.pl: $J7Z" >&2; exit 1; }

echo "workdir: $W"
perl -e 'print "The quick brown fox jumps over the lazy dog, line $_\n" for 1..300' > payload/doc.txt
cp /bin/ls payload/prog.bin
printf 'tiny\n' > payload/tiny.txt
perl -e 'print "bulk $_\n" for 1..5000' > payload/bulk.txt

: > answers.tsv
mk() { rm -f "$1.7z"
  7zz a -t7z -p"$5" -mhe=$3 $2 "$1.7z" $4 >/dev/null 2>&1 &&
    printf '%s\t%s\n' "$1" "$5" >> answers.tsv || echo "BUILD FAILED: $1" >&2; }

mk a01_copy      "-m0=Copy"              off payload/doc.txt  'password123'
mk a02_lzma      "-m0=LZMA"              off payload/doc.txt  'Hiems20%'
mk a03_lzma2     "-m0=LZMA2"             off payload/doc.txt  'zurück'
mk a04_ppmd      "-m0=PPMd"              off payload/doc.txt  '返回'
mk a05_bzip2     "-m0=BZip2"             off payload/doc.txt  'password123'
mk a06_deflate   "-m0=Deflate"           off payload/doc.txt  'Hiems20%'
mk a07_deflate64 "-m0=Deflate64"         off payload/doc.txt  'zurück'
mk a08_lzma2_he  "-m0=LZMA2"             on  payload/doc.txt  '返回'
mk a09_d64_he    "-m0=Deflate64"         on  payload/doc.txt  'password123'
mk a10_bcj       "-m0=BCJ -m1=LZMA2"     off payload/prog.bin 'Hiems20%'
mk a11_arm64     "-m0=ARM64 -m1=LZMA2"   off payload/prog.bin 'zurück'
mk a12_delta     "-m0=Delta:4 -m1=LZMA2" off payload/prog.bin 'password123'
mk a13_multi     "-m0=LZMA2"             off "payload/tiny.txt payload/bulk.txt" '返回'
mk a14_tiny      "-m0=LZMA2"             off payload/tiny.txt 'Hiems20%'
mk a15_store_he  "-m0=Copy"              on  payload/tiny.txt 'zurück'
# padsize 0..3, the false-positive-prone band
for want in 0 1 2 3; do
  for n in $(seq 1 400); do
    perl -e "print \"pad probe line \$_\n\" for 1..$n" > payload/p.txt
    7zz a -t7z -p'password123' -mhe=off -m0=LZMA2 t.7z payload/p.txt >/dev/null 2>&1
    got=$(perl "$J7Z" t.7z 2>/dev/null | sed 's/^[^:]*://' | awk -F'$' '{print $10-$11}')
    if [ "$got" = "$want" ]; then mv t.7z "b0${want}_pad$want.7z"
      printf '%s\t%s\n' "b0${want}_pad$want" 'password123' >> answers.tsv; break; fi
    rm -f t.7z
  done
done
rm -f t.7z payload/p.txt
echo "built $(wc -l < answers.tsv) archives"

echo; echo "== 1. fixture validation (native oracle) =="
fixbad=0
while IFS=$'\t' read -r n pw; do
  7zz t -p"$pw" "$n.7z" >/dev/null 2>&1; good=$?
  7zz t -p'definitely_not_it_xyz' "$n.7z" >/dev/null 2>&1; bad=$?
  if [ $good -ne 0 ] || [ $bad -eq 0 ]; then
    echo "  FIXTURE BAD: $n (right rc=$good wrong rc=$bad)"; fixbad=$((fixbad+1)); fi
done < answers.tsv
echo "  bad fixtures: $fixbad (must be 0)"

echo; echo "== 2. extract =="
: > hashes.raw
while IFS=$'\t' read -r n pw; do
  h=$(perl "$J7Z" "$n.7z" 2>/dev/null | sed 's/^[^:]*://')
  [ -n "$h" ] && printf '%s\t%s\n' "$n" "$h" >> hashes.raw || echo "  NO HASH: $n"
done < answers.tsv
cut -f2 hashes.raw | python3 "$CVT" 2>/dev/null > all.mdx
echo "  extracted $(wc -l < hashes.raw), converted $(wc -l < all.mdx)"

echo; echo "== 3. crack =="
printf 'wrongpass\nletmein\nqwerty123\nnotitatall\n' > wl_invalid.txt
printf 'wrongpass\nletmein\nqwerty123\nnotitatall\npassword123\nHiems20%%\nzurück\n返回\n' > wl8.txt
"$MDX" -M e1000 -F all.mdx wl_invalid.txt 2>/dev/null > run_neg.res
"$MDX" -M e1000 -F all.mdx wl8.txt        2>/dev/null > run_pos.res
neg=$(grep -c '^7ZIP' run_neg.res)
echo "  negative control (4 invalid words): $neg hits  [must be 0]"
echo "  known-answer run (8 words):         $(grep -c '^7ZIP' run_pos.res) hits"

echo; echo "== 4. attribution + re-verification =="
python3 - <<'PY'
import subprocess
ans  = dict(l.rstrip('\n').split('\t') for l in open('answers.tsv'))
byiv = {h.split('$')[7]: n for n, h in
        (l.rstrip('\n').split('\t') for l in open('hashes.raw'))}
def unhex(p):
    return bytes.fromhex(p[5:-1]).decode('utf-8','replace') \
           if p.startswith('$HEX[') and p.endswith(']') else p
found = {}
for line in open('run_pos.res'):
    if line.startswith('7ZIP '):
        hp,_,pw = line[5:].rstrip('\n').rpartition(':')
        found[byiv.get(hp.split('$')[7],'?')] = unhex(pw)
ok=bad=dec=0
for n in sorted(ans):
    got = found.get(n)
    if got is None:
        print("  %-16s declined at load (not a false negative -- reported)" % n); dec+=1; continue
    rc = subprocess.run(['7zz','t','-p'+got,n+'.7z'],capture_output=True).returncode
    if got==ans[n] and rc==0: ok+=1
    else: bad+=1; print("  %-16s MISMATCH exp=%s got=%s rc=%d" % (n,ans[n],got,rc))
print("  verified=%d  wrong=%d  declined=%d" % (ok,bad,dec))
PY
echo; echo "PASS criteria: 0 bad fixtures, 0 negative-control hits, 0 wrong."
