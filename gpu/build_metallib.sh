#!/bin/bash
# build_metallib.sh — Compile all Metal kernel sources into a single mdxfind.metallib
#
# Usage: cd gpu && ./build_metallib.sh
# Or:    gpu/build_metallib.sh   (from project root)
#
# Produces: gpu/mdxfind.metallib (architecture-neutral AIR format)
#
# Strategy: concatenate metal_common.metal + all family sources into one big
# compilation unit, then compile to .air and link to .metallib. This avoids
# duplicate symbol errors from having metal_common in multiple .air files.
#
# Self-contained families (sha512unsalted) are compiled as a separate .air
# and linked alongside the main one.
#
# Families that fail offline compilation (streebog, hmac_sha512) are excluded
# and will fall back to JIT at runtime.

set -e

# Determine script directory (gpu/) regardless of where we're invoked from
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMMON=metal_common.metal
OUTLIB=mdxfind.metallib
AIRDIR=/tmp/mdxfind_metalair_$$
METAL_FLAGS="-std=macos-metal2.4 -mmacos-version-min=13.0 -O2 -Wno-unused-function"

mkdir -p "$AIRDIR"
trap 'rm -rf "$AIRDIR"' EXIT

# Families that need metal_common prepended (compiled as one combined unit)
COMMON_FAMILIES=(
    metal_md5salt
    metal_md5saltpass
    metal_md5_md5saltmd5pass
    metal_sha256
    metal_phpbb3
    metal_descrypt
    metal_md5unsalted
    metal_md4unsalted
    metal_sha1unsalted
    metal_sha256unsalted
    metal_wrlunsalted
    metal_md6256unsalted
    metal_keccakunsalted
    metal_mysql3unsalted
    metal_hmac_rmd160
    metal_hmac_rmd320
    metal_hmac_blake2s
    metal_sha512crypt
    metal_sha256crypt
    metal_rmd160unsalted
    metal_blake2s256unsalted
    metal_bcrypt
    metal_sha1
    metal_md5crypt
)

# Self-contained families (compiled separately, have own MetalParams/preamble)
SELF_CONTAINED_FAMILIES=(
    metal_sha512unsalted
)

# Common-dependent families that need separate compilation (too large for combined)
SEPARATE_COMMON_FAMILIES=(
    metal_streebog
    metal_hmac_sha512
)

AIR_FILES=""

# --- Build main combined source (metal_common + all common-dependent families) ---
echo "  [combine] $COMMON + ${#COMMON_FAMILIES[@]} families"
COMBINED="${AIRDIR}/mdxfind_combined.metal"
cp "$COMMON" "$COMBINED"
for fam in "${COMMON_FAMILIES[@]}"; do
    src="${fam}.metal"
    if [ ! -f "$src" ]; then
        echo "WARNING: $src not found, skipping" >&2
        continue
    fi
    echo "" >> "$COMBINED"
    echo "/* ---- ${fam} ---- */" >> "$COMBINED"
    cat "$src" >> "$COMBINED"
done

echo "  [metal] compiling combined source ($(wc -c < "$COMBINED" | tr -d ' ') bytes)"
if xcrun -sdk macosx metal $METAL_FLAGS -c "$COMBINED" -o "${AIRDIR}/mdxfind_combined.air" 2>&1; then
    AIR_FILES="${AIRDIR}/mdxfind_combined.air"
else
    echo "ERROR: combined source failed to compile" >&2
    exit 1
fi

# --- Build separate common-dependent families (common + family, compiled individually) ---
for fam in "${SEPARATE_COMMON_FAMILIES[@]}"; do
    src="${fam}.metal"
    air="${AIRDIR}/${fam}.air"
    sep="${AIRDIR}/${fam}_sep.metal"
    if [ ! -f "$src" ]; then
        echo "WARNING: $src not found, skipping" >&2
        continue
    fi
    cp "$COMMON" "$sep"
    echo "" >> "$sep"
    cat "$src" >> "$sep"
    echo "  [metal] $src (separate common-dependent)"
    if xcrun -sdk macosx metal $METAL_FLAGS -c "$sep" -o "$air" 2>&1; then
        AIR_FILES="$AIR_FILES $air"
    else
        echo "WARNING: $src failed to compile, will use JIT" >&2
    fi
done

# --- Build self-contained families as separate .air files ---
for fam in "${SELF_CONTAINED_FAMILIES[@]}"; do
    src="${fam}.metal"
    air="${AIRDIR}/${fam}.air"
    if [ ! -f "$src" ]; then
        echo "WARNING: $src not found, skipping" >&2
        continue
    fi
    echo "  [metal] $src (self-contained)"
    if xcrun -sdk macosx metal $METAL_FLAGS -c "$src" -o "$air" 2>&1; then
        AIR_FILES="$AIR_FILES $air"
    else
        echo "WARNING: $src failed to compile, will use JIT" >&2
    fi
done

if [ -z "$AIR_FILES" ]; then
    echo "ERROR: no .air files produced" >&2
    exit 1
fi

echo "  [metallib] linking $OUTLIB"
xcrun -sdk macosx metallib $AIR_FILES -o "$OUTLIB"

echo "Built $OUTLIB ($(wc -c < "$OUTLIB" | tr -d ' ') bytes)"
