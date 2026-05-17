#!/usr/bin/env python3
"""GPU `_core.cl` source code generator.

Per the codegen-reconsideration memo of 2026-05-06 (option d hybrid),
this tool emits per-algorithm extension files (gpu_<name>_core.cl) for
the salted B6 batch — fanning out from a small per-algo spec table in
specs.py instead of hand-writing each core file.

Scope:
  - Salted MD-family algos shaped like MD5SALT / MD5SALTPASS.
  - Does NOT regenerate the 32 hand-written unsalted cores.
  - Does NOT touch host wiring (gpu_opencl.c et al). Emit-only.

Usage:
  python3 codegen.py --all                      Emit every spec into ./out/
  python3 codegen.py --algo md5salt -o /tmp/x   Emit one spec into /tmp/x
  python3 codegen.py --check                    Regen all specs into a temp
                                                dir and diff against the
                                                shipped gpu/<name>_core.cl
                                                files (semantic check; see
                                                README.md for what counts
                                                as semantic).

Workflow to add a new algo:
  1. Add an AlgoSpec(...) entry to specs.py.
  2. Pick the matching salt-position template (or add a new fragment if
     a never-before-shipped salt shape).
  3. Run `python3 codegen.py --algo <name> -o ../`     (writes gpu_<name>_core.cl)
  4. Run `python3 ../cl2str.py gpu_<name>_core.cl`     (writes _str.h)
  5. Wire the host: gpu_opencl.c (~6 sites), template_id_for_job(),
     etc. (see project_memo_b_dispatch_template.md §B6 wiring checklist).
  6. Build mdxfind on .205 via mdx-build, run validation matrix on
     ioblade.

Implementation note: deliberately uses `{{name}}` placeholder syntax with
hand-rolled str.replace() — no jinja2 dependency. The placeholder
vocabulary is intentionally tiny so this remains a 200-line tool.
"""

import argparse
import os
import sys
import difflib
import tempfile

# ---- import specs without forcing a package layout -----------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import specs  # noqa: E402
from specs import AlgoSpec, SaltPosition, IterShape  # noqa: E402

TEMPLATE_DIR = os.path.join(THIS_DIR, "templates")
GPU_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------
def select_main_template(spec: AlgoSpec) -> str:
    """Pick the top-level .cl.tmpl for a spec.

    MD-family (LE message-word build, LE length encoding) and SHA-family
    (BE message-word build, BE length encoding) live in sibling templates
    rather than a single template parameterized by a {{DIGEST_ENDIAN_-
    TOKEN}} — splits over parameterization per the codegen-reconsideration
    memo. The base_algo axis selects the family; the salt_position axis
    selects the finalize fragment within the family."""
    if spec.base_algo in ("md5",):
        if spec.salt_position in (SaltPosition.PREPEND,
                                  SaltPosition.APPEND,
                                  SaltPosition.APPEND_TO_HEX32):
            return os.path.join(TEMPLATE_DIR, "md_style_salted.cl.tmpl")
    if spec.base_algo == "sha1":
        # B6.1 SHA1 fan-out (2026-05-06): SHA1 sibling template carries
        # 5-word state, BE message-word build, BE length encoding,
        # bswap32-on-probe, EMIT_HIT_5. SHA384/512 will need a 64-bit-
        # state sibling (different IV constants, 128-byte block,
        # 16-uint64 schedule) — separate template.
        # B6.5 SHA1PASSSALT fan-out (2026-05-06): the SHA1 main template
        # is salt-position-agnostic — only the {{TEMPLATE_FINALIZE_BODY}}
        # slot (filled from the per-spec finalize fragment) sees the
        # APPEND vs PREPEND distinction. Same template, different
        # fragment. Iter loop is identical (40-char hex feedback with
        # FRESH SHA1 IV — no salt re-application — matches both
        # JOB_SHA1SALTPASS and JOB_SHA1PASSSALT CPU iter shape).
        if spec.salt_position in (SaltPosition.PREPEND,
                                  SaltPosition.APPEND):
            return os.path.join(TEMPLATE_DIR, "sha1_style_salted.cl.tmpl")
    if spec.base_algo == "sha256":
        # B6.2 SHA256 fan-out (2026-05-06): SHA256 sibling template carries
        # 8-word state, BE message-word build, BE length encoding,
        # bswap32-on-probe, EMIT_HIT_8. Iter loop differs from SHA1's:
        # SHA256's 64-char hex output exactly fills one block, so the pad
        # lands in a SECOND block (vs SHA1's 40-char hex which fits in
        # one block).
        #
        # B6.3 SHA224 fan-out (2026-05-06): SHA224 reuses sha256_block but
        # truncates output to 7 words = 56 hex chars. Iter pad layout
        # differs from SHA256 (block 1 holds data + 0x80; block 2 = length).
        # Per "splits over parameterization", SHA224 gets a sibling
        # template (sha224_style_salted.cl.tmpl). Discriminator within
        # base_algo="sha256": hash_words (8 = SHA256, 7 = SHA224).
        # SHA384/512 family uses 64-bit words — separate template family.
        #
        # B6.7 SHA256PASSSALT fan-out (2026-05-06): the SHA256 main template
        # is salt-position-agnostic in the same way SHA1's is — only the
        # {{TEMPLATE_FINALIZE_BODY}} slot (filled from the per-spec finalize
        # fragment) sees the APPEND vs PREPEND distinction. Same template,
        # different fragment (finalize_append_be.cl.frag from B6.5). Iter
        # loop is identical: 64-char hex feedback with FRESH SHA256 IV (no
        # salt re-application) — matches both JOB_SHA256SALTPASS and
        # JOB_SHA256PASSSALT CPU iter shapes (mdxfind.c:27667-27672).
        if spec.salt_position in (SaltPosition.PREPEND,
                                  SaltPosition.APPEND):
            if spec.hash_words == 7:
                return os.path.join(TEMPLATE_DIR, "sha224_style_salted.cl.tmpl")
            return os.path.join(TEMPLATE_DIR, "sha256_style_salted.cl.tmpl")
    if spec.base_algo == "sha512":
        # B6.9 SHA512 fan-out (2026-05-06): first 64-bit-state salted
        # template. State width is 8 × uint64 (vs SHA-256's 8 × uint32);
        # block size 128 bytes (vs 64); length field 128-bit BE in
        # M[14..15] (each ulong = 64 bits, vs 64-bit BE in two uint
        # words for SHA-1/224/256). Per "splits over parameterization",
        # SHA-512 gets a sibling template (sha512_style_salted.cl.tmpl)
        # because all three of those axes (block size, word width,
        # length encoding) cannot be parameterized cleanly into the
        # SHA-256 template without #if-blocks at every M[] read/write
        # site. HASH_WORDS=16 represents the 16 LE-byteswapped uint32
        # emit/probe words (= 8 BE ulong state words). The iter loop
        # also differs (128-char hex output exactly fills one 128-byte
        # block; 0x80 + 16-byte length lands in a second block).
        #
        # B6.10 SHA512PASSSALT fan-out (2026-05-06): the SHA-512 main
        # template is salt-position-agnostic in the same way SHA-1's
        # and SHA-256's are — only the {{TEMPLATE_FINALIZE_BODY}} slot
        # (filled from the per-spec finalize fragment) sees the APPEND
        # vs PREPEND distinction. Same template, different fragment
        # (finalize_append_be64.cl.frag — the 64-bit-state APPEND
        # sibling authored in B6.10). Iter loop is identical: 128-char
        # hex feedback with FRESH SHA-512 IV (no salt re-application)
        # — matches both JOB_SHA512SALTPASS and JOB_SHA512PASSSALT
        # CPU iter shapes (mdxfind.c JOB_SHA512SALTPASS lines 14008-
        # 14017 and JOB_SHA512PASSSALT lines 14069-14127, both running
        # the same prmd5(curin.h, newbuf, 128); mysha512(newbuf, 128, ...)
        # iter step).
        #
        # Family E HMAC-SHA384 carrier (2026-05-08): hash_words=12
        # discriminates SHA-384 from SHA-512 within base_algo="sha512"
        # (same compression primitive sha512_block, different IV +
        # output truncation + iter hex length). Mirrors the SHA-256 vs
        # SHA-224 split where hash_words 8 vs 7 chooses sha256 vs sha224
        # template. SHA-384 needs its own template because EMIT_HIT_12
        # (vs EMIT_HIT_16) and template_state_to_h truncation (6 ulong
        # vs 8) differ — these can't be threaded into sha512_style with
        # #if blocks cleanly per the codegen-reconsideration "splits
        # over parameterization" rule.
        if spec.salt_position in (SaltPosition.PREPEND,
                                  SaltPosition.APPEND):
            if spec.hash_words == 12:
                return os.path.join(TEMPLATE_DIR, "sha384_style_salted.cl.tmpl")
            return os.path.join(TEMPLATE_DIR, "sha512_style_salted.cl.tmpl")
    if spec.base_algo == "rmd160":
        # Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD-160 sibling
        # template carries 5-word LE state, LE message-word build, LE
        # length encoding, LE-direct probe + emit (NO bswap32; matches
        # gpu_ripemd160_core.cl rev 1.1 LE convention), EMIT_HIT_5,
        # 2-arg ripemd160_block(state_ptr, M) compression call (vs
        # MD5's 4-arg or SHA-1's 1-arg-state-pointer-but-BE compression).
        # Per the codegen-reconsideration memo's "splits over
        # parameterization" rule, RMD gets its own template+fragment
        # because the LE+5-word+2-arg-block combination doesn't fit any
        # existing template/fragment cleanly.
        if spec.salt_position in (SaltPosition.PREPEND,
                                  SaltPosition.APPEND):
            return os.path.join(TEMPLATE_DIR, "rmd160_style_salted.cl.tmpl")
    if spec.base_algo == "rmd320":
        # Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD-320 sibling
        # template carries 10-word LE state (vs RMD160's 5), LE message-
        # word build, LE length encoding, LE-direct probe + emit (NO
        # bswap32), EMIT_HIT_10, 2-arg rmd320_block(state_ptr, M)
        # compression call (same signature as rmd160_block — only the
        # round bodies + line/line' accumulation differ). Per the codegen-
        # reconsideration memo's "splits over parameterization" rule,
        # RMD320 gets its own SIBLING main template (rmd320_style_salted.cl
        # .tmpl) — the rmd160_style template is NOT parametric in
        # HASH_WORDS for the iter loop (RMD160 fits in one block @ 40
        # hex chars; RMD320 needs two blocks @ 80 hex chars). Mirrors
        # the sha224 vs sha256 / sha384 vs sha512 split-over-
        # parameterization precedent. The shared finalize_prepend_-
        # rmd.cl.frag fragment IS HASH_WORDS-parametric and serves both
        # templates (HASH_WORDS == 5 branch for RMD160, HASH_WORDS == 10
        # branch for RMD320).
        if spec.salt_position in (SaltPosition.PREPEND,
                                  SaltPosition.APPEND):
            return os.path.join(TEMPLATE_DIR, "rmd320_style_salted.cl.tmpl")
    raise NotImplementedError(
        "no template available for base_algo=%s salt_position=%s — "
        "add a new fragment under templates/ and a selector branch here."
        % (spec.base_algo, spec.salt_position))


def select_finalize_fragment(spec: AlgoSpec) -> str:
    """Pick the body that goes into the {{TEMPLATE_FINALIZE_BODY}} slot.

    Fragment selection is on (salt_position, digest_endianness):
    PREPEND splits between LE (MD-family) and BE (SHA-family) variants
    because the byte-position math differs (LE: byte i in word i/4 at
    shift (i&3)*8; BE: same word but shift (3-(i&3))*8) and length
    encoding differs (LE: M[14]=bits, M[15]=0; BE: M[14]=0, M[15]=bits).
    APPEND_TO_HEX32 is currently MD-family-only. APPEND is a stub."""
    sp = spec.salt_position
    if sp == SaltPosition.PREPEND:
        # B6.1 SHA1 fan-out: BE variant for SHA-family. The base_algo
        # axis selects within the PREPEND family. md5 uses the original
        # LE PREPEND fragment; sha1 / sha256 use the BE PREPEND fragment
        # (the BE fragment is fully parameterized via {{BASE_ALGO}} +
        # HASH_BLOCK_BYTES and works for any SHA-family algo with a
        # 64-byte block; SHA384/512 with 128-byte blocks would need a
        # different fragment).
        # B6.9 SHA512 fan-out (2026-05-06): the BE 32-bit-word fragment
        # CANNOT be reused for SHA-512 — it is hard-coded to uint M[16]
        # (64-byte block), 56-byte tail-fits threshold, and a 32-bit
        # length field. SHA-512 needs ulong M[16] (128-byte block),
        # 112-byte threshold, and a 128-bit length field. Per the
        # codegen-reconsideration memo's "width-bearing constants
        # belong in templates not fragments" rule (and the related
        # codegen_fragment_width_bugs feedback), we author a sibling
        # 64-bit BE PREPEND fragment (finalize_prepend_be64.cl.frag)
        # instead of #if-tangling the existing 32-bit fragment.
        if spec.base_algo == "sha512":
            return os.path.join(TEMPLATE_DIR, "finalize_prepend_be64.cl.frag")
        if spec.base_algo in ("rmd160", "rmd320"):
            # Family G HMAC-RIPEMD-160 carrier (2026-05-08): the LE 32-bit
            # PREPEND fragment for the RMD family. CANNOT reuse the MD5
            # LE PREPEND fragment (4-arg md5_block call signature differs
            # from RMD's 2-arg ripemd<N>_block(state_ptr, M)) NOR the
            # SHA-family BE PREPEND fragment (wrong endianness). Fragment
            # is parametric for the RMD family via {{BASE_ALGO}} +
            # HASH_WORDS gate; HASH_WORDS == 5 branch handles RMD160,
            # HASH_WORDS == 10 branch handles RMD320 (Family H, 2026-05-08).
            # The shared 2-arg rmd<N>_block call signature is what makes
            # this single fragment serve both; only the per-step round
            # bodies + state-accumulation pattern differ between rmd160
            # and rmd320 (those live in gpu_common.cl). The mode-0
            # RMD<N>(salt||pass) main body in the fragment is
            # structurally unreachable in production (host always sets
            # algo_mode 5 or 6 for HMAC dispatch).
            return os.path.join(TEMPLATE_DIR, "finalize_prepend_rmd.cl.frag")
        if spec.base_algo in ("sha1", "sha256"):
            # SHA224 (hash_words=7, base_algo="sha256") also uses the BE
            # PREPEND fragment — the fragment is fully family-agnostic
            # via {{BASE_ALGO}} substitution; the output truncation is
            # handled in the per-template iterate, not in the fragment.
            return os.path.join(TEMPLATE_DIR, "finalize_prepend_be.cl.frag")
        return os.path.join(TEMPLATE_DIR, "finalize_prepend.cl.frag")
    if sp == SaltPosition.APPEND_TO_HEX32:
        return os.path.join(TEMPLATE_DIR, "finalize_append_to_hex32.cl.frag")
    if sp == SaltPosition.APPEND:
        # B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted
        # variant on the codegen path. MD-family LE: password bytes first
        # (offset 0..plen-1), salt bytes second (offset plen..plen+slen-1),
        # then 0x80 padding, then LE length-in-bits in M[14..15].
        # B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-
        # shape variant — uses sibling BE fragment finalize_append_be.cl.frag
        # (same pass-then-salt order, BE byte placement + BE length
        # encoding). Same BE/LE split rationale as PREPEND's BE/LE pair.
        # The BE APPEND fragment is fully family-agnostic via {{BASE_-
        # ALGO}} + HASH_BLOCK_BYTES substitution; future SHA256PASSSALT /
        # SHA224PASSSALT reuse this fragment with no further authoring.
        # B6.7 SHA256PASSSALT fan-out (2026-05-06): SHA256(pass || salt) —
        # second SHA-family APPEND-shape variant. Pure spec reuse — the
        # BE APPEND fragment is family-agnostic and slots in unchanged
        # behind sha256_style_salted.cl.tmpl. defines_str disambiguates
        # from SHA256SALTPASS via SALT_POSITION=APPEND (vs PREPEND); same
        # BASE_ALGO=sha256 + HASH_WORDS=8.
        if spec.base_algo in ("md5",):
            return os.path.join(TEMPLATE_DIR, "finalize_append.cl.frag")
        if spec.base_algo in ("sha1", "sha256"):
            # SHA224 (hash_words=7, base_algo="sha256") would also use
            # this fragment — fragment is family-agnostic via {{BASE_-
            # ALGO}}; output truncation is handled in the per-template
            # iterate, not in this fragment.
            return os.path.join(TEMPLATE_DIR, "finalize_append_be.cl.frag")
        # B6.10 SHA512PASSSALT fan-out (2026-05-06): the 32-bit BE APPEND
        # fragment CANNOT be reused for SHA-512 — same width-bearing
        # mismatch as the PREPEND case (uint M[16] vs ulong M[16],
        # 64-byte block vs 128, 32-bit length field vs 128-bit). Per
        # the codegen-reconsideration memo's "splits over parameterization"
        # rule we author a sibling 64-bit BE APPEND fragment
        # (finalize_append_be64.cl.frag) instead of #if-tangling the
        # existing 32-bit fragment. The 64-bit APPEND fragment mirrors
        # the 64-bit PREPEND sibling (B6.9) for byte-source ordering
        # (pass-first vs salt-first); same M[16] scratch, same per-byte
        # BE position math, same 112-byte tail-fits threshold, same
        # 128-bit BE length field in M[14..15].
        if spec.base_algo == "sha512":
            return os.path.join(TEMPLATE_DIR, "finalize_append_be64.cl.frag")
        raise NotImplementedError(
            "salt_position=APPEND for base_algo=%s — no APPEND fragment "
            "available for this family." % spec.base_algo)
    raise NotImplementedError("salt_position=%s" % sp)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _read(path: str) -> str:
    with open(path, "r") as fh:
        return fh.read()


def _placeholders(spec: AlgoSpec) -> dict:
    return {
        "{{NAME}}": spec.name,
        "{{ONE_LINER}}": spec.one_liner,
        "{{CPU_REFERENCE}}": spec.cpu_reference,
        "{{SALT_POSITION_TOKEN}}": spec.salt_position_token,
        "{{ITER_SHAPE}}": spec.iter_shape.value,
        "{{ITER_NOTE}}": spec.iter_note,
        "{{BASE_ALGO}}": spec.base_algo,
        "{{HASH_WORDS}}": str(spec.hash_words),
        "{{HASH_BLOCK_BYTES}}": str(spec.hash_block_bytes),
    }


def render(spec: AlgoSpec) -> str:
    """Render one spec to a complete .cl source string."""
    main = _read(select_main_template(spec))
    frag = _read(select_finalize_fragment(spec))

    placeholders = _placeholders(spec)

    # Substitute base-algo-style placeholders into the fragment first
    # (the fragment also mentions {{BASE_ALGO}} for md5_block / md5_to_hex_lc).
    for k, v in placeholders.items():
        frag = frag.replace(k, v)

    # The fragment as a whole occupies the {{TEMPLATE_FINALIZE_BODY}} slot.
    # The trailing newline of the fragment is dropped to keep template-driven
    # whitespace tidy.
    out = main.replace("{{TEMPLATE_FINALIZE_BODY}}", frag.rstrip("\n"))

    # Apply remaining placeholders to the main body.
    for k, v in placeholders.items():
        out = out.replace(k, v)

    # Sanity: all placeholders resolved.
    if "{{" in out:
        for line_no, line in enumerate(out.splitlines(), start=1):
            if "{{" in line:
                raise RuntimeError(
                    "unresolved placeholder at line %d: %r" % (line_no, line))

    return out


def emit(spec: AlgoSpec, output_dir: str) -> str:
    """Write the rendered spec to <output_dir>/gpu_<name>_core.cl."""
    out_path = os.path.join(output_dir, "gpu_%s_core.cl" % spec.name)
    src = render(spec)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(src)
    return out_path


# ---------------------------------------------------------------------------
# --check: semantic diff against shipped reference
# ---------------------------------------------------------------------------
SEMANTIC_TOKENS = (
    "template_state", "template_init", "template_transform",
    "template_finalize", "template_iterate", "template_digest_compare",
    "template_emit_hit", "template_emit_hit_or_overflow",
    "HASH_WORDS", "HASH_BLOCK_BYTES",
    "probe_compact_idx",
    # Per-family primitives. Tokens are checked one-way: shipped uses ->
    # generated must use. md5_block / md5_to_hex_lc are MD-family only;
    # sha1_block is SHA-family only. The check loop is per-spec, so a
    # spec that doesn't reference a token won't trigger the gate.
    "md5_block", "md5_to_hex_lc",
    "sha1_block",
    "sha256_block",
    "sha512_block", "sha512_to_hex_lc",
    "EMIT_HIT_4", "EMIT_HIT_5", "EMIT_HIT_8", "EMIT_HIT_16",
)


def _strip_comments_and_blank(src: str) -> str:
    """Crude C-comment stripper for semantic-only comparison.

    This is intentionally not a full C parser — it removes /* ... */ blocks
    and // line comments, collapses runs of whitespace, and drops blank
    lines. The resulting string captures the algorithmic content well
    enough for drift detection (intentional functional changes WILL show
    as a diff; whitespace-only or comment-only differences will not).
    """
    out = []
    i = 0
    n = len(src)
    in_block = False
    in_line = False
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_line:
            if c == "\n":
                in_line = False
                out.append("\n")
            i += 1
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if c == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        out.append(c)
        i += 1
    text = "".join(out)
    # collapse whitespace runs to single spaces; preserve newlines for
    # readable diff output.
    norm_lines = []
    for line in text.splitlines():
        s = " ".join(line.split())
        if s:
            norm_lines.append(s)
    return "\n".join(norm_lines) + "\n"


def check(spec: AlgoSpec, shipped_dir: str) -> int:
    """Return 0 if generated == shipped (modulo comments/whitespace),
    nonzero with a printed diff otherwise."""
    shipped_path = os.path.join(shipped_dir, "gpu_%s_core.cl" % spec.name)
    if not os.path.isfile(shipped_path):
        print("[%s] no shipped reference at %s — skipping check"
              % (spec.name, shipped_path))
        return 0

    generated = render(spec)
    shipped = _read(shipped_path)

    g = _strip_comments_and_blank(generated)
    s = _strip_comments_and_blank(shipped)

    # Sanity: both sides reference the same semantic tokens.
    for tok in SEMANTIC_TOKENS:
        if tok in shipped and tok not in generated:
            print("[%s] CHECK FAIL: shipped uses %r but generated does not"
                  % (spec.name, tok))
            return 1

    if g == s:
        print("[%s] check OK (semantic match)" % spec.name)
        return 0

    print("[%s] CHECK DIFF (semantic):" % spec.name)
    diff = difflib.unified_diff(
        s.splitlines(keepends=True),
        g.splitlines(keepends=True),
        fromfile="shipped/%s" % os.path.basename(shipped_path),
        tofile="generated/gpu_%s_core.cl" % spec.name,
        n=3,
    )
    sys.stdout.writelines(diff)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", help="emit/check a single algo by name")
    p.add_argument("--all", action="store_true",
                   help="operate on every spec in specs.py")
    p.add_argument("--check", action="store_true",
                   help="diff generated against shipped (no write)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="where to write generated .cl files (default: ./out/)")
    args = p.parse_args()

    if not args.algo and not args.all:
        p.error("specify --algo NAME or --all")

    selected = (specs.ALGOS if args.all
                else [specs.by_name(args.algo)])

    if args.check:
        rc = 0
        for s in selected:
            rc |= check(s, GPU_DIR)
        sys.exit(rc)

    out_dir = args.output_dir or os.path.join(THIS_DIR, "out")
    for s in selected:
        path = emit(s, out_dir)
        print("[%s] wrote %s (%d bytes)"
              % (s.name, path, os.path.getsize(path)))


if __name__ == "__main__":
    _cli()
