"""Per-algorithm specs for the gpu_<algo>_core.cl code generator.

Each spec is a small structured-data record. The generator (codegen.py)
selects a template based on (base_algo, salt_position, iter_shape) and
substitutes the spec's fields into the template's {{...}} placeholders.

Scope: B6 salted algos that fit the gpu_template.cl extension model
(MD5SALT family, simple PREPEND/APPEND salt shapes, double-MD5 chains).
The 32 unsalted cores stay hand-written per the codegen-reconsideration
memo of 2026-05-06; the codegen tool does NOT regenerate them.

Validation gate: `codegen.py --check` regenerates each spec into
/tmp/codegen_out/ and diffs the algorithmic semantics against the
shipped gpu_<name>_core.cl reference. Cosmetic diffs (whitespace,
comment wording) are expected; the diff target is byte-exactness of
the algorithmic body (template_state, template_init, template_transform,
template_finalize, template_iterate, template_digest_compare, the two
template_emit_hit macros, and the HASH_WORDS/HASH_BLOCK_BYTES defines).

The validation oracle that matters most is *runtime byte-exactness on
GPU* — that is performed via building mdxfind with the generated cores
swapped in and running the salted validation matrix on ioblade.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SaltPosition(Enum):
    """How salt is mixed with the password buffer in template_finalize.

    NONE             — unsalted (32 hand-written cores; not in codegen scope)
    PREPEND          — MD5(salt || pass)             — JOB_MD5SALTPASS
    APPEND           — MD5(pass || salt)             — JOB_MD5SALTRAW family
    APPEND_TO_HEX32  — MD5(hex32(MD5(pass)) || salt) — JOB_MD5SALT
    """
    NONE = "NONE"
    PREPEND = "PREPEND"
    APPEND = "APPEND"
    APPEND_TO_HEX32 = "APPEND_TO_HEX32"


class IterShape(Enum):
    """Shape of the -i iteration loop step.

    NONE          — no per-iter rehash (iter=1 is the only path)
    HEX_FEEDBACK  — re-hex the prior digest, MD5 with fresh IV (no salt re-application)
    BIN_FEEDBACK  — re-feed the binary digest into a fresh-IV transform
    """
    NONE = "NONE"
    HEX_FEEDBACK = "HEX_FEEDBACK"
    BIN_FEEDBACK = "BIN_FEEDBACK"


class DigestEndian(Enum):
    """Endianness of digest words for emit + probe.

    LE_DIRECT — words are little-endian (MD-family default)
    BE_BSWAP  — words need bswap before emit (SHA-family)
    """
    LE_DIRECT = "LE_DIRECT"
    BE_BSWAP = "BE_BSWAP"


@dataclass
class AlgoSpec:
    """Spec for one salted GPU algorithm core."""

    # --- identifying ---
    name: str                       # short tag → filename gpu_<name>_core.cl
    job_enum: str                   # mdxfind enum, e.g. "JOB_MD5SALT"
    template_enum_value: int        # GPU_TEMPLATE_<NAME> integer value
    base_algo: str                  # "md5", "sha1", "sha256", ...

    # --- geometry (must match template defines) ---
    hash_words: int                 # 4 (MD5), 5 (SHA1), 8 (SHA256), ...
    hash_block_bytes: int           # 64 (MD-family), 128 (SHA-512 family)

    # --- salt / iter behavior ---
    salt_position: SaltPosition
    iter_shape: IterShape

    # --- emit / probe / digest layout ---
    digest_endianness: DigestEndian
    emit_width: int                 # number of digest words emitted (4..16)

    # --- hashcat / mdxfind reference ---
    hashcat_mode: str               # "10", "20", "30", ...
    cpu_reference: str              # comment: "mdxfind.c JOB_*** at lines NNNN-MMMM"

    # --- cache disambiguation (defines_str) ---
    salt_position_token: str = ""   # auto-derived if blank

    # --- one-liner identity comment ---
    one_liner: str = ""             # used in the file header

    # --- author guidance comments (algorithm-specific) ---
    iter_note: str = ""             # extra commentary for template_iterate

    def __post_init__(self):
        if not self.salt_position_token:
            self.salt_position_token = self.salt_position.value


# ---------------------------------------------------------------------------
# Initial spec list — MD5SALT + MD5SALTPASS, both with shipped reference cores.
# These two are the validation gate for the codegen tool itself.
# ---------------------------------------------------------------------------
ALGOS: List[AlgoSpec] = [
    AlgoSpec(
        name="md5salt",
        job_enum="JOB_MD5SALT",
        template_enum_value=33,
        base_algo="md5",
        hash_words=4,
        hash_block_bytes=64,
        salt_position=SaltPosition.APPEND_TO_HEX32,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.LE_DIRECT,
        emit_width=4,
        hashcat_mode="10",
        cpu_reference="mdxfind.c JOB_MD5SALT at lines 21943-21974",
        one_liner=("MD5SALT (JOB_MD5SALT, hashcat -m 10): "
                   "MD5(hex32(MD5(pass)) || salt) — DOUBLE-MD5 chain"),
        iter_note=(
            "for x > 1, mdxfind iter feedback for JOB_MD5SALT is the "
            "lowercase hex of the prior 32-char digest re-hashed (mdxfind.c "
            "default `prmd5` at line 9706). The default HEX_FEEDBACK template_iterate "
            "matches this exactly. We REUSE the unsalted MD5 template_iterate — "
            "no per-iter salt re-application (matches CPU JOB_MD5SALT iter convention)."),
    ),
    AlgoSpec(
        name="md5saltpass",
        job_enum="JOB_MD5SALTPASS",
        template_enum_value=34,
        base_algo="md5",
        hash_words=4,
        hash_block_bytes=64,
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.LE_DIRECT,
        emit_width=4,
        hashcat_mode="20",
        cpu_reference="mdxfind.c JOB_MD5SALTPASS at lines 15776-15832",
        one_liner=("MD5SALTPASS (JOB_MD5SALTPASS, hashcat -m 20): "
                   "MD5(salt || pass) — simple PREPEND-salt MD5"),
        iter_note=(
            "lowercase hex of prior 32-char digest re-MD5'd with FRESH IV "
            "(no salt re-application). Lines 15819-15827. Identical to "
            "unsalted MD5 template_iterate."),
    ),
    # B6.1 SHA1 fan-out (2026-05-06): first SHA-family salted variant.
    # Validates the codegen tool's "splits over parameterization" design
    # principle — the BE/LE concern justified a sibling template
    # (sha1_style_salted.cl.tmpl) and a sibling fragment
    # (finalize_prepend_be.cl.frag) rather than threading a {{DIGEST_-
    # ENDIAN_TOKEN}} placeholder through md_style_salted.cl.tmpl with
    # 6+ #if BE blocks. After SHA1SALTPASS ships, future SHA224/256/384/
    # 512 SALTPASS specs reuse the SHA1 sibling template + fragment with
    # only HASH_WORDS / hash_block_bytes / IV adjustments.
    AlgoSpec(
        name="sha1saltpass",
        job_enum="JOB_SHA1SALTPASS",
        template_enum_value=35,
        base_algo="sha1",
        hash_words=5,
        hash_block_bytes=64,
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=5,
        hashcat_mode="110",
        cpu_reference="mdxfind.c JOB_SHA1SALTPASS at lines 14369-14418",
        one_liner=("SHA1SALTPASS (JOB_SHA1SALTPASS, hashcat -m 110): "
                   "SHA1(salt || pass) — simple PREPEND-salt SHA1"),
        iter_note=(
            "lowercase hex of prior 40-char digest re-SHA1'd with FRESH "
            "IV (no salt re-application). Lines 14411-14412: "
            "prmd5(curin.h, newbuf, 40); mysha1(newbuf, 40, ...). "
            "Identical iter shape to MD5SALTPASS but with SHA1 + 40-char "
            "hex (vs 32-char for MD5)."),
    ),
    # B6.2 SHA256 fan-out (2026-05-06): second SHA-family salted variant.
    # Reuses finalize_prepend_be.cl.frag (the BE fragment is fully
    # parameterized via {{BASE_ALGO}} + HASH_BLOCK_BYTES). Cannot reuse
    # sha1_style_salted.cl.tmpl because of the iter-loop divergence: SHA256
    # produces 64-char hex output that exactly fills one 64-byte block,
    # forcing the 0x80 pad marker to land in a SECOND block. SHA1's 40-char
    # output fits in one block (M[10]=0x80, M[15]=320). Per the README's
    # "splits over parameterization" rule, this justifies a sibling
    # sha256_style_salted.cl.tmpl (~225 lines, mirrors sha1_style with
    # 8-word state + EMIT_HIT_8 + two-block iter pad). Future SHA224SALTPASS
    # reuses sha256_style with only the IV constants differing (handled
    # via spec → template_init render pattern in a follow-up). SHA384/512
    # need a 64-bit-state sibling — separate template family.
    # B6.3 SHA224 fan-out (2026-05-06): SHA-family salted variant that
    # reuses sha256_block for compression but truncates output to 7 words.
    # Per the codegen-reconsideration memo's "splits over parameterization"
    # rule, SHA224 gets its own sibling template (sha224_style_salted.cl.tmpl)
    # because it differs from sha256_style on 6 axes: IV constants ×2,
    # iter encoding loop bound (7 vs 8 state words), iter pad-marker
    # placement (M[14] vs M[0] of block 2), iter length value (448 vs 512),
    # template_state width comment, and EMIT_HIT_7 vs EMIT_HIT_8.
    # template_state.h[8] internally (sha256_block needs 8); HASH_WORDS=7
    # represents emit + iter encoding width.
    AlgoSpec(
        name="sha224saltpass",
        job_enum="JOB_SHA224SALTPASS",
        template_enum_value=37,
        base_algo="sha256",     # uses sha256_block; routing uses hash_words=7 to pick sha224 template
        hash_words=7,
        hash_block_bytes=64,
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=7,
        hashcat_mode="1310",
        cpu_reference="mdxfind.c JOB_SHA224SALTPASS at lines 30447-30487",
        one_liner=("SHA224SALTPASS (JOB_SHA224SALTPASS, hashcat -m 1310): "
                   "SHA224(salt || pass) — simple PREPEND-salt SHA224 "
                   "(SHA256 compress, output truncated to 28 bytes)"),
        iter_note=(
            "lowercase hex of prior 56-char digest re-SHA224'd with FRESH "
            "IV (no salt re-application). Lines 30482-30484: "
            "prmd5(curin.h, newbuf, 56); mysha224(newbuf, 56, ...). "
            "Same compress as SHA256 but output is 28 bytes (vs 32) and "
            "iter input is 56 chars (vs 64) — the 56-char input fits in "
            "block 1 with 0x80 marker at M[14], length 448 in block 2."),
    ),
    # B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted
    # variant on the codegen path. Same MD-family LE compress as
    # MD5SALTPASS (md5_block, HASH_WORDS=4, EMIT_HIT_4) but salt is at
    # the END of the message, not the beginning. Cache disambiguated
    # from MD5SALTPASS via SALT_POSITION=APPEND in defines_str (vs
    # PREPEND); same BASE_ALGO=md5 + HASH_WORDS=4 axes. Authors the
    # finalize_append.cl.frag fragment which unblocks future SHA1PASSSALT
    # + SHA256PASSSALT (both APPEND, but SHA-family BE — will need a
    # sibling finalize_append_be.cl.frag in those fan-outs).
    AlgoSpec(
        name="md5passsalt",
        job_enum="JOB_MD5PASSSALT",
        template_enum_value=38,
        base_algo="md5",
        hash_words=4,
        hash_block_bytes=64,
        salt_position=SaltPosition.APPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.LE_DIRECT,
        emit_width=4,
        hashcat_mode="10",
        cpu_reference="mdxfind.c JOB_MD5PASSSALT at lines 16627-16669",
        one_liner=("MD5PASSSALT (JOB_MD5PASSSALT, hashcat -m 10): "
                   "MD5(pass || salt) — simple APPEND-salt MD5"),
        iter_note=(
            "lowercase hex of prior 32-char digest re-MD5'd with FRESH IV "
            "(no salt re-application). Lines 16661-16664: "
            "prmd5(md5buf.h, newbuf, 32); mymd5(newbuf, 32, ...). "
            "Identical iter shape to MD5SALTPASS — only the salt POSITION "
            "differs at template_finalize time."),
    ),
    # B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-
    # shape salted variant. SHA1(pass || salt). Same SHA1 compression +
    # 5-word BE state as SHA1SALTPASS but salt is at the END of the
    # message, not the beginning. Cache disambiguated from SHA1SALTPASS
    # via SALT_POSITION=APPEND in defines_str (vs PREPEND); same
    # BASE_ALGO=sha1 + HASH_WORDS=5 axes. Authors the
    # finalize_append_be.cl.frag fragment which unblocks future SHA-
    # family APPEND variants (SHA256PASSSALT becomes pure spec reuse).
    # 40-char hex iter feedback identical to SHA1SALTPASS — only the
    # template_finalize byte order differs.
    AlgoSpec(
        name="sha1passsalt",
        job_enum="JOB_SHA1PASSSALT",
        template_enum_value=39,
        base_algo="sha1",
        hash_words=5,
        hash_block_bytes=64,
        salt_position=SaltPosition.APPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=5,
        hashcat_mode="100",
        cpu_reference="mdxfind.c JOB_SHA1PASSSALT at lines 14227-14270",
        one_liner=("SHA1PASSSALT (JOB_SHA1PASSSALT, hashcat -m 100): "
                   "SHA1(pass || salt) — simple APPEND-salt SHA1"),
        iter_note=(
            "lowercase hex of prior 40-char digest re-SHA1'd with FRESH "
            "IV (no salt re-application). Lines 14262-14265: "
            "prmd5(curin.h, newbuf, 40); mysha1(newbuf, 40, ...). "
            "Identical iter shape to SHA1SALTPASS — only the salt "
            "POSITION differs at template_finalize time (APPEND vs "
            "PREPEND)."),
    ),
    AlgoSpec(
        name="sha256saltpass",
        job_enum="JOB_SHA256SALTPASS",
        template_enum_value=36,
        base_algo="sha256",
        hash_words=8,
        hash_block_bytes=64,
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=8,
        hashcat_mode="1410",
        cpu_reference="mdxfind.c JOB_SHA256SALTPASS at lines 27603-27651",
        one_liner=("SHA256SALTPASS (JOB_SHA256SALTPASS, hashcat -m 1410): "
                   "SHA256(salt || pass) — simple PREPEND-salt SHA256"),
        iter_note=(
            "lowercase hex of prior 64-char digest re-SHA256'd with FRESH "
            "IV (no salt re-application). Lines 27640-27642: "
            "prmd5(curin.h, newbuf, 64); mysha256(newbuf, 64, ...). "
            "Identical iter shape to SHA1SALTPASS but with SHA256 + "
            "64-char hex (vs 40-char for SHA1) — the 64-char output "
            "exactly fills one 64-byte block, so the pad marker lands "
            "in a SECOND block (M[0]=0x80, M[15]=512)."),
    ),
    # B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family APPEND-
    # shape salted variant. SHA256(pass || salt). Same SHA256 compression +
    # 8-word BE state as SHA256SALTPASS but salt is at the END of the
    # message, not the beginning. Cache disambiguated from SHA256SALTPASS
    # via SALT_POSITION=APPEND in defines_str (vs PREPEND); same
    # BASE_ALGO=sha256 + HASH_WORDS=8 axes. Pure spec reuse — both the
    # main template (sha256_style_salted.cl.tmpl from B6.2) and the
    # finalize fragment (finalize_append_be.cl.frag from B6.5) are
    # already shipped; codegen.py routing extended in B6.7 to allow
    # SHA256 + APPEND. 64-char hex iter feedback identical to
    # SHA256SALTPASS — only the template_finalize byte order differs.
    # B6.9 SHA512 fan-out (2026-05-06): first 64-bit-state salted variant
    # on the codegen path. SHA512(salt || pass) (simple PREPEND) — hashcat
    # -m 1710. Authors a new sibling main template (sha512_style_salted.cl.tmpl)
    # AND a new sibling fragment (finalize_prepend_be64.cl.frag) because
    # the SHA-512 family differs from SHA-1/SHA-256 on three width-bearing
    # axes: state width (64-bit ulong vs 32-bit uint), block size (128 vs
    # 64), and length field width (128-bit vs 64-bit). Per the codegen-
    # reconsideration memo's "splits over parameterization" + "width-
    # bearing constants belong in templates" rules, threading these axes
    # into the SHA-256 template/fragment with #if blocks would be more
    # fragile than two-file authoring. After SHA512SALTPASS lands, future
    # SHA512PASSSALT (-m 1710 / e386) is pure spec reuse — APPEND fragment
    # for the 64-bit family is the only remaining authoring (or extend
    # the SaltPosition.APPEND branch to detect base_algo=sha512 and emit
    # a finalize_append_be64.cl.frag at that point).
    #
    # R2 risk on gfx1201: unsalted SHA-512 reading was 42,520 B priv_mem
    # — only 504 B headroom under the 43,024 B 3080 spill-region ceiling.
    # The salted finalize body is structurally identical-cost (same M[16]
    # scratch, same per-byte loop) plus a salt-sourced byte fetch that
    # adds maybe one VGPR for the index variable. Expected reading
    # 42,500-42,800 B. HARD GATE: if priv_mem > 43,024 B on gfx1201, the
    # agent reports the reading + structural mitigations rather than
    # continuing past the priv_mem probe.
    AlgoSpec(
        name="sha512saltpass",
        job_enum="JOB_SHA512SALTPASS",
        template_enum_value=44,
        base_algo="sha512",
        hash_words=16,           # 16 LE-byteswapped uint32 emit/probe words (= 8 BE ulong state)
        hash_block_bytes=128,    # SHA-512 = 1024-bit block (vs 512 for MD5/SHA-1/SHA-2-32)
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=16,
        hashcat_mode="1710",
        cpu_reference="mdxfind.c JOB_SHA512SALTPASS at lines 13981-14023",
        one_liner=("SHA512SALTPASS (JOB_SHA512SALTPASS, hashcat -m 1710): "
                   "SHA512(salt || pass) — simple PREPEND-salt SHA-512"),
        iter_note=(
            "lowercase hex of prior 128-char digest re-SHA512'd with FRESH "
            "IV (no salt re-application). Lines 14015-14017: "
            "prmd5(curin.h, newbuf, 128); mysha512(newbuf, 128, ...). "
            "128-char hex output exactly fills one 128-byte block, so "
            "the 0x80 pad marker lands in a SECOND block "
            "(M[0]=0x8000000000000000UL BE, M[15]=128*8=1024)."),
    ),
    AlgoSpec(
        name="sha256passsalt",
        job_enum="JOB_SHA256PASSSALT",
        template_enum_value=43,
        base_algo="sha256",
        hash_words=8,
        hash_block_bytes=64,
        salt_position=SaltPosition.APPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=8,
        hashcat_mode="1410",
        cpu_reference="mdxfind.c JOB_SHA256PASSSALT at lines 27639-27677",
        one_liner=("SHA256PASSSALT (JOB_SHA256PASSSALT, hashcat -m 1410): "
                   "SHA256(pass || salt) — simple APPEND-salt SHA256"),
        iter_note=(
            "lowercase hex of prior 64-char digest re-SHA256'd with FRESH "
            "IV (no salt re-application). Lines 27670-27671: "
            "prmd5(curin.h, newbuf, 64); mysha256(newbuf, 64, ...). "
            "Identical iter shape to SHA256SALTPASS — only the salt "
            "POSITION differs at template_finalize time (APPEND vs "
            "PREPEND). 64-char hex output exactly fills one block, so "
            "the pad marker lands in a SECOND block (M[0]=0x80, "
            "M[15]=512)."),
    ),
    # B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512(pass || salt).
    # FINAL B6 ladder step (after this lands, B8 slab retirement opens
    # up). APPEND-shape sibling of B6.9's SHA512SALTPASS — same 64-bit-
    # state SHA-512 family, same 128-byte block, same 128-bit BE length
    # field; only the salt POSITION at template_finalize differs (APPEND
    # vs PREPEND). Cache disambiguated from SHA512SALTPASS via
    # SALT_POSITION=APPEND in defines_str (vs PREPEND); same BASE_ALGO=
    # sha512 + HASH_WORDS=16 + HASH_BLOCK_BYTES=128 axes — single-axis
    # delta. Pure spec reuse on the SHA-512 template (sha512_style_-
    # salted.cl.tmpl, salt-position-agnostic per B6.9), plus ONE new
    # fragment authoring (finalize_append_be64.cl.frag — sibling of
    # finalize_prepend_be64.cl.frag, mirrors the byte-source ordering
    # change of finalize_append_be.cl.frag vs finalize_prepend_be.cl.frag
    # but at 64-bit width).
    #
    # 128-char hex iter feedback identical to SHA512SALTPASS — only the
    # template_finalize byte order differs at the kernel side. Mirrors
    # mdxfind.c JOB_SHA512PASSSALT (lines 14069-14127) which shares the
    # iter step with JOB_SHA512SALTPASS (both run prmd5(curin.h, newbuf,
    # 128); mysha512(newbuf, 128, ...) — no salt re-application).
    AlgoSpec(
        name="sha512passsalt",
        job_enum="JOB_SHA512PASSSALT",
        template_enum_value=45,
        base_algo="sha512",
        hash_words=16,           # 16 LE-byteswapped uint32 emit/probe words (= 8 BE ulong state)
        hash_block_bytes=128,    # SHA-512 = 1024-bit block (vs 512 for MD5/SHA-1/SHA-2-32)
        salt_position=SaltPosition.APPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=16,
        hashcat_mode="1720",
        cpu_reference="mdxfind.c JOB_SHA512PASSSALT at lines 14069-14127",
        one_liner=("SHA512PASSSALT (JOB_SHA512PASSSALT, hashcat -m 1720): "
                   "SHA512(pass || salt) — simple APPEND-salt SHA-512"),
        iter_note=(
            "lowercase hex of prior 128-char digest re-SHA512'd with FRESH "
            "IV (no salt re-application). Lines 14113-14122 (shared with "
            "JOB_SHA512SALTPASS): prmd5(curin.h, newbuf, 128); mysha512("
            "newbuf, 128, ...). Identical iter shape to SHA512SALTPASS — "
            "only the salt POSITION differs at template_finalize time "
            "(APPEND vs PREPEND). 128-char hex output exactly fills one "
            "128-byte block, so the 0x80 pad marker lands in a SECOND "
            "block (M[0]=0x8000000000000000UL BE, M[15]=128*8=1024)."),
    ),
    # Family E HMAC-SHA384 carrier (2026-05-08): SHA384(salt || pass) carrier
    # for HMAC-SHA384 dispatch. There is no JOB_SHA384SALTPASS algorithm in
    # mdxfind; this AlgoSpec exists ONLY to author gpu_sha384saltpass_core.cl
    # (HASH_WORDS=12, EMIT_HIT_12, SHA-384 IV, SHA-384 96-char hex iter)
    # which carries the HMAC-SHA384 (e543) and HMAC-SHA384_KPASS (e796) HMAC
    # body via the shared finalize_prepend_be64.cl.frag (HASH_WORDS == 12 &&
    # algo_mode >= 5u gate). The mode-0 SHA384(salt||pass) main body is
    # structurally unreachable in production — host always sets algo_mode 5
    # or 6 for HMAC dispatch. The template_enum_value 46 is for cache key
    # disambiguation; no GPU_TEMPLATE_SHA384SALTPASS host enum is required
    # (the resolver dispatches HMAC-SHA384 via the kernel handle directly).
    #
    # KEY DELTA FROM sha512saltpass: HASH_WORDS=12 (vs 16) → EMIT_HIT_12 (vs
    # EMIT_HIT_16) → emits 6 ulong = 48 bytes (vs 8 ulong = 64 bytes). The
    # SHA-384 IV differs from SHA-512 IV (FIPS 180-4 §5.3.4 vs §5.3.5).
    # Iter loop hex feedback: 96 chars = 12 ulong (vs SHA-512's 128 chars =
    # 16 ulong) — fits in ONE block (vs SHA-512's two-block iter).
    AlgoSpec(
        name="sha384saltpass",
        job_enum="JOB_HMAC_SHA384",     # carrier: real op routed via algo_mode
        template_enum_value=46,
        base_algo="sha512",             # uses sha512_block compression
        hash_words=12,                  # 12 LE-byteswapped uint32 emit/probe words (= 6 BE ulong state)
        hash_block_bytes=128,           # SHA-384 = 1024-bit block (same as SHA-512)
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.BE_BSWAP,
        emit_width=12,
        hashcat_mode="(carrier; HMAC-SHA384 = -m 11800)",
        cpu_reference="N/A (no JOB_SHA384SALTPASS in mdxfind; carrier for HMAC-SHA384 e543/e796)",
        one_liner=("SHA384SALTPASS carrier (Family E, HMAC-SHA384 e543 / "
                   "e796 KPASS): SHA384 carrier kernel for HMAC dispatch via "
                   "algo_mode 5/6 — no production SHA384SALTPASS algorithm"),
        iter_note=(
            "lowercase hex of prior 96-char digest re-SHA384'd with FRESH "
            "IV (no salt re-application). 96 hex chars + 0x80 + 16 length = "
            "113 bytes — fits in ONE 128-byte block (vs SHA-512's two-block "
            "iter). Dead under HMAC dispatch (max_iter=1 forced host-side). "
            "Kept for symmetry with SHA-512 sibling and for future SHA384-"
            "SALTPASS support if/when added to mdxfind."),
    ),
    # Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD-160(salt || pass)
    # carrier for HMAC-RIPEMD-160 dispatch. There is no JOB_RIPEMD160SALTPASS
    # algorithm in mdxfind; this AlgoSpec exists ONLY to author gpu_-
    # ripemd160saltpass_core.cl (HASH_WORDS=5, EMIT_HIT_5, RIPEMD-160 IV,
    # RIPEMD-160 40-char hex iter) which carries the HMAC-RIPEMD-160 (e211)
    # and HMAC-RIPEMD-160_KPASS (e798) HMAC body via the new sibling fragment
    # finalize_prepend_rmd.cl.frag (HASH_WORDS == 5 && algo_mode >= 5u gate).
    # The mode-0 RMD160(salt||pass) main body is structurally unreachable in
    # production — host always sets algo_mode 5 or 6 for HMAC dispatch.
    # The template_enum_value 48 is for cache key disambiguation; no
    # GPU_TEMPLATE_RIPEMD160SALTPASS host enum is required by the resolver
    # (the resolver dispatches HMAC-RMD160 via the kernel handle directly).
    #
    # KEY DELTA FROM sha1saltpass (the closest structural relative):
    # base_algo="ripemd160" (not "sha1") - distinct compression primitive
    # routed to a new sibling main template (rmd160_style_salted.cl.tmpl,
    # B5 sub-batch 2 of unsalted core's compression style) and a new
    # sibling fragment (finalize_prepend_rmd.cl.frag, parametric for the
    # RMD family). LE-direct probe + emit (no bswap32) vs SHA1's BE-bswap.
    # Future HMAC-RMD320 (Family H) reuses the same fragment with a
    # HASH_WORDS == 10 branch addition.
    AlgoSpec(
        name="ripemd160saltpass",
        job_enum="JOB_HMAC_RMD160",     # carrier: real op routed via algo_mode
        template_enum_value=48,
        base_algo="rmd160",             # uses rmd160_block compression (2-arg) — primitive
                                        # name in gpu_common.cl rev 1.x line 960
        hash_words=5,                   # 5 LE uint32 state words (= 20-byte digest)
        hash_block_bytes=64,            # RMD160 = 512-bit block (same as MD5/SHA-1)
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.LE_DIRECT,
        emit_width=5,
        hashcat_mode="(carrier; HMAC-RIPEMD-160 = -m 6060/6050)",
        cpu_reference="N/A (no JOB_RIPEMD160SALTPASS in mdxfind; carrier for HMAC-RMD160 e211/e798)",
        one_liner=("RIPEMD160SALTPASS carrier (Family G, HMAC-RMD160 e211 / "
                   "e798 KPASS): RMD160 carrier kernel for HMAC dispatch via "
                   "algo_mode 5/6 — no production RIPEMD160SALTPASS algorithm"),
        iter_note=(
            "lowercase hex of prior 40-char digest re-RMD160'd with FRESH "
            "IV (no salt re-application). 40 hex chars + 0x80 + 8 length = "
            "49 bytes — fits in ONE 64-byte block (mirrors SHA-1's iter "
            "shape but LE byte ordering). Dead under HMAC dispatch "
            "(max_iter=1 forced host-side). Kept for symmetry with the "
            "unsalted RMD160 core and for future RIPEMD160SALTPASS support "
            "if/when added to mdxfind."),
    ),
    # Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD-320(salt || pass)
    # carrier for HMAC-RIPEMD-320 dispatch. There is no JOB_RIPEMD320SALTPASS
    # algorithm in mdxfind; this AlgoSpec exists ONLY to author gpu_-
    # ripemd320saltpass_core.cl (HASH_WORDS=10, EMIT_HIT_10, RIPEMD-320 IV,
    # RIPEMD-320 80-char two-block hex iter) which carries the HMAC-
    # RIPEMD-320 (e213) and HMAC-RIPEMD-320_KPASS (e799) HMAC body via the
    # shared finalize_prepend_rmd.cl.frag (HASH_WORDS == 10 && algo_mode
    # >= 5u gate). The mode-0 RMD320(salt||pass) main body is structurally
    # unreachable in production — host always sets algo_mode 5 or 6 for
    # HMAC dispatch. The template_enum_value 49 is for cache-key
    # disambiguation; no GPU_TEMPLATE_RIPEMD320SALTPASS host enum is
    # required by the resolver (the resolver dispatches HMAC-RMD320 via
    # the kernel handle directly).
    #
    # KEY DELTA FROM ripemd160saltpass (closest structural relative — both
    # use the LE 32-bit RMD family, 2-arg block call, LE-direct probe/emit,
    # PREPEND salt, single shared fragment via HASH_WORDS gate):
    # base_algo="rmd320" (vs "rmd160") - distinct compression primitive
    # routed to the SIBLING main template (rmd320_style_salted.cl.tmpl)
    # because the iter loop differs (80 hex chars span two 64-byte blocks
    # vs 40 hex chars fitting one block). The shared fragment fans out
    # via the HASH_WORDS axis (5 vs 10) — same algo_mode 5/6 encoding,
    # same M[] LE byte-pack, different IV constants (10-word vs 5-word),
    # different outer-block geometry (0x80 at byte 40 / length 832 bits
    # vs byte 20 / length 672 bits). EMIT_HIT_10 emits 10 LE uint32 = 40
    # bytes (matches HMAC-RMD320's 40-byte digest).
    AlgoSpec(
        name="ripemd320saltpass",
        job_enum="JOB_HMAC_RMD320",     # carrier: real op routed via algo_mode
        template_enum_value=49,
        base_algo="rmd320",             # uses rmd320_block compression (2-arg) — primitive
                                        # name in gpu_common.cl rev 1.12 line 1023
        hash_words=10,                  # 10 LE uint32 state words (= 40-byte digest)
        hash_block_bytes=64,            # RMD320 = 512-bit block (same as RMD160 / MD5 / SHA-1)
        salt_position=SaltPosition.PREPEND,
        iter_shape=IterShape.HEX_FEEDBACK,
        digest_endianness=DigestEndian.LE_DIRECT,
        emit_width=10,
        hashcat_mode="(carrier; HMAC-RIPEMD-320 has no hashcat -m equivalent)",
        cpu_reference="N/A (no JOB_RIPEMD320SALTPASS in mdxfind; carrier for HMAC-RMD320 e213/e799)",
        one_liner=("RIPEMD320SALTPASS carrier (Family H, HMAC-RMD320 e213 / "
                   "e799 KPASS): RMD320 carrier kernel for HMAC dispatch via "
                   "algo_mode 5/6 — no production RIPEMD320SALTPASS algorithm"),
        iter_note=(
            "lowercase hex of prior 80-char digest re-RMD320'd with FRESH "
            "IV (no salt re-application). 80 hex chars + 0x80 + 8 length = "
            "89 bytes — REQUIRES two 64-byte blocks (block 1 holds 64 hex "
            "chars from state[0..7]; block 2 holds 16 hex chars from "
            "state[8..9] + 0x80 at M[4] + length 640 in M[14]). Dead under "
            "HMAC dispatch (max_iter=1 forced host-side). Kept for symmetry "
            "with the unsalted RMD320 core (gpu_ripemd320_core.cl) and for "
            "future RIPEMD320SALTPASS support if/when added to mdxfind."),
    ),
]


def by_name(name: str) -> AlgoSpec:
    for s in ALGOS:
        if s.name == name:
            return s
    raise KeyError("unknown spec: %s" % name)
