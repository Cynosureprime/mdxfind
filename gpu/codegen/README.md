# gpu/codegen — per-algorithm `_core.cl` source generator + OpenCL-to-Metal translator

Two Python tools live here:

  1. `codegen.py` — emits `gpu_<name>_core.cl` extension files from a
     per-algorithm spec table (B6 salted algos only).

  2. `cl2metal.py` — translates `gpu/gpu_<algo>_core.cl` -> Metal core
     (Phase 2d.1+, per `project_metal_phase2d_arch.md`).

The `_core.cl` generator is documented immediately below; the Metal
translator is in the **OpenCL-to-Metal translator** section further down.

A small Python tool that emits `gpu_<name>_core.cl` extension files from
a per-algorithm spec table. Per the codegen-reconsideration memo of
2026-05-06 (option d hybrid), this is **only used for B6 salted algos**.
The 32 unsalted cores stay hand-written; this tool does not regenerate
them and `--check` skips files for which there is no shipped reference.

## What this tool is

- An emitter for the per-algorithm hooks the generic `gpu_template.cl`
  expects (`template_state`, `template_init`, `template_transform`,
  `template_finalize`, `template_iterate`, `template_digest_compare`,
  `template_emit_hit*`).
- Driven by `specs.py` (a list of `AlgoSpec` dataclasses).
- Templates live under `templates/` as plain `.cl` text with `{{...}}`
  placeholders. Hand-rolled `str.replace()` substitution; no third-party
  dependencies.

## What this tool is NOT

- It does not touch host wiring (`gpu_opencl.c`, `template_id_for_job()`,
  `gpu_gpujob_opencl.c`). All ~6 host wiring sites per algo are still
  manual.
- It does not regenerate the 32 hand-written unsalted `_core.cl` files.
- It does not emit `_str.h` blobs — that is `cl2str.py`'s job. Run it
  after generation.
- It is not a Metal generator (Phase 3 of the memo, not in scope here).

## Files

| File | Purpose |
|---|---|
| `codegen.py` | Main driver. CLI: `--all`, `--algo NAME`, `--check`, `-o DIR`. |
| `specs.py` | Per-algorithm `AlgoSpec` dataclass + the `ALGOS` list. |
| `templates/md_style_salted.cl.tmpl` | Top-level template for MD-family salted cores. |
| `templates/finalize_prepend.cl.frag` | `template_finalize` body for `MD5(salt \|\| pass)`. |
| `templates/finalize_append_to_hex32.cl.frag` | `template_finalize` body for `MD5(hex32(MD5(pass)) \|\| salt)` (double-MD5 chain). |

## Usage

### Generate all specs

```
python3 codegen.py --all -o /tmp/codegen_out
```

### Generate one spec

```
python3 codegen.py --algo md5salt -o /tmp/codegen_out
```

### Check specs against shipped cores

`--check` regenerates each spec in memory and runs a comment-and-
whitespace-stripped diff against the corresponding `gpu/<name>_core.cl`.
This is a drift detector: if the shipped core is hand-edited away from
the templates, `--check` will print a unified diff and exit nonzero.

```
python3 codegen.py --check --all
```

The `--check` semantic-strip is intentionally crude (no full C parser).
It removes `/* ... */` and `// ...` comments and normalizes whitespace.
Intentional algorithmic changes still show as a diff. Comment-only or
whitespace-only differences do not.

The check is a **forward investment** — once new specs ship, future
agents can run `--check` periodically to confirm nobody quietly
hand-tuned a generated core out of sync with its spec.

## Workflow: add a new salted algo (example: SHA1SALT)

1. **Read the CPU reference** in `mdxfind.c` for the target `JOB_*` enum.
   Identify the salt position (`PREPEND` vs `APPEND` vs the rare
   `APPEND_TO_HEX*`), the digest geometry, the iter shape.

2. **Add an `AlgoSpec(...)` entry** to `specs.py`. Example for SHA1SALT
   (hashcat -m 110, `MD5(salt || pass)`-shaped but with SHA1):

   ```python
   AlgoSpec(
       name="sha1salt",
       job_enum="JOB_SHA1SALT",
       template_enum_value=35,            # next free GPU_TEMPLATE_*
       base_algo="sha1",
       hash_words=5,                      # SHA1: 5 uint32 digest
       hash_block_bytes=64,
       salt_position=SaltPosition.APPEND, # MD5(pass || salt) shape
       iter_shape=IterShape.HEX_FEEDBACK,
       digest_endianness=DigestEndian.BE_BSWAP,  # SHA1 is BE
       emit_width=5,
       hashcat_mode="110",
       cpu_reference="mdxfind.c JOB_SHA1SALT at lines NNNN-MMMM",
       one_liner="SHA1SALT (JOB_SHA1SALT, hashcat -m 110): SHA1(pass || salt)",
       iter_note="(SHA1 hex feedback, fresh IV).",
   ),
   ```

3. **Pick or author a template.** If the salt shape and base algo match
   an existing template/fragment combination, no template work is
   needed. If you are introducing the first SHA1-family or
   APPEND-family algo, you must:
   - Add a `select_main_template()` branch in `codegen.py` for the new
     `(base_algo, salt_position)` pair (or extend the SHA-family scope
     of the existing template), AND
   - Add a `select_finalize_fragment()` branch for any new salt-position
     variant (the `APPEND` fragment is currently a stub — first author
     to need it should add `templates/finalize_append.cl.frag`).

4. **Generate the core file.**

   ```
   python3 codegen.py --algo sha1salt -o /tmp/codegen_out
   ```

5. **Diff against an oracle.** If you have a hand-built reference for
   visual review, pass it as `gpu/gpu_sha1salt_core.cl` and run
   `--check`. Otherwise inspect the generated file directly — line
   counts in the 200-300 range are normal.

6. **Promote to the working tree.** When satisfied:

   ```
   cp /tmp/codegen_out/gpu_sha1salt_core.cl /Users/dlr/src/mdfind/gpu/
   python3 /Users/dlr/src/mdfind/gpu/cl2str.py /Users/dlr/src/mdfind/gpu/gpu_sha1salt_core.cl
   ```

   This produces `gpu_sha1salt_core_str.h` next to it.

7. **Wire the host (manual).** Per the Memo B B6 wiring checklist
   (`project_memo_b_dispatch_template.md`):
   - `gpu_opencl.c`: add the source-string concat case in
     `compile_template_program()`.
   - `gpu_opencl.c`: add the `template_id_for_job()` enum branch.
   - `gpu_opencl.c`: add the salt-axis feature flag if needed.
   - `gpu_opencl.h`: bump the `GPU_TEMPLATE_<NAME>` enum.
   - `mdxfind.c`: route the `JOB_*` case to the template path.
   - `gpu/RCS/`: `cp file /tmp/file.pre-checkin-$(date +%s); ci -l file`.
     Both `.cl` AND `_str.h` AND `gpu_opencl.c/h` get `ci -l`'d.

8. **Build + validate** on .205 via `mdx-build`, run the salted
   validation matrix on ioblade (24 cells: algo × iter ∈ {1, 3} × GPU ∈
   {gfx1201, RTX 4070 Ti, RTX 3080} × salt-page ∈ {single, multi}). PASS
   = 24/24 byte-exact match against CPU oracle.

## Validation log: MD5SALT + MD5SALTPASS regen against shipped cores

Run on 2026-05-06 immediately after the cores shipped (rev 1.1 each):

```
$ python3 codegen.py --check --all
[md5salt] check OK (semantic match)
[md5saltpass] check OK (semantic match)
```

Comment-and-whitespace-stripped output of generated and shipped is
**byte-equal** for both cores (4487 bytes md5salt, 3487 bytes
md5saltpass). The raw `diff` is comment-only (the shipped cores carry
extra historical commentary about the slab / hex32 / two-block paths
that the templates do not need to repeat per-algo).

## Adding a new template

Templates use `{{KEY}}` placeholder syntax. Keys recognized today:

| Key | Source | Example |
|---|---|---|
| `{{NAME}}` | `spec.name` | `md5salt` |
| `{{ONE_LINER}}` | `spec.one_liner` | `MD5SALT (...): MD5(hex32(MD5(pass)) \|\| salt) — DOUBLE-MD5 chain` |
| `{{CPU_REFERENCE}}` | `spec.cpu_reference` | `mdxfind.c JOB_MD5SALT at lines 21943-21974` |
| `{{SALT_POSITION_TOKEN}}` | `spec.salt_position_token` | `APPEND_TO_HEX32` |
| `{{ITER_SHAPE}}` | `spec.iter_shape.value` | `HEX_FEEDBACK` |
| `{{ITER_NOTE}}` | `spec.iter_note` | algorithm-specific commentary |
| `{{BASE_ALGO}}` | `spec.base_algo` | `md5`, `sha1`, ... |
| `{{HASH_WORDS}}` | `spec.hash_words` | `4`, `5`, `8` |
| `{{HASH_BLOCK_BYTES}}` | `spec.hash_block_bytes` | `64`, `128` |
| `{{TEMPLATE_FINALIZE_BODY}}` | finalize fragment file | (full function body) |

Add a placeholder by editing `_placeholders()` in `codegen.py`. Keep
the dict tiny — if a template needs more than ~3 new placeholder
regions, prefer authoring a separate template/fragment over
parameterizing further.

## Constraints carried from the design memo

- Python 3, no third-party deps (no jinja2).
- The `.cl` files must pass `cl2str.py` without modification.
- `codegen.py` does **not** overwrite shipped `gpu_*_core.cl` files
  (always specify `-o` to a side directory; the workflow is "generate
  to /tmp, copy in").
- `codegen.py` does **not** modify host wiring sites — emits `.cl` only.
- Templates stay simple. Splits over parameterization.

---

# OpenCL-to-Metal translator (`cl2metal.py`)

A hybrid translator that ports `gpu/gpu_<algo>_core.cl` to a Metal kernel
core via three passes: tokenize-and-substitute, structural rewriter, and
per-kernel YAML overlay. See `project_metal_phase2d_arch.md` for the
architecture memo. Phase 2d.1 (initial ship): translator + regression
test + overlays for md5 and md5salt.

## What this tool is

- A one-shot human-initiated translator: invoked by the developer; the
  output `*.metal.generated` is reviewed and (in Phase 2d.2+) promoted to
  `gpu/metal_<algo>_core.metal`.
- Three-pass design per memo §3:
  1. Tokenize-and-substitute -- `__global -> device`, `__private ->
     thread`, `barrier(CLK_LOCAL_MEM_FENCE) ->
     threadgroup_barrier(mem_flags::mem_threadgroup)`, etc.
  2. Structural rewriter -- bracket-balanced parsing for function signatures
     and struct decls. Rewrites `typedef struct {} T;` -> `struct T {};`,
     `T *st` -> `thread T &st` (state-by-ref), `st->h[0]` -> `st.h[0]`
     within fn bodies, and `(st)->h[0]` -> `(st).h[0]` within macros.
  3. Per-kernel overlay -- reads `cl2metal_overrides/<algo>.yaml` for
     Apple-specific tweaks (skip functions/structs, line-range skips,
     per-arg address-space override).

## What this tool is NOT

- Not a bidirectional translator (Metal -> OpenCL is out of scope).
- Not a build-time codegen rule -- the developer runs it manually, reviews
  the diff, and decides whether to ship.
- Not a host-wiring generator. New algorithms still need ~6 host-wiring
  sites per `feedback_codegen_host_wiring_gaps.md` and
  `feedback_architect_host_wiring_reflex.md`.
- Not a replacement for hand-written Metal cores. Apple-specific tweaks
  (e.g., task #250 device-buf migration, Phase 2e pre-salt hoist,
  algo_mode pruning) belong in the overlay or in hand-curated cores.

## Files

| File | Purpose |
|---|---|
| `cl2metal.py` | Main translator (~900 LOC). |
| `cl2metal_overrides/<algo>.yaml` | Per-algo overlay (Apple tweaks). |
| `tests/test_cl2metal.py` | Regression harness; runs md5 + md5salt + xcrun metal compile. |

## Usage

```sh
# Default: write to <input>.metal.generated next to the source.
python3 gpu/codegen/cl2metal.py gpu/gpu_md5_core.cl

# Explicit output path; diff against an existing hand-port:
python3 gpu/codegen/cl2metal.py gpu/gpu_md5_core.cl \
    -o /tmp/test_md5.metal \
    --diff gpu/metal_md5_core.metal

# Specify a non-default overlay:
python3 gpu/codegen/cl2metal.py gpu/gpu_<algo>_core.cl \
    --overlay gpu/codegen/cl2metal_overrides/<algo>.yaml

# Run forbidden-token lint after translation:
python3 gpu/codegen/cl2metal.py gpu/gpu_md5_core.cl --check
```

## Overlay format (`cl2metal_overrides/<algo>.yaml`)

```yaml
algo: md5salt

# Drop entire static-inline function definitions or struct decls.
skip_functions:
  - salt_pack_uint
  - some_unused_helper

# Drop arbitrary line ranges from the SOURCE (line numbers refer to the
# original .cl file). Useful for removing mode-switch branches or
# legacy code blocks that don't translate cleanly.
skip_line_ranges:
  - start: 200
    end:   250
    reason: "HMAC modes -- deferred to Phase 2d+"

# Override the address-space inference for specific function args.
# Default rule: unqualified pointer becomes `thread` (or `thread const`
# if `const`-qualified in the source). Override to `device` /
# `device const` etc. when an Apple-specific tweak applies.
arg_address_space:
  template_finalize:
    data: "device const"
    salt_buf: "device const"
```

## Regression harness

`tests/test_cl2metal.py` invokes the translator on `gpu/gpu_md5_core.cl`
and `gpu/gpu_md5salt_core.cl`, then validates:

  - **LOC delta vs hand-port** (loose: comments dominate)
  - **Semantic delta vs hand-port** (comment-stripped; tight for md5,
    loose for md5salt because the hand-port is a 38%-smaller reimpl)
  - **Fidelity delta vs OpenCL source** (translator faithfulness: tight)
  - **Forbidden-token lint**: no `__global`, `__private`, `barrier(CLK_*)`,
    `__kernel`, `typedef struct`, `as_uint(`, `mul_hi(` surviving.
  - **State-array bounds lint**: `h[N]` accesses where N >= HASH_WORDS
    are flagged (per memo §6 closing paragraph).
  - **xcrun metal compile**: TU `metal_common.metal + <generated>` compiles
    cleanly.

```sh
python3 gpu/codegen/tests/test_cl2metal.py
```

Exit codes: 0 PASS, 1 gate failure, 2 harness setup error.

## Phase 2d.1 known limit: md5salt fidelity vs hand-port

The Metal hand-port `gpu/metal_md5salt_core.metal` (RCS 1.1) is a
**38%-smaller reimplementation** of `gpu/gpu_md5salt_core.cl` (~492 LOC vs
796), not a pure translation:

  - `salt_pack_uint` helper elided (replaced inline byte loops).
  - `template_finalize` modes 1-6 (HMAC + variant-mode branches) elided.
  - `template_emit_hit` (simple) macro elided (only the dedup variant ships).

For Phase 2d.1, the translator:

  - **PORTS the full OpenCL source** including modes 1-6 and salt_pack_uint
    (fidelity vs OpenCL source: 0.98%, well under 5% budget).
  - **Diverges from the hand-port by ~70% semantic content** (this delta
    is the hand-port's deliberate content pruning, not a translator gap).

A future Phase 2d.2+ overlay primitive (`replace_function_body`,
`replace_calls`, or `code_block_drop_with_decls`) could express the
hand-port's structural pruning cleanly. For Phase 2d.1, the translator
ships as a **mechanical translator only**; the md5salt hand-port stays in
place as the production file, and the translator's md5salt output is
exercised for the compile-and-lint gates only.

## Constraints carried from the architect memo

- Python 3 stdlib only (PyYAML used when available; falls back to a
  minimal hand-written YAML reader otherwise).
- The translator does **not** modify host wiring (`gpu_metal.m`,
  `mdxfind.c`, etc.) -- emits `.metal` only.
- The translator does **not** auto-fan-out to Phase 2d.2+ kernels;
  each family is gated by a Phase 2d.N memo + manual decision.
- Per memo §6 closing paragraph: the translator NEVER substitutes
  IV/HASH_WORDS constants; widths are verbatim from source.

