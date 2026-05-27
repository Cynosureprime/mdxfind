# mdxfind v1.503 — Mask iteration scale fixes + hx.8 catalog doc corrections

Source: mdxfind.c rev 1.505, gpu/gpu_opencl.c rev 1.195, gpu_metal.m rev 1.118, hx.8 rev 1.12, hashpipe.c rev 1.90.

Window: 2026-05-26.

## Mask iterator scale fixes

The mask iterator's progress counter and per-thread loop bound were both 32-bit signed. With keyspaces commonly exceeding 2^31 (e.g., `?d?d?d?d?d?d?d?d?d?d` = 10^10 ≈ 2.3 × 2^32), the iterator would silently wrap, exit early, or produce incorrect candidate counts. Both `number_iter` and `loop_bound` are now `uint64_t` end-to-end; full 10^10+ keyspaces traverse correctly.

Multi-thread mask attacks now actually use the requested thread count. The previous chunking math computed the per-thread range using the (overflow-prone) 32-bit counter, which on large keyspaces collapsed all work onto thread 0 while the other workers spun idle. The chunker has been rewritten over the widened 64-bit counter so each worker takes a proportional non-overlapping slice; `-T N` scales as expected on long mask runs.

## MAX_MASK_POS raised (CPU 16 → 256; GPU stays at 16 with explicit cap symbol)

`MAX_MASK_POS` (CPU-side per-side mask position cap) raised from 16 to 256. The GPU per-side cap is now an explicit, separately named symbol `MAX_MASK_POS_GPU_SIDE = 16` documenting what the bundled GPU kernels actually support (the kernels' unrolled per-position state still hard-codes 16, and raising that requires a coordinated multi-file kernel edit — see Known issues below).

Previously, masks longer than 16 positions on either side were silently truncated by the parser, producing a much smaller keyspace than the user asked for with no diagnostic. The parser now emits a **FATAL** message naming the offending mask, its position count, and the active per-side cap, then exits non-zero.

GPU upload sites in both OpenCL (`gpu/gpu_opencl.c`) and Metal (`gpu_metal.m`) gained a runtime guard: when a mask attack on a GPU target exceeds `MAX_MASK_POS_GPU_SIDE`, the run is gracefully demoted to CPU with a one-line warning naming the mask and the per-side cap. No crash, no silent truncation.

## hx.8 catalog doc corrections

Three entries in the hx algorithm catalog (manual page `hx(8)`) carried stale expressions that didn't match what mdxfind actually computes. Corrected this release:

- `e232 MD5BASE64` — was `base64(md5_bin(pass))`, now `md5(base64(pass))`
- `e539 MYSQL5MD5` — was `"*" . upper(sha1(sha1_bin(md5(pass))))`, now `sha1(sha1_bin(md5(pass)))`
- `e993 WPBCRYPT` — was `bcrypt(pass, salt, 10)`, now `bcrypt(base64(hmac_sha384_bin(pass, "wp-sha384")), salt, N)` (matches hashcat m35500)

No compute changes — only the documented expressions were drifted. CPU and GPU dispatch were always computing the corrected forms above; the manual now reflects reality.

The canonical hx language manual is published at <https://www.mdxfind.com/hx.pdf>. The hx parser and algorithm catalog source live in the upstream hashpipe project: <https://github.com/Cynosureprime/hashpipe>.

## Known issues / scope

- The GPU per-side mask cap stays at 16 this release. Raising it requires coordinated edits across eight kernel sources, their header companions, the `cl2str.py` / `cl2metal.py` string-literal serializers, and the host upload enums. The work is straightforward but mechanical, and not blocking; deferred to a future release with a real driver. Customers needing mask attacks with more than 16 positions per side run on CPU — the parser and iterator now support up to 256 CPU-side positions.
- `hashpipe.c` ships with no net source-functionality change vs v1.502; the file's revision counter advanced because intermediate experimental work was added and then reverted. Behavior is identical to v1.502 hashpipe.

# mdxfind v1.502 — Phase 4 + Phase 5a hx codegen (e347 + 7 MAKE_MD5PASS family GPU acceleration)

Source: mdxfind.c rev 1.501, gpu/gpu_opencl.c rev 1.194, gpu/gpu_opencl.h rev 1.41, gpu/gpujob_opencl.c rev 1.152, gpu_metal.m rev 1.117, gpu_metal.h rev 1.59, gpu/gpujob_metal.m rev 1.32, gpu/gpu_common.cl rev 1.25, gpu/metal_common.metal rev 1.24, gpu/gpu_codegen_eligible.{c,h} rev 1.1 (NEW), codegen/ tree (NEW: ~15 files, in-process hx P4 state-machine codegen), tools/hx8_to_c (NEW: build-time hx.8 → C-literal serializer).

Window: 2026-05-20 through 2026-05-23.

## Phase 4 — Production GPU dispatch for e347 (MD5(MD5(MD5(pass)).salt)) via codegen

A new in-process hx codegen pipeline produces JIT-compiled kernel B for `JOB_MD5MD5SALT` (e347) on both OpenCL (Pascal and newer) and Apple Metal. The codegen walker reads the `hx_program` bytecode for the algorithm, applies a pattern detector (`HX_PATTERN_E347_MD5MD5MD5SALT`) that recognizes the hand-tunable shape, and emits a specialized kernel source tuned to that shape (per-thread serial SALT_BATCH=64 inner loop, register-held pre-state, salt-axis amortization — the tp0 pattern that empirically wins on Pascal salted-MD5).

Cross-arch byte-exact validated against the CPU oracle on 1,048,576-pair fixtures (smoke / medium / large × 2 backends):

- Apple M2 Max: 0.25 s end-to-end on the large fixture; zero diff
- Pascal GTX 1080: 1.05 s end-to-end on the large fixture; zero diff

The hand-written `gpu_kernelb_md5md5salt_nocache.cl` (previously the e347 production path) is **retired**. The hand-written kernel carried a long-standing chain-drift bug for non-trivial salt cardinalities; codegen output is byte-exact against both the CPU implementation and hashpipe (independent reference). The legacy file has been deleted from the working tree.

`MDXFIND_HX_CODEGEN=0` is the (now-deprecated) opt-out env var. Setting it on a Phase-4+ build is **FATAL** with a deprecation message — the legacy hand-written code path is gone, there is nothing to fall back to.

## Phase 5a — MAKE_MD5PASS family GPU acceleration via codegen

Seven new algorithms GPU-accelerated by the same codegen pipeline, each computing `outer_hash(md5_hex(pass) . pass)`:

| eN  | Name           | Outer    | Digest width |
|----:|----------------|----------|-------------:|
| 122 | MD4MD5PASS     | MD4      |     16 bytes |
| 159 | RMD160MD5PASS  | RIPEMD160 |    20 bytes |
| 161 | SHA1MD5PASS    | SHA-1    |     20 bytes |
| 163 | SHA224MD5PASS  | SHA-224  |     28 bytes |
| 165 | SHA256MD5PASS  | SHA-256  |     32 bytes |
| 167 | SHA384MD5PASS  | SHA-384  |     48 bytes |
| 169 | SHA512MD5PASS  | SHA-512  |     64 bytes |

Codegen uses a per-primitive emit dispatch table (`codegen/hx_emit_primitives.c`, new in sub-phase 5a.2) so each family member shares the inner-hash + concat scaffolding and differs only in the outer-hash primitive selection.

Full 70-cell cross-arch validation matrix (7 algorithms × 5 fixtures × 2 backends) — all PASS, all byte-exact.

Usage: `./mdxfind -m e<N> -G 0 -F hashes -M <NAME> wordlist` selects GPU dispatch automatically for any of the 7 family members above. The `gpu_codegen_kernelb_family_md5pass_eligible()` admit-predicate helper (new file `gpu/gpu_codegen_eligible.{c,h}`) widens the chokepoint OR-chain to admit these JOBs.

Twenty-two additional MAKE_MD5PASS family members (MD2, GOST family, Haval ×15, RMD128, Tiger, Whirlpool, Snefru-256/512) remain CPU-only pending Phase 5b — their block primitives need lifting into `gpu/gpu_common.cl` first. e123 MD5MD5PASS (multi-emit canonical + colon variant) stays CPU-only until multi-emit codegen lands in a future sub-phase.

## Documentation

- The hx algorithm spec (Appendix A of the hx manual) audit and 32 doc-fix corrections: MAKE_MD5PASS family had missing concat operators in the canonical expressions; MD5MD5USER had user/pass argument transposition. Multi-emit families now annotated with Note [24] markers — 28 entries identifying mdxfind's multi-output emission patterns.
- The canonical hx language manual is published at <https://www.mdxfind.com/hx.pdf>. The hx parser + algorithm catalog source live in the upstream hashpipe project: <https://github.com/Cynosureprime/hashpipe>.

## Build / infrastructure

- New `codegen/` directory containing the in-process P4 state-machine codegen — `hx_walker.c` (state machine + bytecode dispatch), `hx_emit_opencl.c` + `hx_emit_metal.c` (per-backend emit helpers), `hx_patterns.c` (pattern detector for hand-tunable shapes), `hx_emit_primitives.c` (per-primitive outer-hash dispatch), `hx_dump.c` (env-flag source dump), and `hx_specs_data.c` (the compiled `hx_program` table, ~28 KLOC of generated C literals).
- New `tools/hx8_to_c.c` build-time tool that converts the hx algorithm catalog to `codegen/hx_specs_data.c`. Shipped here for completeness, but requires the upstream hashpipe source tree to compile (it links the hx parser library). External users build directly against the pre-generated `codegen/hx_specs_data.c` checked in here; that file is regenerated upstream when the hx catalog changes.
- New `gpu/gpu_codegen_eligible.{c,h}` — pure-C admit-predicate helper used by both OpenCL and Metal builds.
- New `hx_vm.h` and `hx_ast.h` — header-only types shared by codegen and the upstream hx VM (no implementation files; codegen consumes the compiled `hx_program` data).
- New kernel A variants under `gpu/` — `gpu_kernel_a_{rules,masks,rules_masks,bruteforce}.{cl,_str.h}` plus their Metal twins `metal_kernel_a_*.{metal,_str.h}` — hand-written rule / mask / brute-force producers from Phase 1a, used by the two-kernel pipeline for e347 and the family ops.
- `MDXFIND_HX_CODEGEN_DUMP=/tmp/x.cl` env var dumps the emitted codegen kernel source to the named path for post-mortem inspection.
- `MDXFIND_HX_CODEGEN_VALIDATE=1` + `MDXFIND_HX_CODEGEN_FIXTURE=<path>` env vars exercise the byte-exact validation harness against a fixture file (developer mode; exits 0 / 1 based on diff vs CPU oracle).

## Known issues / scope

- Codegen `kernelb_hx_codegen_phase0` ships the canonical e347 + 7 MAKE_MD5PASS family members. All other GPU ops continue to use the existing template-kernel infrastructure (unchanged from v1.485).
- e123 MD5MD5PASS remains CPU-only — multi-emit codegen is a future sub-phase.
- bcrypt, yescrypt, argon2, descrypt all remain hand-written kernels (out of codegen scope by design — they don't fit the hx expression model).
- The hand-port kernel A variants are unchanged from Phase 1a; Phase 1b "template kernel migration to codegen" is a low-priority follow-on (no perf or correctness motivation to rush it).
# mdxfind v1.475 — Metal coverage expansion + shared-loader refactor

Source: mdxfind.c rev 1.475, gpu_metal.m rev 1.100, gpu_metal.h rev 1.49, gpu/gpu_opencl.c rev 1.171, gpu/gpujob_metal.m rev 1.25, gpu/gpujob_opencl.c rev 1.138, gpu/codegen/cl2metal.py rev 1.9.

Window: 2026-05-16 through 2026-05-17.

## Metal coverage: 25 → 52 families

Apple Metal GPU acceleration extended from 25 to 52 algorithm families. New this release, grouped by phase:

- **Phase 2d.6** — RIPEMD: ripemd160 (e17), ripemd320 (e816)
- **Phase 2d.7a** — Blake2: blake2s256 (e844), blake2b256 (e845), blake2b512 (e841)
- **Phase 2d.7b** — Keccak + SHA-3: keccak{224,256,384,512} (e84-e87), sha3_{224,256,384,512} (e88-e91)
- **Phase 2d.7c** — Streebog: streebog256 (e430), streebog512 (e431)
- **Phase 2d.7d** — HMAC siblings: hmac_blake2s (e828), hmac_streebog256 (e837 KPASS / e838 KSALT), hmac_streebog512 (e839 KPASS / e840 KSALT)
- **Phase 2d.8a** — Iter-loop: phpbb3 (e455), md5crypt (e511)
- **Phase 2d.8b** — SHACRYPT: sha256crypt (e512), sha512crypt (e513), sha512cryptmd5 (e538)
- **Phase 2d.9a** — Feistel hand-port: descrypt (e500)
- **Phase 2d.9b** — Eksblowfish hand-port: bcrypt (e450)

All 52 families verified byte-exact CPU/Metal parity on Apple M1 and M2 Max. Admission summary: 52 families admitted, 208/208 variants admitted, 0 prunes, 0 CPU-only.

## Shared-loader refactor (gpu_metal.m −79.6%)

Per-family hand-cloned scaffolding replaced with a generic loader driven by an extended `struct gpu_metal_family`. New fields: `core_str`, `base_macros`, `dispatch_tg_size`, `fam_idx`. Parallel arrays `metal_family_libs[CAP][8]` + `metal_family_psos[CAP][8]` hidden in the .m so the .h stays pure C.

| Component                | Pre-refactor | Post-refactor | Delta            |
|--------------------------|-------------:|--------------:|-----------------:|
| gpu_metal.m              |       20,764 |         4,242 | −16,522 (−79.6%) |
| gpu_metal.h              |        1,002 |           432 |    −570 (−56.9%) |

47 families use `metal_pso_for_variant_default` (generic). 5 families keep custom resolvers:

- **md5salt** — PRESALT V_S|V_R fold; V_S, V_S|V_M, V_S|V_R|V_M route through generic
- **sha512cryptmd5** — aliases sha512crypt's compiled PSO at dispatch
- **hmac_streebog256_kpass / _ksalt** — dual-struct entries share one PSO via canonical fam_idx
- **hmac_streebog512_kpass / _ksalt** — same dual-struct pattern
- **bcrypt** — `dispatch_tg_size=8` struct override (replaces hardcoded `op == JOB_BCRYPT` check)

## External-failure-fatal discipline

New headers `gpu/gpu_fatal.h` + `gpu/gpu_debug.h`. Approximately 57 silent-failure sites converted across gpu_metal.m, gpu/gpujob_metal.m, gpu/gpu_opencl.c, gpu/gpujob_opencl.c. Runtime failures (PSO create, buffer alloc, dispatch error, clEnqueue errors) now call `GPU_FATAL` / `MTL_FATAL_NSERR` with file:line + op + error string and `_Exit(1)`.

Init-time eager-compile failures route through the admission-prune path (capability check, not runtime failure). Query `gpu_metal_op_variant_admitted()` exposed for host-side gpu_ops[] consumption (deferred).

## Debug emissions compile-time gated

`MDXFIND_GPU_DEBUG` macro in `gpu/gpu_debug.h`. Default builds omit:

- Per-(family, variant) JIT-compiled / PSO-created-lazily markers
- Per-run trace (`salts uploaded`, `first dispatch issued`, `buf_scratch_pool allocated`, `salt-chunked dispatch`)
- Init chatter
- `STDERR: GPU admission` summary line

Debug builds restore all markers:

```
make CFLAGS_EXTRA="-DMDXFIND_GPU_DEBUG=1" mdxfind
```

Verified via `strings | grep` that the production binary physically omits the debug strings from `.rodata`. Kept unconditional in production: `GPU_FATAL` / `MTL_FATAL_NSERR` runtime errors, device identity line, per-pruned-combo prune lines, end-of-job per-device stats.

## Breaking changes (operator-facing)

1. **Stderr format**. Operators previously grepping per-(family, variant) markers (`Metal: sha512-variant library JIT-compiled` and similar) must either rebuild with `-DMDXFIND_GPU_DEBUG=1` or re-target their grep to end-of-job per-device stats. `STDERR: GPU admission: N families admitted ...` is also debug-only now.
2. **Marker format**. When a debug build is active, generic-loader markers use the form `(generic vbits=0x... rules=N mask=N salt=N)` rather than the pre-refactor per-family fixed format. Backward-compatible substring `library JIT-compiled` still matches both.

## Platform notes

A pre-existing Apple Metal compiler bug in macOS Ventura 13.7.x affected SHA-2/512-family PSO creation on M2 Max. Fixed upstream in macOS 26.5 / Xcode 26.5. M2 Max users should upgrade for SHA-2/512-family GPU acceleration. M1 hosts on macOS 14+ are unaffected throughout.

### Intel Mac (macOS)

Metal GPU acceleration is **disabled at compile time** on Intel Mac. Apple's `MTLCompilerService` XPC daemon hangs on JIT PSO creation for AMD GCN GPUs (e.g., Radeon Pro 580X) on macOS Sequoia 15.x — confirmed on iMac and nutshack at rev 1.475. The hang is in Apple's driver and not fixable from mdxfind. Intel Mac users get CPU mode (or OpenCL if their Makefile enables it). Apple Silicon Macs (M-series) are unaffected; Metal GPU acceleration is fully supported on ARM macOS.

## Deferred

- Host-side `gpu_ops[]` consumption of the new `gpu_metal_op_variant_admitted()` query (defense-in-depth)
- mdxfind source-side `-h REGEX` runaway guard
- OpenCL shared-loader refactor parallel to Metal (estimated ~3K LOC savings)
- Phase 2e Dynsize Task #226 (long-standing backlog)
- OpenCL init-tier debug emissions (~285 ambiguous sites left as production pending adjudication)

## Files

Added:

- `gpu/gpu_fatal.h` (rev 1.1)
- `gpu/gpu_debug.h` (rev 1.1)
- `gpu/metal_*_core.metal` + paired `_str.h` for each Phase 2d.6–2d.9 family
- `gpu/codegen/cl2metal_overrides/*.yaml` for each translated family

Heavily modified:

- `gpu_metal.m` rev 1.93 → 1.100 (refactor + debug gating + admission summary)
- `gpu_metal.h` rev 1.47 → 1.49
- `gpu/gpujob_metal.m` rev 1.22 → 1.25
- `gpu/gpu_opencl.c` rev 1.169 → 1.171 (D5b FATAL conversion + debug gating)
- `gpu/gpujob_opencl.c` rev 1.137 → 1.138
- `mdxfind.c` rev 1.473 → 1.475 (Phase 2d.5.7 + 2d.7a host-wiring fixes)
- `gpu/codegen/cl2metal.py` rev 1.5 → 1.9 (pointer-state helpers, dual_addr_space_helpers overlay, _rewrite_local_uchar_casts, _FN_HEAD_RE widening, extra_scalar_ref_types overlay)
