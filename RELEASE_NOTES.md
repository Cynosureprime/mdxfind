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

All 52 families verified byte-exact CPU/Metal parity on dev1 (M1) + dev3 (M2 Max). Admission summary: 52 families admitted, 208/208 variants admitted, 0 prunes, 0 CPU-only.

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
