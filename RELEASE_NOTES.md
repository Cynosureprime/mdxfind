# mdxfind v1.504 — Phase 5b Tier 1: MD2 / RMD128 MAKE_MD5PASS family GPU acceleration + RIPEMD-128 / RIPEMD-160 standard-conformance bug fix

Source: mdxfind.c rev 1.507, rmd128.c rev 1.1 (NEW; bug fix), rmd160.c rev 1.1 (NEW; bug fix), gpu/gpu_common.cl rev 1.27, gpu/gpu_common_str.h rev 1.19, gpu/metal_common.metal rev 1.26, gpu/metal_common_str.h rev 1.5, gpu/gpu_codegen_eligible.c rev 1.3, codegen/hx_emit_primitives.c rev 1.3, codegen/hx_emit_opencl.c rev 1.11, codegen/hx_emit_metal.c rev 1.10, codegen/tests/run_validation_family_md5pass.sh rev 1.4, codegen/tests/family_md5pass/e120_smoke.txt rev 1.1 (NEW 5b.1a), codegen/tests/family_md5pass/e157_smoke.txt rev 1.1 (NEW 5b.1b), codegen/tests/test_rmd128_bosselaers.c rev 1.1 (NEW; regression), codegen/tests/test_rmd160_bosselaers.c rev 1.1 (NEW; regression).

Window: 2026-05-27.

Phase 5b Tier 1 lifts two MAKE_MD5PASS outer primitives — MD2 and RMD128 — into the shared GPU helper sources and wires them into the family codegen pipeline. After Tier 1 the family has 9 GPU-eligible members; 20 remain CPU-only pending Tier 2-4 primitive lifts (tiger, wrl, haval×15, gost, gost_crypto, sne128, sne256). During Tier 1 validation a long-standing standard-conformance bug in the in-tree RIPEMD-128 implementation was discovered and fixed; subsequent inspection found the identical bug pattern in the in-tree RIPEMD-160 implementation and that is fixed in the same release. See the "Bug fix" section below.

## Bug fix — RIPEMD-128 / RIPEMD-160 length-encoding standard-conformance

A pair of long-standing length-encoding bugs in the in-tree RIPEMD-128 (`rmd128.c`) and RIPEMD-160 (`rmd160.c`) implementations (both Bosselaers ESAT-COSIC 1996 donor lineage) are fixed this release. The bug pattern is identical in both files: the `RIPEMDxxx()` wrapper's per-block consumer loop modifies the local `len` parameter, and the final `MDfinish()` call was passing the post-loop residual byte count instead of the original total length. Because `MDfinish()` encodes its `lswlen` argument as the message bit-length in the standard MD-family length suffix, the digest of any message longer than 63 bytes was non-standard — the encoded length was `(total_bytes mod 64) * 8` instead of `total_bytes * 8`. The `MDfinish()` header docstring in `rmd128.h` and `rmd160.h` had always documented the intended semantics (`lswlen` is the TOTAL byte length; only `lswlen mod 64` bytes remain in `strptr`), so the bug was solely in the callers.

After the fix, the in-tree RIPEMD-128 and RIPEMD-160 implementations match Bosselaers's 1996 reference values byte-for-byte, including the canonical test vectors:

### RIPEMD-128 (`rmd128.c` rev 1.1)

| Input                                              | Expected (Bosselaers)              | Pre-fix mdxfind                    | Post-fix mdxfind                   |
|----------------------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| empty string                                       | `cdf26213a150dc3ecb610f18f6b38b46` | (unchanged — single-block)         | `cdf26213a150dc3ecb610f18f6b38b46` |
| `"a"`                                              | `86be7afa339d0fc7cfc785e72f578d33` | (unchanged — single-block)         | `86be7afa339d0fc7cfc785e72f578d33` |
| `"abc"`                                            | `c14a12199c66e4ba84636b0f69144c77` | (unchanged — single-block)         | `c14a12199c66e4ba84636b0f69144c77` |
| `"message digest"`                                 | `9e327b3d6e523062afc1132d7df9d1b8` | (unchanged — single-block)         | `9e327b3d6e523062afc1132d7df9d1b8` |
| `a..z` (26 bytes)                                  | `fd2aa607f71dc8f510714922b371834e` | (unchanged — single-block)         | `fd2aa607f71dc8f510714922b371834e` |
| `A..Z+a..z+0..9` (62 bytes)                        | `d1e959eb179c911faea4624c60c5c702` | (unchanged — single-block)         | `d1e959eb179c911faea4624c60c5c702` |
| `"1234567890"` × 8 (80 bytes)                      | `3f45ef194732c2dbb2c4a2c769795fa3` | `1959258deca4645654950534f3537250` | `3f45ef194732c2dbb2c4a2c769795fa3` |
| `"a"` × 1,000,000                                  | `4a7f5723f954eba1216c9d8f6320431f` | (non-conformant)                   | `4a7f5723f954eba1216c9d8f6320431f` |

All 8 RIPEMD-128 vectors PASS post-fix on the standalone regression test `codegen/tests/test_rmd128_bosselaers.c` (rev 1.1 NEW). After the fix, the in-tree `RIPEMD128()` agrees with `sph_ripemd128` (sphlib-3.0), and with every other standard implementation.

### RIPEMD-160 (`rmd160.c` rev 1.1)

| Input                                              | Expected (Bosselaers)                        | Pre-fix in-tree                              | Post-fix in-tree                             |
|----------------------------------------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|
| empty string                                       | `9c1185a5c5e9fc54612808977ee8f548b2258d31`   | (unchanged — single-block)                   | `9c1185a5c5e9fc54612808977ee8f548b2258d31`   |
| `"a"`                                              | `0bdc9d2d256b3ee9daae347be6f4dc835a467ffe`   | (unchanged — single-block)                   | `0bdc9d2d256b3ee9daae347be6f4dc835a467ffe`   |
| `"abc"`                                            | `8eb208f7e05d987a9b044a8e98c6b087f15a0bfc`   | (unchanged — single-block)                   | `8eb208f7e05d987a9b044a8e98c6b087f15a0bfc`   |
| `"message digest"`                                 | `5d0689ef49d2fae572b881b123a85ffa21595f36`   | (unchanged — single-block)                   | `5d0689ef49d2fae572b881b123a85ffa21595f36`   |
| `a..z` (26 bytes)                                  | `f71c27109c692c1b56bbdceb5b9d2865b3708dbc`   | (unchanged — single-block)                   | `f71c27109c692c1b56bbdceb5b9d2865b3708dbc`   |
| `A..Z+a..z+0..9` (62 bytes)                        | `b0e20b6e3116640286ed3a87a5713079b21f5189`   | (unchanged — single-block)                   | `b0e20b6e3116640286ed3a87a5713079b21f5189`   |
| `"1234567890"` × 8 (80 bytes)                      | `9b752e45573d4b39f4dbd3323cab82bf63326bfb`   | `5f7ffbbfd70ae0b9ad611b7961a32a7646f9c384`   | `9b752e45573d4b39f4dbd3323cab82bf63326bfb`   |
| `"a"` × 1,000,000                                  | `52783243c1697bdbe16d37f97f68f08325dc1528`   | `eb56500397007b3e6e07fe58db85a7ceaa78d37f`   | `52783243c1697bdbe16d37f97f68f08325dc1528`   |

All 8 RIPEMD-160 vectors PASS post-fix on the standalone regression test `codegen/tests/test_rmd160_bosselaers.c` (rev 1.1 NEW). After the fix, the in-tree `RIPEMD160()` agrees with `sph_ripemd160` (sphlib-3.0) and OpenSSL's `RIPEMD160`, and with every other standard implementation.

### Scope and fix shape

Both fixes are confined to the `RIPEMDxxx()` wrapper functions in `rmd128.c` and `rmd160.c`: each saves the original message length into local `total_lswlen` and `total_mswlen` before the per-block consumer loop, then passes those totals to `MDfinish()`. No signature changes; `MDfinish()` is unchanged in either file.

**Scope of RIPEMD-128 behavior change**: every `mdxfind` catalog entry that calls `RIPEMD128()` (e16 RMD128, e156 RMD128MD5, e157 RMD128MD5PASS, e210 HMAC-RMD128, e231 RMD128MD5MD5, e498 RMD128MD4, e714 SHA1RMD128) produces different output post-fix for any RIPEMD-128 input longer than 63 bytes. For inputs ≤ 63 bytes (single-block, no consumer-loop iterations) the digest is byte-identical to pre-fix.

**Scope of RIPEMD-160 behavior change**: the in-tree `RIPEMD160()` symbol is shadowed at link time by OpenSSL's `libcrypto.a` `RIPEMD160` in the `mdxfind` binary (the in-tree `rmd160.o` is only linked into the separate `mdxocl` target). The `mdxfind` binary therefore was already using a standard-conformant RIPEMD-160; the fix to `rmd160.c` restores the in-tree implementation to standard conformance for any caller that links it directly (including `mdxocl`, the standalone Bosselaers regression test, and any future build target that links `rmd160.o`). The five `mdxfind` catalog entries that mention `RIPEMD160()` in their case bodies (e17 RMD160, e158 RMD160MD5, e159 RMD160MD5PASS, e196 MD5RMD160, e746 SHA1RMD160TRUNC) and the `oracle_compute_md5pass_family()` arm for e159 all resolve `RIPEMD160` to the OpenSSL implementation in the production binary, so their behavior is unchanged. HMAC paths (e211 HMAC-RMD160, e798 HMAC-RMD160 KPASS) use the `mhash` library's `MHASH_RIPEMD160` and are likewise unchanged.

The user has confirmed (2026-05-27) that no production solved-hash archives include RIPEMD-128 or RIPEMD-160 computations on inputs longer than 60 bytes for any catalog entry, so neither fix orphans any existing cracking workflow.

The GPU codegen emit helpers (`emit_outer_rmd128_concat_then_hash` in `hx_emit_opencl.c` rev 1.11 and `emit_outer_rmd128_concat_then_hash_metal` in `hx_emit_metal.c` rev 1.10) had carried a `bug_lswlen` workaround introduced during the initial 5b.1b validation pass to keep the GPU output byte-exact with the (then-buggy) CPU oracle. That workaround is reverted this release: the length suffix in both backends now uses `total_len * 8` unconditionally, matching the now-conformant CPU oracle.

The corresponding RMD-160 GPU codegen emit helpers (`emit_outer_rmd160_concat_then_hash` / `emit_outer_rmd160_concat_then_hash_metal`) never carried a `bug_lswlen` workaround — they were always standard-correct because the production `mdxfind` binary's `RIPEMD160` was already coming from OpenSSL. No emit-helper revert is required for RMD-160; no GPU file changes for the RMD-160 fix.

## Sub-phase 5b.1a — MD2MD5PASS (e120) GPU acceleration

`e120 MD2MD5PASS` (`md2(md5_hex(pass) . pass)`) joins the 7 Phase 5a family members on GPU. Eighth GPU-eligible member of the 30-entry MAKE_MD5PASS family.

The MD2 outer-hash primitive (`md2_block`) is new to both `gpu/gpu_common.cl` and `gpu/metal_common.metal` this sub-phase. The 256-byte MD2 S-box (RFC 1319 Table T) lives in `__constant` address space on OpenCL and `constant` address space on Metal. The block compression matches B-Con and sph_md2 reference byte-for-byte (18 rounds × 48-byte state + 16-byte checksum). The emit helper (`emit_outer_md2_concat_then_hash`) is bespoke because MD2's structure diverges from the MD4/MD5 family (16-byte block, PKCS padding, checksum-as-final-block).

## Sub-phase 5b.1b — RMD128MD5PASS (e157) GPU acceleration

`e157 RMD128MD5PASS` (`rmd128(md5_hex(pass) . pass)`) becomes the ninth GPU-eligible member of the family.

The RMD-128 outer-hash primitive (`rmd128_block`) is new to both `gpu/gpu_common.cl` (4-uint state, dual pipeline, reuses RMD_F1..F4 macros from the existing RMD-160 helper, defines local RMD128_STEP / LL1..LL4 / RR1..RR4 round-K macros) and `gpu/metal_common.metal` (Metal twin via hand-port; same structure with RMD128_STEP_METAL / LL1M..LL4M / RR1M..RR4M). The dual pipeline runs **left line F1->F2->F3->F4 and right line F4->F3->F2->F1** per Bosselaers Table 4 — the right-line ordering is inverted relative to RMD-160 (which uses F5->F4->F3->F2->F1) and is the highest-risk transcription point per spec R2.

The emit helper (`emit_outer_rmd128_concat_then_hash`) clones the existing RMD-160 helper with state width adjusted from 5 uints to 4 uints. The standard-conformant length suffix (`total_len * 8`) is encoded directly; see the "Bug fix" section above for the resolution of the temporary `bug_lswlen` workaround that was carried during the initial 5b.1b validation pass.

## Validation matrix

Sub-phase 5b.1a (MD2) 10-cell Tier 1 matrix: 5 fixtures × 2 backends (OpenCL Pascal GTX 1080 + Metal Apple M2 Max) × e120 → 10/10 PASS, 2,097,792 password-digest verifications byte-exact vs CPU oracle.

Sub-phase 5b.1b (RMD128) 10-cell Tier 1 matrix: same shape × e157 → 10/10 PASS, 2,099,728 password-digest verifications byte-exact vs CPU oracle (verified with the rmd128.c bug fix and the matching emit-helper revert in place; the previously-failing `family_edge_maxlen` cell — 128 plaintexts with plens 56-128 exercising the multi-block path — now passes on both backends with zero diffs).

Aggregate 90-cell post-Tier-1 family regression (9 family members × 5 fixtures × 2 backends): 90/90 PASS. Zero regressions on the 7 Phase 5a members (e122 e159 e161 e163 e165 e167 e169); e120 and e157 admit cleanly without disturbing any of them. None of the non-RMD128 family entries are affected by the RIPEMD-128 bug fix.

Phase 4 e347 production-dispatcher regression (2 fixtures × 2 backends): 4/4 PASS, byte-exact unchanged. e347 (MD5MD5SALT) does not use RIPEMD-128, and the e347 regression confirms no collateral damage from the rmd128.c edit.

| Fixture / Backend     | OpenCL e120 | OpenCL e157 | Metal e120 | Metal e157 |
|-----------------------|-------------|-------------|------------|------------|
| family_smoke (8)      | PASS        | PASS        | PASS       | PASS       |
| family_medium (1024)  | PASS        | PASS        | PASS       | PASS       |
| family_large (1048576)| PASS        | PASS        | PASS       | PASS       |
| family_edge_minlen    | PASS        | PASS        | PASS       | PASS       |
| family_edge_maxlen    | PASS        | PASS        | PASS       | PASS       |

The `gpu_codegen_kernelb_family_md5pass_eligible()` admit-predicate widens from 7 arms (Phase 5a) to 9 arms (adding case 120 and case 157). The aggregate runner `run_validation_family_md5pass.sh` widens its `FAMILY_JOBS` list from 7 entries to 9 (numeric-sorted: `120 122 157 159 161 163 165 167 169`).

## Coming in Tier 2-4 (future releases)

Tier 2 (tiger + wrl), Tier 3 (haval × 15 variants), Tier 4 (snefru + gost + gost_crypto) ship per their corresponding `<primitive>_block` lifts into `gpu_common.cl` and `metal_common.metal`. The Phase 5b scoping memo lays out per-tier priority and per-family architect spec discipline.

---

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
