# mdxfind v1.502 — Phase 4 + Phase 5a hx codegen (e347 + 7 MAKE_MD5PASS family GPU acceleration)

Source: mdxfind.c rev 1.501, gpu/gpu_opencl.c rev 1.194, gpu/gpu_opencl.h rev 1.41, gpu/gpujob_opencl.c rev 1.152, gpu_metal.m rev 1.117, gpu_metal.h rev 1.59, gpu/gpujob_metal.m rev 1.32, gpu/gpu_common.cl rev 1.25, gpu/metal_common.metal rev 1.24, gpu/gpu_codegen_eligible.{c,h} rev 1.1 (NEW), codegen/ tree (NEW: ~15 files, in-process hx P4 state-machine codegen), tools/hx8_to_c (NEW: build-time hx.8 → C-literal serializer).

Window: 2026-05-20 through 2026-05-23.

## Phase 4 — Production GPU dispatch for e347 (MD5(MD5(MD5(pass)).salt)) via codegen

A new in-process hx codegen pipeline produces JIT-compiled kernel B for `JOB_MD5MD5SALT` (e347) on both OpenCL (Pascal and newer) and Apple Metal. The codegen walker reads the `hx_program` bytecode for the algorithm, applies a pattern detector (`HX_PATTERN_E347_MD5MD5MD5SALT`) that recognizes the hand-tunable shape, and emits a specialized kernel source tuned to that shape (per-thread serial SALT_BATCH=64 inner loop, register-held pre-state, salt-axis amortization — the `tp0` pattern that empirically wins on Pascal salted-MD5; see `feedback_tp0_pattern_is_correct_for_pascal_salted_md5.md`).

Cross-arch byte-exact validated against the CPU oracle on 1,048,576-pair fixtures (smoke / medium / large × 2 backends):

- Apple M2 Max (dev3.local): 0.25 s end-to-end on the large fixture; zero diff
- Pascal GTX 1080 (fpga.local): 1.05 s end-to-end on the large fixture; zero diff

The hand-written `gpu_kernelb_md5md5salt_nocache.cl` (previously the e347 production path) is **retired**. The hand-written kernel carried a long-standing chain-drift bug for non-trivial salt cardinalities; codegen output is byte-exact against both the CPU implementation and `hashpipe` (independent reference). The legacy file has been deleted from the working tree (the v1.485 deletion remains; this release confirms the swap to codegen as the only production path).

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

Full 70-cell cross-arch validation matrix (7 algorithms × 5 fixtures × 2 backends) — all PASS, all byte-exact. See `codegen/tests/family_md5pass/MATRIX_RESULTS.md` in the iMac source tree for the per-cell record.

Usage: `./mdxfind -m e<N> -G 0 -F hashes -M <NAME> wordlist` selects GPU dispatch automatically for any of the 7 family members above. The `gpu_codegen_kernelb_family_md5pass_eligible()` admit-predicate helper (new file `gpu/gpu_codegen_eligible.{c,h}`) widens the chokepoint OR-chain to admit these JOBs.

Twenty-two additional MAKE_MD5PASS family members (MD2, GOST family, Haval ×15, RMD128, Tiger, Whirlpool, Snefru-256/512) remain CPU-only pending Phase 5b — their block primitives need lifting into `gpu/gpu_common.cl` first. e123 MD5MD5PASS (multi-emit canonical + colon variant) stays CPU-only until multi-emit codegen lands in a future sub-phase.

## Documentation

- `hx.8` (Appendix A of the hx manual on the iMac troff tree at `~/Documents/troff/mdxfind/hx.8`) audit and 32 doc-fix corrections: MAKE_MD5PASS family had missing concat operators in the canonical expressions; MD5MD5USER had user/pass argument transposition.
- Multi-emit families now annotated in `hx.8` Note [24] — 28 entries gained `(see Note [24])` markers identifying mdxfind's multi-output emission patterns.

The `hx.8` troff source is not shipped in this public repository; the canonical hx language manual lives in the upstream `hashpipe` distribution.

## Build / infrastructure

- New `codegen/` directory containing the in-process P4 state-machine codegen — `hx_walker.c` (state machine + bytecode dispatch), `hx_emit_opencl.c` + `hx_emit_metal.c` (per-backend emit helpers), `hx_patterns.c` (pattern detector for hand-tunable shapes), `hx_emit_primitives.c` (per-primitive outer-hash dispatch), `hx_dump.c` (env-flag source dump), and `hx_specs_data.c` (the compiled `hx_program` table, ~28 KLOC of generated C literals).
- New `tools/hx8_to_c.c` build-time tool that converts `hx.8` to `codegen/hx_specs_data.c`. Shipped here for completeness, but it requires the upstream `hashpipe` source tree to compile (it links the hx parser library). External users build directly against the pre-generated `codegen/hx_specs_data.c` checked in here; that file is regenerated on the iMac when `hx.8` changes.
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

# mdxfind v1.485 — Word-retirement ETA + GPU iter accounting + SHA512CRYPT perf

Source: mdxfind.c rev 1.485, mdxfind.h rev 1.24, gpu/gpu_common.cl rev 1.22, gpu/gpu_md5salt_core.cl rev 1.7, gpu/gpu_opencl.c rev 1.172, gpu/gpu_shacrypt_core.cl rev 1.4, gpu/gpu_template.cl rev 1.18, gpu/gpujob_opencl.c rev 1.142, gpu/gpujob_metal.m rev 1.26, gpu/metal_common.metal rev 1.21, gpu/metal_md5salt_core.metal rev 1.3, gpu/metal_shacrypt_core.metal rev 1.3.

Window: 2026-05-18 through 2026-05-19.

## Word-retirement ETA (architectural inversion)

Replaces the previous "absolute-hash-operation-rate" ETA basis, which used a denominator (total expected hash ops) that was unknowable for iterated hash types where the per-hash iteration count is parsed from the hash itself (SHA512CRYPT `rounds=N$`, BCRYPT `$2b$NN$`, PHPBB3 `$H$X`, etc.). Symptoms included "100% complete" displayed within 15 seconds of starting an hour-long run, "ETA finishing" / "ETA done" while real work continued, and inflated hash-rate displays.

New mechanism:

- Each input word is "retired" when every active hash type has completed all of its work (rules × masks × salts × internal iterations) for that word.
- Per-op `retired_line` counter, monotonic across the entire run; minimum across active ops gives `RetiredLines_now`.
- 15-second rate window: `words/sec = (RetiredLines_now − RetiredLines_lasttick) / 15`.
- ETA = `(TotalLines − RetiredLines_now) / words_per_second`.

Display now reads:

```
[T+ 15.020s] Working on rockyou.txt, w=496, line 70396, 4693.07 lines/s (0.5%), 23.46Mh/s, Found=0, ETA 50m40s
```

Immune to per-hash iteration count, salt retirement timing, rule cardinality, mask cardinality, and CPU/GPU asymmetry. Salt-retirement acceleration (per-word time shrinks as salts retire late in a run) is captured automatically because the rate window catches it as it happens.

Bootstrap fallback path for the first 15 seconds (before the first retirement-rate sample) uses the legacy hash-rate denominator and prefixes the ETA with `~` to indicate provisional.

Tradeoff documented: retirement is credited at procjob hand-off time (not actual GPU completion), so under very high rule-set or mask workloads where the GPU work queue depth is large, the displayed retirement frontier runs slightly ahead of true GPU completion. Visible mostly on rule-processed BF runs with deep queues; ETA stays accurate, percentage runs slightly optimistic.

## GPU iter-aware Tothash accounting (correctness fix)

The GPU per-dispatch hash count formula at `gpujob_opencl.c` previously multiplied (words × rules × masks × salts × external_iter) — omitting the per-algorithm **internal** iteration count entirely. For iterated GPU types this caused Tothash to undercount by the iter multiplier:

| Type | Default iter | Prior undercount |
|------|------:|------:|
| MD5CRYPT (e511) | 1000 | 1000× |
| SHA256CRYPT (e512) | 5000 | 5000× |
| SHA512CRYPT (e513) | 5000 | 5000× |
| SHA512CRYPTMD5 (e538) | 5000 | 5000× |
| BCRYPT (e450) | 2^cost | up to 2^31× |
| PHPBB3 (e455) | 2^count | up to 2^30× |
| DESCRYPT (e500) | 25 | 25× |
| SHA1DRU (e404) | 1,000,000 | 1,000,000× |

New `gpu_compute_iter_sum()` helper in both OpenCL and Metal backends parses iter at accounting time from the packed salt string (fixed-constant or `rounds=N$` / cost-char), sums across all packed salts (correctly handles mixed-iter loads where different hashes carry different `rounds=` values), and folds into the per-dispatch multiplier. Zero kernel changes; affects ~30 LOC across `gpujob_opencl.c` + `gpujob_metal.m`.

Symptom example: GTX 1080 SHA512CRYPT previously displayed `332 h/s` declining — actually ~1.66 Mh/s of SHA-512 round operations. Now matches CPU semantics.

## SHA512CRYPT performance — Steps A + C (ported from hashcat)

`sha512_block` (Step A, `gpu_common.cl` + `metal_common.metal`) — replaced the loop-based 80-round body with hashcat's flat-unrolled pattern: 16 scalar `ulong w0_t..wf_t` (no W[80] array), source-inlined steps with macro-rotated argument order, bitselect-based Ch/Maj on platforms supporting it. Affects 81 call sites across 8 SHA-512 / SHA-384 / HMAC-SHA-512 kernel files — all byte-exact regression PASS on Pascal, Maxwell, Ada, RDNA4, and Apple M1/M2 Max.

Metal twin caught one Xcode 26 toolchain regression mid-port: `bitselect()` for scalar `ulong` is not supported (only vector overloads). Switched to arithmetic Ch/Maj forms — semantically identical, no perf cost, more portable.

SHACRYPT digest chain (Step C, `gpu_shacrypt_core.cl` + `metal_shacrypt_core.metal`) — replaced the per-iter `sc_init`/`sc_update`/`sc_final` byte-RMW chain with hashcat's pre-computed `wpc[8][16]` template + 8-template boolean-cube index pattern. The mdxfind `if (r&1) / if (r%3) / if (r%7)` 4-branch sequence is algebraically identical to `pc = (r&1) + ((r%3)?2:0) + ((r%7)?4:0)` (verified by hand for all 8 cases). Replaces dozens of read-modify-write byte operations per iteration with 16 ulong copies + 1 conditional byte splice. Affects all three SHACRYPT variants (e512, e513, e538) byte-exact.

Measured on fpga GTX 1080, e513 SHA512CRYPT:
- Pre-Step-A baseline: ~2,400 H/s
- Post-Step-A: ~2,652 H/s (+10%)
- Post-Step-C: ~3,226 H/s (+34% cumulative)

Hashcat m1800 reference on the same GPU is ~12,646 H/s. Remaining gap is structural — mdxfind's `template_phase0` kernel carries rule walker, mask decomposer, B7 mask shift, and cursor logic in every dispatch; closing it would require either a SHACRYPT-specialised kernel or a state-machine codegen of the kernel per dispatch shape. Out of scope this release.

## Phase 2h MD5SALT Metal pre-roll (M-series perf)

`gpu/metal_md5salt_core.metal` — added a "pre-salt hoist" that pre-rolls the outer MD5's first 8 FF rounds (which depend only on the salt-independent hex32 of the inner MD5) once per word, saving 12.5% of outer-MD5 work per (word, salt) pair. Carrier struct shrunk from 17 uints to 13 uints. Symmetric OpenCL port (`gpu/gpu_md5salt_core.cl`) added for parity — measured zero perf delta on NVIDIA Pascal (NVCC already CSE-hoists rounds 1-8 across the SALT_BATCH loop), +14.5% wall reduction on Apple M1 (1546s → 1322s on the canonical e31 sm-saltfull rockyou benchmark).

## Phase 2g salt-refresh hybrid trigger

`gpu/gpujob_opencl.c` + `gpu/gpujob_metal.m` — extended the salt-snapshot refresh trigger from "every 10 batches" to "every 10 batches OR ≥5% of salts retired since last refresh". Symmetric across both backends. Net win when combined with Phase 2h's per-hash cost reduction; quiescent overhead when not.

## Phase 2f wordlist in-order ordering for GPU-salted ops

`mdxfind.c` — capped per-procjob `numline` request at 32K for GPU+salted-hash combinations (was `ULLONG_MAX`). Matches the jobg slot size so each procjob produces exactly one ordered slot, preserving wordlist in-order processing across the double-buffer concurrent procjob workers. Measured ~40% wall reduction on M1 dev1 e31 sm-saltfull (~2400s → 1440s).

## Wordlist throughput instrumentation

Two new stderr emits for wordlist-performance characterization, both dead code when not active:

- `linecount: <file> (N MB): scanned in Xms = Y MB/s` — fires after each cache-miss fresh scan in `linecount_thread`. On-disk byte rate, derived from stat `sb.st_size`. Cache hits skip the emit.
- `-w skip: N lines (M MB) in Xms = Y MB/s` — fires in the main wordlist loop when SkipLine transitions to 0. Bytes via `CurfileBytesRead` atomic; only fires when `-w` is active.

## Other fixes

- **Nested block comment in `gpu/gpu_template.cl`** broke NVIDIA OpenCL JIT compile (silent CPU fallback). Inner `/* */` annotation inside an outer doc block terminated the outer block early, exposing prose as code. Fixed; new memory rule documented internally.
- **Double-credit retirement on GPU completion path** — both procjob and GPU dispatch were crediting `retired_line` for the same words, causing 2× over-count visible as apparent hash-rate undercount. Removed the GPU-side credit; procjob hand-off is now the sole authority.
- **Cross-chunk retirement reset** — `retired_line` was being zeroed at every `cacheline()` chunk boundary, breaking monotonicity. Now reset only at process start (implicit via `calloc`).
- **BF servo statics restoration** — `bf_rate_ema`, `bf_chunks_produced`, `bf_first_feedback_seen`, `num_devs_v` were inadvertently deleted in 1.477 during the warning sweep, breaking .205 / .209 builds. Restored.

## Cross-platform validation

Steps A + C + iter accounting + word-retirement ETA validated byte-exact on:

- NVIDIA Pascal (GTX 1080, Linux)
- NVIDIA Maxwell (GTX 960, Linux)
- NVIDIA Ada (RTX 4070 Ti, Linux)
- AMD RDNA4 (RX 9070 / gfx1201, Linux ROCm)
- Apple M1 (Metal, macOS)
- Apple M2 Max (Metal, macOS)

Windows NVIDIA preflight intentionally deferred to natural post-release testing.

## Caveats

- Word-retirement ETA percentage runs slightly optimistic under high rules / high masks workloads where the GPU work queue depth dominates per-job time (retirement credited at procjob hand-off, not GPU completion). Architecturally correct alternative (per-job CPU-vs-GPU classification at procjob completion) is documented in the source comment at the credit site for future revisit.
- mdxfind GPU SHA512CRYPT remains ~3.9× slower than hashcat m1800 on the same hardware. Remaining gap is the `template_phase0` rule/mask/cursor scaffolding that fires per-dispatch even when not needed.
