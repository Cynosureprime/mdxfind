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
